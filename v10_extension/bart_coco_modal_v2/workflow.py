"""Mode-local clean training, actual text controls, frozen and forward audits."""
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from bart_coco_modal_v2 import VERSION
from bart_coco_modal_v2.core import seed_for,tensor_signature,metrics,summarize
from bart_coco_modal_v2.data import source_wire
from bart_coco_modal_v2.channel import clean,papr
from bart_clip_joint_v1.workflow import frozen_signatures,verify_frozen


def text_row(item,mode,condition,result):
    return dict(image_id=item['image_id'],annotation_id=item['annotation_id'],mode=mode,condition=condition,
        source_sha256=item['source_sha256'],source_split=item['split'],source_text=item['text'],
        covered_text=item['covered_text'],received_caption=result['text'],
        input_truncated=item['truncated'],stopped_by_capacity=result['stopped_by_capacity'],
        source_utf8_bytes=item['source_utf8_bytes'],**metrics(item['text'],item['covered_text'],result['text']))


@torch.no_grad()
def text_controls(model,items,tokenizer,device,budget,directory,label,atomic_json):
    memories=[]
    for item in items:
        budget(); x=source_wire(model,item,device)
        memories.append(model.receive_payload(*clean(x))['text_memory'])
    if len(items)<2: raise ValueError('Two distinct sources required')
    rows=[]
    with (directory/(label+'_controls.jsonl')).open('x',encoding='utf-8') as f:
        for control in ('matched','zero','shuffled'):
            for i,item in enumerate(items):
                budget(); memory=memories[i] if control=='matched' else torch.zeros_like(memories[i]) if control=='zero' else memories[(i+1)%len(items)]
                output=model.generate(memory,tokenizer)
                row=text_row(item,model.modality,'clean',output)
                row.update(control=control,memory_sha256=tensor_signature(memory),diagnostic_private=model.private_diagnostic)
                rows.append(row); f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); f.flush()
    groups={k:summarize([r for r in rows if r['control']==k]) for k in ('matched','zero','shuffled')}
    atomic_json(directory/(label+'_controls.summary.json'),groups)
    return groups


def consistency(model,item,device,cfg,budget,directory,atomic_json):
    budget(); x=source_wire(model,item,device); a=model.receive_payload(*clean(x))
    with torch.no_grad():
        xx=source_wire(model,item,device); b=model.receive_payload(*clean(xx))
    keys=('text_memory',) if model.modality=='text_only' else ('latent',) if model.modality=='image_only' else ('text_memory','latent')
    errors=dict(source=float((x.detach()-xx).abs().max()))
    errors.update({k:float((a[k].detach()-b[k]).abs().max()) for k in keys})
    passed=all(math.isfinite(v) and v<=cfg['forward_tolerance'] for v in errors.values())
    atomic_json(directory/'forward_consistency.json',dict(passed=passed,errors=errors,tolerance=cfg['forward_tolerance'],
        actual_pretrained_reference=True,checks='same weights/inputs in grad-enabled vs no-grad forward; no changes to kernels'))
    if not passed: raise RuntimeError('Grad/no-grad forward mismatch; preserve audit, do not interpret as dataset failure')
    del x,xx,a,b


def train(model,items,cfg,cc,device,directory,stage,budget,atomic_json,atomic_checkpoint,vae=None):
    directory.mkdir(exist_ok=False); model.eval(); consistency(model,items[0],device,cfg,budget,directory,atomic_json)
    active={n:p for n,p in model.named_parameters() if p.requires_grad}
    frozen=frozen_signatures(model); before=model.bridge_state()
    opt=torch.optim.AdamW(list(active.values()),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    diagnostic=stage=='diagnostic'; total=cfg['diagnostic_updates'] if diagnostic else len(items)*cfg['clean_epochs']
    epochs=math.ceil(total/len(items)); step=updates=0; gradients=set(); started=time.monotonic()
    opt.zero_grad(set_to_none=True)
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    atomic_json(directory/'trainability.json',dict(mode=model.modality,private_diagnostic=model.private_diagnostic,
        active_parameters=sum(p.numel() for p in active.values()),active_names=list(active),
        all_bridge_registered_parameters=sum(v.numel() for v in before.values()),
        Bart_CLIP_VAE_historical_frozen=True,no_noisy_training_in_this_source_dataset_screen=True))
    with (directory/'history.jsonl').open('x',encoding='utf-8') as f:
        for epoch in range(epochs):
            order=np.random.default_rng(seed_for(cfg['seed'],epoch,'bart-order')).permutation(len(items))
            for index in order:
                if step>=total: break
                budget(); item=items[int(index)]; wire=source_wire(model,item,device)
                out=model.receive_payload(*clean(wire))
                zero=wire.new_zeros(()); ce=feature=image_loss=pixel_loss=alignment=zero
                warmup=diagnostic and model.modality!='image_only' and step<cfg['feature_warmup_updates']
                if model.modality!='image_only':
                    mask=item['mask'].to(device); memory=out['text_memory']
                    feature=((memory-item['bart_hidden'].to(device)).square().mean(-1)*mask).sum()/mask.sum().clamp_min(1)
                    if not warmup: ce=model.teacher_loss(item['input_ids'].to(device),mask,memory)
                if model.modality!='text_only':
                    if vae is None: raise ValueError('Frozen VAE image loss needs actual decoder')
                    image_loss=F.mse_loss(out['latent'],item['latent'].to(device))
                    scale=float(getattr(vae.config,'scaling_factor',.18215))
                    image=(vae.decode(out['latent']/scale).sample+1)/2
                    pixel_loss=F.mse_loss(image,item['image'].to(device))
                if not model.private_diagnostic:
                    teacher=item['clip_image'].to(device) if model.modality=='image_only' else item['clip_text'].to(device) if model.modality=='text_only' else F.normalize(item['clip_image'].to(device)+item['clip_text'].to(device),dim=-1)
                    alignment=(1-F.cosine_similarity(out['common_alignment'],teacher)).mean()
                loss=ce+cfg['feature_weight']*feature+cfg['image_latent_weight']*image_loss+cfg['image_pixel_weight']*pixel_loss
                loss=loss+cfg['common_alignment_weight']*alignment
                if not model.private_diagnostic: loss=loss+cfg['papr_weight']*papr(wire,cc)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss, no skipped update')
                divisor=min(cfg['grad_accum'],total-(step//cfg['grad_accum'])*cfg['grad_accum'])
                (loss/divisor).backward(); step+=1
                for n,p in active.items():
                    if p.grad is not None:
                        if not torch.isfinite(p.grad).all(): raise FloatingPointError('Nonfinite gradient '+n)
                        if bool(p.grad.abs().sum()>0): gradients.add(n)
                if step%cfg['grad_accum']==0 or step==total:
                    torch.nn.utils.clip_grad_norm_(list(active.values()),1.,error_if_nonfinite=True)
                    opt.step(); opt.zero_grad(set_to_none=True); updates+=1
                elapsed=time.monotonic()-started
                row=dict(step=step,total_steps=total,optimizer_updates=updates,phase='feature_warmup' if warmup else stage,
                    mode=model.modality,private_diagnostic=model.private_diagnostic,ce=float(ce.detach()),feature_mse=float(feature.detach()),
                    image_latent_mse=float(image_loss.detach()),image_pixel_mse=float(pixel_loss.detach()),loss=float(loss.detach()),
                    elapsed_seconds=elapsed,estimated_remaining_train_seconds=elapsed/step*(total-step),
                    cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None,
                    cuda_peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30 if device.type=='cuda' else None)
                f.write(json.dumps(row,allow_nan=False)+'\n'); f.flush()
                if step==1 or step%20==0 or step==total:
                    atomic_json(directory.parent/'progress.json',dict(stage=directory.name,**row))
                    print('MODAL {} {} {}/{} CE={:.4f} imageMSE={:.5f} elapsed={:.1f}s ETAtrain={:.1f}s'.format(model.modality,stage,step,total,row['ce'],row['image_pixel_mse'],elapsed,row['estimated_remaining_train_seconds']),flush=True)
    after=model.bridge_state(); audit={}
    for group,prefixes in model.learning_groups().items():
        names=[n for n in active if n.startswith(prefixes)]
        audit[group]=dict(parameters=len(names),nonzero_gradient=any(n in gradients for n in names),changed=any(not torch.equal(before[n],after[n]) for n in names))
    verify_frozen(model,frozen)
    passed=all(v['parameters'] and v['nonzero_gradient'] and v['changed'] for v in audit.values())
    atomic_json(directory/'learning_audit.json',dict(passed=bool(passed),groups=audit,frozen_unchanged=True,
        active_parameters=sum(p.numel() for p in active.values()),actual_nonzero_gradient_parameters=sum(active[n].numel() for n in gradients)))
    atomic_checkpoint(directory/'last_bridge.pth',dict(version=VERSION,mode=model.modality,private_diagnostic=model.private_diagnostic,
        stage=stage,config=cfg,weights=after,precise_resume_supported=False))
    atomic_json(directory/'TRAIN_COMPLETE.json',dict(steps=step,optimizer_updates=updates,learning_audit_passed=bool(passed),wall_seconds=time.monotonic()-started))
    if not passed: raise RuntimeError('Required mode groups did not learn; inspect preserved audit')
    return after
