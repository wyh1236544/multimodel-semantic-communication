"""Small independent prefit, paired source/RX learning and actual greedy gates."""
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from docci_longclip_joint_v8.core import atomic_json,digest
from docci_longclip_joint_v8.data import tensors
from docci_longclip_joint_v8.model import prediction,generate
from docci_longclip_joint_v8.channel import training_receive
from fso_jscc_v1.core import seed_for
from docci_longclip_joint_v8.metrics import metrics,summarize


def save_checkpoint(folder,model,c,stage,updates):
    path=folder/'last.pth'; tmp=folder/'last.pth.part'
    with tmp.open('xb') as f: torch.save(dict(schema='docci-longclip-joint-v8',config=c,stage=stage,arm=model.arm,updates=updates,state=model.saved()),f)
    tmp.replace(path)
    loaded=torch.load(str(path),map_location='cpu')
    before=model.saved(); model.restore(loaded['state'])
    if set(before)!=set(model.saved()) or any(not torch.equal(v,model.saved()[n]) for n,v in before.items()): raise RuntimeError('Checkpoint roundtrip changed weights')
    atomic_json(folder/'TRAIN_COMPLETE.json',dict(checkpoint_sha256=digest(path),stage=stage,arm=model.arm,updates=updates,
        actual_roundtrip_verified=True,quality_not_implied=True))
    return before


def crop_mse(predicted,target,rectangle):
    l,t,r,b=[int(v) for v in rectangle]
    return F.mse_loss(predicted[:,:,t:b,l:r],target[:,:,t:b,l:r])


def fit(model,items,c,models,device,root,stage,steps,budget):
    root.mkdir(exist_ok=False); model.configure(model.arm,stage if stage in ('text','image') else 'joint')
    bart,vae,_=models; active={n:p for n,p in model.named_parameters() if p.requires_grad}
    initial={n:p.detach().cpu().clone() for n,p in active.items()}; gradients=set()
    groups=[]
    for prefix,lr in (('text.',c['text_learning_rate'] if stage=='text' else c['joint_text_learning_rate']),('',c['learning_rate'])):
        chosen=[p for n,p in active.items() if n.startswith('text.')==bool(prefix)]
        if chosen: groups.append(dict(params=chosen,lr=lr))
    optimizer=torch.optim.AdamW(groups,weight_decay=.01); optimizer.zero_grad(set_to_none=True)
    start=time.monotonic(); updates=0
    with (root/'history.jsonl').open('x') as log:
        for step in range(steps):
            budget(); epoch,position=divmod(step,len(items))
            order=np.random.default_rng(seed_for(c['seed'],'v8-order',epoch)).permutation(len(items))
            item=items[int(order[position])]; source=tensors(item,device); ce=source['latent'].new_zeros(()); pixel=ce; common=ce; details=None
            if stage=='text':
                wire=model.text.transmit(source['bart_hidden'].float(),source['ids'].ne(bart.config.pad_token_id)); output=model.text.receive_text(wire)
                logits,labels=prediction(bart,source['ids'],source['ids'].ne(bart.config.pad_token_id),output['memory'],output['receiver_mask'])
                ce=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),ignore_index=-100); loss=ce
            elif stage=='image':
                wire,_=model.image_encoder(source['latent'].float())
                h=model.image_decoder.features(wire,wire.new_full((1,),40.),wire.new_ones(1)); recovered=model.image_decoder.output(h)
                latent=F.mse_loss(recovered,source['latent'].float())
                decoded=(vae.decode(recovered/float(vae.config.scaling_factor)).sample+1)/2
                target=torch.as_tensor(item['rgb'].copy(),device=device).permute(2,0,1)[None].float()/255.
                pixel=crop_mse(decoded,target,item['rectangle']); loss=latent+c['pixel_weight']*pixel
            else:
                wire=model.encode(source)
                rng=np.random.default_rng(seed_for(c['seed'],'v8-anchor',step,item['example_id']))
                noisy=stage=='fso' and rng.random()>=c['clean_anchor_weight']
                observed,available,snr,rel,details=training_receive(wire,c,step,item['example_id'],noisy)
                out=model.receive(observed,available,snr,rel)
                logits,labels=prediction(bart,source['ids'],source['ids'].ne(bart.config.pad_token_id),out['text']['memory'],out['text']['receiver_mask'])
                ce=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),ignore_index=-100)
                latent=F.mse_loss(out['latent'],source['latent'].float())
                decoded=(vae.decode(out['latent']/float(vae.config.scaling_factor)).sample+1)/2
                target=torch.as_tensor(item['rgb'].copy(),device=device).permute(2,0,1)[None].float()/255.
                pixel=crop_mse(decoded,target,item['rectangle'])
                common=(1-F.cosine_similarity(out['image_semantic'],source['clip_image'].float(),dim=-1)).mean()
                common=common+(1-F.cosine_similarity(out['text_semantic'],source['clip_summary'].float(),dim=-1)).mean()
                common=common+F.mse_loss(out['common_coarse'],F.adaptive_avg_pool2d(source['latent'].float(),(8,8)))
                anchor=ce.new_zeros(())
                if 'text_anchor' in item:
                    # Warm private codec memory is a TRAIN loss target, never an RX input.
                    anchor=F.mse_loss(out['text']['memory'],torch.as_tensor(item['text_anchor'],device=device)[None])
                loss=ce+latent+c['pixel_weight']*pixel+c['common_weight']*common+.01*anchor
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss; no skipped update')
            divisor=min(c['grad_accum'],steps-(step//c['grad_accum'])*c['grad_accum']); (loss/divisor).backward()
            for name,p in active.items():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all(): raise FloatingPointError('Nonfinite gradient '+name)
                    if bool(p.grad.abs().sum()>0): gradients.add(name)
            if (step+1)%c['grad_accum']==0 or step+1==steps:
                torch.nn.utils.clip_grad_norm_(list(active.values()),1.,error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); updates+=1
            elapsed=time.monotonic()-start
            record=dict(stage=stage,arm=model.arm,step=step+1,total_steps=steps,updates=updates,example_id=item['example_id'],
                loss=float(loss.detach()),ce=float(ce.detach()),pixel_mse=float(pixel.detach()),common=float(common.detach()),proxy=details,
                elapsed_seconds=elapsed,eta_train_seconds=elapsed/(step+1)*(steps-step-1),
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30 if device.type=='cuda' else None)
            log.write(json.dumps(record,allow_nan=False)+'\n'); log.flush()
            if step==0 or (step+1)%20==0:
                atomic_json(root.parent.parent/'progress.json',record)
                print('V8 {} {} {}/{} loss={:.4f} CE={:.4f} elapsed={:.1f}s ETAtrain={:.1f}s'.format(model.arm,stage,step+1,steps,record['loss'],record['ce'],elapsed,record['eta_train_seconds']),flush=True)
    prefixes=['text.compressor.','text.memory_projection.','text.reconstructor.'] if stage=='text' else ['image_encoder.','image_decoder.'] if stage=='image' else [
        'text.compressor.','text.memory_projection.','image_encoder.','image_decoder.','common_attention.','common_tx.','common_rx.','image_fusion.','text_fusion.']
    audit={}
    for prefix in prefixes:
        names=[n for n in active if n.startswith(prefix)]
        audit[prefix]=dict(parameters=sum(active[n].numel() for n in names),nonzero_gradient=any(n in gradients for n in names),
            changed=any(not torch.equal(initial[n],active[n].detach().cpu()) for n in names))
    passed=all(v['parameters'] and v['nonzero_gradient'] and v['changed'] for v in audit.values())
    atomic_json(root/'learning_audit.json',dict(passed=bool(passed),groups=audit,pretrained_backbone_training=False))
    if not passed: raise RuntimeError('Required communication group did not learn; preserve audit')
    return save_checkpoint(root,model,c,stage,updates)


@torch.no_grad()
def text_controls(model,items,tok,device,budget,path,scope):
    memories=[]
    for item in items:
        budget(); s=tensors(item,device)
        memories.append(model.text.receive_text(model.text.transmit(s['bart_hidden'].float(),s['ids'].ne(tok.pad_token_id))))
    groups={k:[] for k in ('matched','zero','shuffled')}
    with Path(path).open('x',encoding='utf-8') as f:
        for i,item in enumerate(items):
            for control in groups:
                budget(); own=memories[i]; memory=own['memory'] if control=='matched' else torch.zeros_like(own['memory']) if control=='zero' else memories[(i+1)%len(items)]['memory']
                result=generate(model,dict(own,memory=memory),tok)
                row=dict(example_id=item['example_id'],source_split=item['official_split'],source_text=item['text'],covered_text=item['text'],
                    input_truncated=False,control=control,received_caption=result['text'],stopped_by_capacity=result['stopped_by_capacity'],
                    receiver_failure=result['receiver_failure'],scope=scope,**metrics(item['text'],item['text'],result['text']))
                groups[control].append(row); f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); f.flush()
    return {k:summarize(v) for k,v in groups.items()}


def text_gate(groups,c):
    matched=groups['matched']['full_wer']; margins={k:groups[k]['full_wer']-matched for k in ('zero','shuffled')}
    return dict(passed=matched<=c['text_wer_gate'] and min(margins.values())>=c['memory_margin_gate'] and groups['matched']['capacity_stop_fraction']==0,
        full_wer=matched,memory_margins=margins,natural_eos_fraction=1-groups['matched']['capacity_stop_fraction'],
        limits=dict(wer=c['text_wer_gate'],margin=c['memory_margin_gate']),scope='engineering prefit validation, NOT paper quality proof')
