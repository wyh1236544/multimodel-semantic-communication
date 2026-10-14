"""Matched real example presentations; pixel-first loss, explicit gradient audits."""
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from docci_reconstruction_v10.core import atomic_json,digest,choose_indices,objective
from docci_reconstruction_v10.data import batch
from docci_reconstruction_v10.model import prediction
from docci_reconstruction_v10.channel import clean,noisy


def task(model,source,output,items,indices,backbone,c):
    if model.branch=='text':
        ids,_=model.sources(source); mask=ids.ne(model.bart.config.pad_token_id)
        memory=output['memory'].flatten(0,1); receiver_mask=output['receiver_mask'].flatten(0,1)
        logits,labels=prediction(model.bart,ids,mask,memory,receiver_mask)
        ce=F.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=-100)
        return ce,ce,ce.new_zeros(())
    latent=F.mse_loss(output['latent'],source['latent'].float())
    decoded=(backbone.decode(output['latent']/float(backbone.config.scaling_factor)).sample+1)/2
    pixels=[]
    for i,index in enumerate(indices):
        item=items[index]; l,t,r,b=[int(v) for v in item['rectangle']]
        target=torch.as_tensor(item['rgb'].copy(),device=decoded.device).permute(2,0,1).float()/255.
        pixels.append(F.mse_loss(decoded[i,:,t:b,l:r],target[:,t:b,l:r]))
    pixel=torch.stack(pixels).mean()
    return objective(c,model.arm,latent,pixel),latent,pixel


def save(folder,model,c,stage,updates):
    state=model.saved(); file=folder/'last.pth'; temp=file.with_name('last.pth.part')
    with temp.open('xb') as f: torch.save(dict(version='docci-reconstruction-v10',arm=model.arm,branch=model.branch,config=c,stage=stage,updates=updates,state=state),f)
    temp.replace(file); loaded=torch.load(str(file),map_location='cpu')
    if loaded['config']!=c or loaded['arm']!=model.arm or loaded['updates']!=updates: raise ValueError('Disk checkpoint identity differs')
    model.restore(loaded['state'])
    if any(not torch.equal(v,model.saved()[k]) for k,v in state.items()): raise RuntimeError('Actual saved tensor roundtrip differs')
    atomic_json(folder/'TRAIN_COMPLETE.json',dict(checkpoint_sha256=digest(file),arm=model.arm,stage=stage,optimizer_updates=updates,exact_disk_roundtrip=True))
    return state


def fit(model,items,c,backbone,device,folder,stage,updates,budget,on_snapshot=None):
    folder.mkdir(exist_ok=False); model.configure(); active={n:p for n,p in model.named_parameters() if p.requires_grad}
    initial={n:p.detach().cpu().clone() for n,p in active.items()}; gradients=set(); seen=set()
    continuation=stage in ('clean_continue','fso_continue'); lr=c['continue_lr'] if continuation else c[model.branch+'_learning_rate']
    optimizer=torch.optim.AdamW(list(active.values()),lr=lr,weight_decay=c['weight_decay'])
    size=min(c['text_batch_size'],len(items)) if model.branch=='text' else 1
    accum=1 if model.branch=='text' else c['image_grad_accum']; key='memory' if model.branch=='text' else 'latent'
    src=batch(items,list(range(size)),model.branch,device); wire=model.encode(src); out=model.receive(*clean(wire))
    with torch.no_grad(): again=model.encode(src); rx=model.receive(*clean(again))
    errors=dict(wire=float((wire.detach()-again).abs().max()),receiver=float((out[key].detach()-rx[key]).abs().max()))
    if max(errors.values())>.0002: raise RuntimeError('Actual grad/no-grad interface differs')
    atomic_json(folder/'forward_consistency.json',dict(passed=True,errors=errors,tolerance=.0002))
    atomic_json(folder/'trainability.json',dict(arm=model.arm,stage=stage,active_parameters=sum(p.numel() for p in active.values()),
        optimizer_updates=updates,microbatch_size=size,gradient_accumulation=accum,example_presentations=updates*size*accum,
        training_examples=[i['example_id'] for i in items],backbone_frozen=True))
    del src,wire,out,again,rx
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    started=time.monotonic()
    with (folder/'history.jsonl').open('x') as log:
        for update in range(updates):
            budget(); optimizer.zero_grad(set_to_none=True)
            warmup=min(c['lr_warmup_updates'],max(1,updates//4)); fraction=min(1.,(update+1)/warmup)*max(.1,1-update/updates)
            for group in optimizer.param_groups: group['lr']=lr*fraction
            measures=[]; proxies=[]; ids_all=[]; probes=[]
            for micro in range(accum):
                indices=choose_indices(len(items),size,update,micro,c['seed'],accum)
                ids=[items[i]['example_id'] for i in indices]; seen.update(ids); ids_all+=ids
                src=batch(items,indices,model.branch,device); wire=model.encode(src); out=model.receive(*clean(wire))
                loss,main,pixel=task(model,src,out,items,indices,backbone,c); consistency=loss.new_zeros(())
                if model.branch=='image' and (update==0 or (update+1)%100==0):
                    # Weighted gradients AT reconstructed latent, not a guessed value-fraction or all-parameter ratio.
                    gm=torch.autograd.grad(main,out['latent'],retain_graph=True)[0]
                    gp=torch.autograd.grad(pixel,out['latent'],retain_graph=True)[0]
                    lm,pm=(1.,c['pixel_weight']) if model.arm=='I0_latent' else (c['latent_aux_weight'],1.)
                    probes.append(dict(latent_weighted_gradient_norm=float(gm.norm()*lm),pixel_weighted_gradient_norm=float(gp.norm()*pm),
                        scope='gradient of each weighted loss at received latent; NOT total parameter-gradient ratio'))
                if continuation:
                    target=torch.as_tensor(np.stack([items[i]['clean_teacher'] for i in indices]),device=device)
                    if stage=='fso_continue':
                        y,ok,snr,rel,stats=noisy(wire,c,update,ids); damaged=model.receive(y,ok,snr,rel)
                        damaged_loss,main,pixel=task(model,src,damaged,items,indices,backbone,c)
                        loss=(1-c['clean_anchor_weight'])*damaged_loss+c['clean_anchor_weight']*loss
                        consistency=F.mse_loss(damaged[key],target); proxies+=stats
                    else: consistency=F.mse_loss(out[key],target)
                loss=loss+c['memory_consistency_weight']*consistency
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss; no silently skipped update')
                (loss/accum).backward(); measures.append([float(loss.detach()),float(main.detach()),float(pixel.detach()),float(consistency.detach())])
            for n,p in active.items():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all(): raise FloatingPointError('Nonfinite gradient '+n)
                    if bool(p.grad.abs().sum()>0): gradients.add(n)
            torch.nn.utils.clip_grad_norm_(list(active.values()),1.,error_if_nonfinite=True); optimizer.step()
            avg=np.mean(measures,0); elapsed=time.monotonic()-started
            row=dict(arm=model.arm,stage=stage,update=update+1,total_updates=updates,batch_example_ids=ids_all,unique_examples_seen=len(seen),
                microbatch_size=size,gradient_accumulation=accum,cumulative_example_presentations=(update+1)*size*accum,
                loss=float(avg[0]),ce_or_latent_mse=float(avg[1]),pixel_mse=float(avg[2]),consistency=float(avg[3]),loss_gradient_probes=probes,
                lr=optimizer.param_groups[0]['lr'],proxy=proxies,elapsed_seconds=elapsed,eta_train_seconds=elapsed/(update+1)*(updates-update-1),
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30 if device.type=='cuda' else None)
            log.write(json.dumps(row,allow_nan=False)+'\n'); log.flush()
            if update==0 or (update+1)%20==0:
                atomic_json(folder.parent/'progress.json',row)
                print('DOCCI {} {} update={}/{} unique={} loss={:.4f} elapsed={:.1f}s ETAtrain={:.1f}s'.format(model.arm,stage,update+1,updates,len(seen),row['loss'],elapsed,row['eta_train_seconds']),flush=True)
            if on_snapshot and (update+1)%c['snapshot_every']==0 and update+1<updates: on_snapshot(update+1)
    audit={}
    for group,prefixes in model.groups().items():
        names=[n for n in active if n.startswith(prefixes)]
        audit[group]=dict(parameters=sum(active[n].numel() for n in names),nonzero_gradient=any(n in gradients for n in names),changed=any(not torch.equal(initial[n],active[n].detach().cpu()) for n in names))
    passed=all(v['parameters'] and v['nonzero_gradient'] and v['changed'] for v in audit.values())
    atomic_json(folder/'learning_audit.json',dict(passed=bool(passed),groups=audit,actual_optimizer_updates=updates,unique_examples_seen=len(seen)))
    if not passed: raise RuntimeError('Required independent codec group did not learn; logs retained')
    return save(folder,model,c,stage,updates)
