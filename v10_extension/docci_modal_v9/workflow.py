"""Actual optimizer-update budget, old P0 recipe and independent branch audits."""
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from docci_modal_v9.core import atomic_json,digest
from docci_modal_v9.data import batch
from docci_modal_v9.model import prediction
from docci_modal_v9.channel import clean,noisy
from fso_jscc_v1.core import seed_for


def choose_indices(count,size,update,micro,seed,accumulation=1):
    start=(update*accumulation+micro)*size
    result=[]
    for cursor in range(start,start+size):
        order=np.random.default_rng(seed_for(seed,'v9-order',cursor//count)).permutation(count)
        result.append(int(order[cursor%count]))
    return result


def task(model,source,output,items,indices,backbone,c):
    if model.branch=='text':
        mask=source['ids'].ne(model.bart.config.pad_token_id)
        logits,labels=prediction(model.bart,source['ids'],mask,output['memory'],output['receiver_mask'])
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
    return latent+c['pixel_weight']*pixel,latent,pixel


def save(folder,model,c,stage,updates):
    path=folder/'last.pth'; temp=path.with_name(path.name+'.part')
    state=model.saved()
    with temp.open('xb') as f: torch.save(dict(version='docci-modal-v9',branch=model.branch,stage=stage,config=c,updates=updates,state=state,optimizer_resumed=False),f)
    temp.replace(path); loaded=torch.load(str(path),map_location='cpu')
    if loaded['config']!=c or loaded['branch']!=model.branch: raise ValueError('Actual saved identity differs')
    model.restore(loaded['state']); restored=model.saved()
    if any(not torch.equal(v,restored[n]) for n,v in state.items()): raise RuntimeError('Actual disk checkpoint roundtrip differs')
    atomic_json(folder/'TRAIN_COMPLETE.json',dict(checkpoint_sha256=digest(path),stage=stage,branch=model.branch,optimizer_updates=updates,
        exact_disk_roundtrip=True,quality_not_implied=True))
    return state


def fit(model,items,c,backbone,device,folder,stage,updates,budget,on_snapshot=None):
    folder.mkdir(exist_ok=False); model.configure(); active={n:p for n,p in model.named_parameters() if p.requires_grad}
    initial={n:p.detach().cpu().clone() for n,p in active.items()}; gradients=set()
    continuation=stage in ('clean_continue','fso_continue'); lr=c['continue_lr'] if continuation else c[model.branch+'_learning_rate']
    optimizer=torch.optim.AdamW(list(active.values()),lr=lr,weight_decay=c['weight_decay'])
    size=min(c['text_batch_size'],len(items)) if model.branch=='text' else 1
    accumulation=1 if model.branch=='text' else c['image_grad_accum']
    indices=list(range(size)); source=batch(items,indices,model.branch,device)
    wire=model.encode(source); output=model.receive(*clean(wire))
    with torch.no_grad(): again=model.encode(source); rx=model.receive(*clean(again))
    key='memory' if model.branch=='text' else 'latent'
    errors=dict(wire=float((wire.detach()-again).abs().max()),receiver=float((output[key].detach()-rx[key]).abs().max()))
    if max(errors.values())>.0002: raise RuntimeError('Actual grad/no-grad interface differs')
    atomic_json(folder/'forward_consistency.json',dict(passed=True,errors=errors,tolerance=.0002))
    atomic_json(folder/'trainability.json',dict(active_parameters=sum(p.numel() for p in active.values()),optimizer_updates=updates,
        microbatch_size=size,gradient_accumulation=accumulation,example_presentations=updates*size*accumulation,
        branch=model.branch,stage=stage,backbone_frozen=True,opposite_modality_loaded=False))
    del source,wire,output,again,rx
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    start=time.monotonic()
    with (folder/'history.jsonl').open('x') as log:
        for update in range(updates):
            budget(); optimizer.zero_grad(set_to_none=True)
            warmup=min(c['lr_warmup_updates'],max(1,updates//4)); fraction=min(1.,(update+1)/warmup)*max(.1,1-update/updates)
            for group in optimizer.param_groups: group['lr']=lr*fraction
            measures=[]; channel_records=[]; batch_ids=[]
            for micro in range(accumulation):
                indices=choose_indices(len(items),size,update,micro,c['seed'],accumulation); source=batch(items,indices,model.branch,device)
                ids=[items[i]['example_id'] for i in indices]; batch_ids+=ids
                wire=model.encode(source); output=model.receive(*clean(wire)); loss,main,pixel=task(model,source,output,items,indices,backbone,c)
                consistency=loss.new_zeros(())
                if continuation:
                    target=torch.as_tensor(np.stack([items[i]['clean_teacher'] for i in indices]),device=device)
                    if stage=='fso_continue':
                        y,available,snr,rel,stats=noisy(wire,c,update,ids)
                        damaged=model.receive(y,available,snr,rel); damaged_loss,main,pixel=task(model,source,damaged,items,indices,backbone,c)
                        consistency=F.mse_loss(damaged[key],target)
                        loss=(1-c['clean_anchor_weight'])*damaged_loss+c['clean_anchor_weight']*loss
                        channel_records+=stats
                    else: consistency=F.mse_loss(output[key],target)
                loss=loss+c['memory_consistency_weight']*consistency
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite branch loss; no skipped update')
                (loss/accumulation).backward()
                measures.append([float(loss.detach()),float(main.detach()),float(pixel.detach()),float(consistency.detach())])
            for name,p in active.items():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all(): raise FloatingPointError('Nonfinite branch gradient '+name)
                    if bool(p.grad.abs().sum()>0): gradients.add(name)
            torch.nn.utils.clip_grad_norm_(list(active.values()),1.,error_if_nonfinite=True); optimizer.step()
            elapsed=time.monotonic()-start; avg=np.mean(measures,0)
            record=dict(branch=model.branch,stage=stage,update=update+1,total_updates=updates,batch_example_ids=batch_ids,
                microbatch_size=size,gradient_accumulation=accumulation,cumulative_example_presentations=(update+1)*size*accumulation,
                loss=float(avg[0]),ce_or_latent_mse=float(avg[1]),pixel_mse=float(avg[2]),memory_consistency=float(avg[3]),lr=optimizer.param_groups[0]['lr'],proxy=channel_records,
                elapsed_seconds=elapsed,eta_train_seconds=elapsed/(update+1)*(updates-update-1),
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30 if device.type=='cuda' else None)
            log.write(json.dumps(record,allow_nan=False)+'\n'); log.flush()
            if update==0 or (update+1)%20==0:
                atomic_json(folder.parent/'progress.json',record)
                print('PRIVATE {} {} update={}/{} loss={:.4f} elapsed={:.1f}s ETAtrain={:.1f}s'.format(model.branch,stage,update+1,updates,record['loss'],elapsed,record['eta_train_seconds']),flush=True)
            if on_snapshot and ((update+1)%c['snapshot_every']==0 or update+1==updates): on_snapshot(update+1)
    audit={}
    for group,prefixes in model.groups().items():
        names=[n for n in active if n.startswith(prefixes)]
        audit[group]=dict(parameters=sum(active[n].numel() for n in names),nonzero_gradient=any(n in gradients for n in names),
            changed=any(not torch.equal(initial[n],active[n].detach().cpu()) for n in names))
    passed=all(v['parameters'] and v['nonzero_gradient'] and v['changed'] for v in audit.values())
    atomic_json(folder/'learning_audit.json',dict(passed=bool(passed),groups=audit,actual_optimizer_updates=updates))
    if not passed: raise RuntimeError('Required independent codec group did not learn; keep evidence')
    return save(folder,model,c,stage,updates)
