"""Pure CE additional phase; global data cursor, new AdamW, fixed last."""
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from docci_reconstruction_v10.data import batch
from docci_reconstruction_v10.model import prediction
from docci_reconstruction_v10.channel import clean
from docci_text_continue_v11.core import atomic_json,digest,choose_indices,phase_lr,state_hash


def save(folder,model,optimizer,phase,schedule,global_update,c):
    folder.mkdir(exist_ok=False); file=folder/'last.pth'; state=model.saved()
    record=dict(schema='docci-text-continue-v11-checkpoint',state=state,optimizer=optimizer.state_dict(),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        numpy_rng=np.random.get_state(),phase=phase,schedule=schedule,global_update=global_update,config=c,
        weight_only_parent=True,parent_optimizer_NOT_restored=True)
    with file.open('xb') as f: torch.save(record,f)
    loaded=torch.load(str(file),map_location='cpu'); model.restore(loaded['state'])
    if state_hash(state)!=state_hash(model.saved()) or loaded['global_update']!=global_update: raise ValueError('Actual checkpoint roundtrip differs')
    atomic_json(folder/'CHECKPOINT.json',dict(checkpoint_sha256=digest(file),global_update=global_update,
        actual_disk_roundtrip=True,state_hashes=state_hash(state),optimizer_and_RNG_saved=True))


def fit(model,items,oldc,phase,schedule,device,folder,budget,callback,smoke=False):
    folder.mkdir(exist_ok=False); model.configure()
    active={n:p for n,p in model.named_parameters() if p.requires_grad}; initial={n:p.detach().cpu().clone() for n,p in active.items()}
    grads=set(); seen=set(); optimizer=torch.optim.AdamW(list(active.values()),lr=phase['phase_learning_rate'],weight_decay=oldc['weight_decay'])
    updates=phase['smoke_updates'] if smoke else schedule['additional_updates']; offset=schedule['parent_updates']; size=oldc['text_batch_size']
    atomic_json(folder/'trainability.json',dict(parameters=sum(p.numel() for p in active.values()),
        frozen_BART=True,loss='pure cross entropy',optimizer='NEW AdamW',parent_optimizer_restored=False,
        updates=updates,batch=size,parent_global_update=offset,smoke=smoke))
    started=time.monotonic()
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    with (folder/'history.jsonl').open('x') as log:
        for local in range(updates):
            budget(); optimizer.zero_grad(set_to_none=True); global_update=offset+local+1
            lr=phase_lr(phase,local,updates)
            for group in optimizer.param_groups: group['lr']=lr
            ix=choose_indices(len(items),size,global_update-1,0,oldc['seed'],1)
            ids=[items[i]['example_id'] for i in ix]; seen.update(ids)
            src=batch(items,ix,'text',device); output=model.receive(*clean(model.encode(src)))
            target,_=model.sources(src); mask=target.ne(model.bart.config.pad_token_id)
            logits,labels=prediction(model.bart,target,mask,output['memory'].flatten(0,1),output['receiver_mask'].flatten(0,1))
            loss=F.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=-100)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss; no silent skip')
            loss.backward()
            for n,p in active.items():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all(): raise FloatingPointError('Nonfinite gradient '+n)
                    if bool(p.grad.abs().sum()>0): grads.add(n)
            torch.nn.utils.clip_grad_norm_(list(active.values()),1.,error_if_nonfinite=True); optimizer.step()
            elapsed=time.monotonic()-started
            row=dict(local_update=local+1,global_update=global_update,total_additional_updates=updates,
                batch_example_ids=ids,unique_examples_seen=len(seen),loss=float(loss.detach()),lr=lr,
                cumulative_presentations=global_update*size,phase_presentations=(local+1)*size,
                elapsed_including_probe_seconds=elapsed,smoke=smoke,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None)
            log.write(json.dumps(row,allow_nan=False)+'\n'); log.flush()
            if local==0 or (local+1)%20==0:
                atomic_json(folder.parent/'progress.json',dict(stage='training',**row))
                if smoke:
                    print('SMOKE update={}/{} CE={:.4f} (discarded, not main epoch)'.format(local+1,updates,row['loss']),flush=True)
                else:
                    print('TRAIN global={}/{} local={}/{} epochs={:.3f} CE={:.4f} elapsed={:.1f}s'.format(global_update,
                        schedule['target_updates'],local+1,updates,global_update*size/len(items),row['loss'],elapsed),flush=True)
            if not smoke and (local+1)%phase['probe_every']==0 and local+1<updates:
                save(folder.parent/('checkpoint_'+str(global_update)),model,optimizer,phase,schedule,global_update,oldc)
                callback(global_update)
    audit={}
    for group,prefixes in model.groups().items():
        names=[n for n in active if n.startswith(prefixes)]
        audit[group]=dict(parameters=sum(active[n].numel() for n in names),nonzero_gradient=any(n in grads for n in names),
            changed=any(not torch.equal(initial[n],active[n].detach().cpu()) for n in names))
    passed=all(v['parameters'] and v['nonzero_gradient'] and v['changed'] for v in audit.values())
    atomic_json(folder/'learning_audit.json',dict(passed=bool(passed),groups=audit,updates=updates,unique_examples_seen=len(seen)))
    if not passed: raise RuntimeError('Required codec group did not learn; evidence retained')
    final=offset+updates; save(folder.parent/('checkpoint_'+str(final)),model,optimizer,phase,schedule,final,oldc)
    atomic_json(folder/'TRAIN_COMPLETE.json',dict(updates=updates,global_update=final,loss='pure CE',weights_only_warm_start=True))
