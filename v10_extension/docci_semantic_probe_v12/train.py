"""Bounded matched optimizers; full-vocabulary TRAIN-only KD; no skip-on-NaN."""
import json
import time
import torch
import torch.nn.functional as F
from docci_semantic_probe_v12.core import atomic_json, digest, indices, lr, damage_numpy, seed_for
from docci_reconstruction_v10.core import state_hash


def text_loss(model, src, kd_weight, temperature):
    from docci_reconstruction_v10.model import prediction
    from docci_reconstruction_v10.channel import clean
    from bart_latent_codec_v4.model import distillation
    output = model.receive(*clean(model.encode(src)))
    ids, hidden = model.sources(src); mask = ids.ne(model.bart.config.pad_token_id)
    # Teacher computation is never performed by the CE control or evaluation.
    teacher = None
    if kd_weight:
        with torch.no_grad(): teacher, teacher_labels = prediction(model.bart,ids,mask,hidden.float(),mask)
    logits, labels = prediction(model.bart,ids,mask,output['memory'].flatten(0,1),output['receiver_mask'].flatten(0,1))
    ce = F.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=-100)
    kd = ce.new_zeros(())
    if kd_weight:
        if not torch.equal(labels,teacher_labels): raise ValueError('Teacher/student target alignment differs')
        kd = distillation(logits,teacher,labels,temperature)
    return ce+kd_weight*kd, dict(CE=float(ce.detach()),KD=float(kd.detach()),teacher_executed=bool(kd_weight))


def checkpoint(folder, model, optimizer, c, task, arm, updates):
    state = model.saved(); path = folder/'last.pth'
    with path.open('xb') as f:
        torch.save(dict(schema='v12-checkpoint',task=task,arm=arm,state=state,config=c,updates=updates,
            optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),f)
    record = torch.load(str(path),map_location='cpu'); model.restore(record['state'])
    if state_hash(model.saved()) != state_hash(state): raise ValueError('Actual checkpoint disk roundtrip differs')
    atomic_json(folder/'CHECKPOINT.json',dict(checkpoint_sha256=digest(path),state_hashes=state_hash(state),updates=updates,
        exact_disk_roundtrip=True,optimizer_saved=True,parent_optimizer_restored=False,fixed_final_no_selection=True))


def fit(model, items, c, task, arm, device, folder, budget, backbone=None, smoke=False):
    from docci_reconstruction_v10.data import batch
    folder.mkdir(exist_ok=False); model.configure()
    if not items or any(x['official_split'] != 'train' for x in items):
        raise ValueError('Only official TRAIN may enter optimizer or teacher loss')
    active = {n:p for n,p in model.named_parameters() if p.requires_grad}
    initial = {n:p.detach().cpu().clone() for n,p in active.items()}; gradients = set(); seen = set()
    updates = 2 if smoke else c[task+'_updates']; size = c['text_batch_size'] if task == 'text' else c['image_grad_accum']
    optimizer = torch.optim.AdamW(list(active.values()),lr=c[task+'_learning_rate'],weight_decay=c['weight_decay'])
    teacher_weight = c['kd_weight'] if arm == 'B1_ce_kd' else 0.
    atomic_json(folder/'trainability.json',dict(task=task,arm=arm,active_parameters=sum(p.numel() for p in active.values()),
        active_names=sorted(active),initial_state_hashes=state_hash(model.saved()),updates=updates,batch_presentations=size,
        backbone_frozen=True,smoke_weights_NEVER_reused=smoke,optimizer='NEW AdamW',parent_optimizer_restored=False,
        loss='CE + weight*T^2*KL(teacher||student)' if task=='text' else 'pixel MSE + latent auxiliary MSE',
        kd_weight=teacher_weight,kd_temperature=c['kd_temperature'],teacher_TRAIN_only=True,
        training_example_ids=[x['example_id'] for x in items]))
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats(device) if device.type=='cuda' else None
    with (folder/'history.jsonl').open('x') as log:
        for step in range(updates):
            budget(); optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups: group['lr'] = lr(c,task,step,updates)
            ix = indices(len(items),size,step,c['seed']); ids = [items[i]['example_id'] for i in ix]; seen.update(ids)
            if task == 'text':
                loss, terms = text_loss(model,batch(items,ix,'text',device),teacher_weight,c['kd_temperature'])
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite text loss')
                loss.backward(); value = float(loss.detach()); condition = 'clean'
            else:
                condition = c['image_train_conditions'][step%len(c['image_train_conditions'])]; values=[]; pixels=[]; latents=[]
                for index in ix:
                    budget(); item = items[index]
                    observed = damage_numpy(item['wire'],condition,seed_for(c['seed'],'TRAIN',step,item['example_id']))
                    wire = torch.as_tensor(observed,device=device)[None]
                    semantic = torch.as_tensor(item['semantic'],device=device)[None]
                    reconstructed = model.receive(wire,semantic)
                    target_latent = torch.as_tensor(item['latent'],device=device)[None].float()
                    decoded = (backbone.decode(reconstructed/float(backbone.config.scaling_factor)).sample+1)/2
                    l,t,r,b = [int(v) for v in item['rectangle']]
                    target = torch.as_tensor(item['rgb'].copy(),device=device).permute(2,0,1)[None].float()/255.
                    pixel = F.mse_loss(decoded[:,:,t:b,l:r],target[:,:,t:b,l:r])
                    latent = F.mse_loss(reconstructed,target_latent)
                    loss = pixel+c['image_latent_weight']*latent
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite image loss')
                    (loss/size).backward(); values.append(float(loss.detach())); pixels.append(float(pixel.detach())); latents.append(float(latent.detach()))
                value=sum(values)/size; terms=dict(pixel_MSE=sum(pixels)/size,latent_MSE=sum(latents)/size,teacher_executed=False)
            for n,p in active.items():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all(): raise FloatingPointError('Nonfinite gradient '+n)
                    if bool(p.grad.abs().sum()>0): gradients.add(n)
            norm = torch.nn.utils.clip_grad_norm_(list(active.values()),1.,error_if_nonfinite=True); optimizer.step()
            elapsed = time.monotonic()-started
            row = dict(task=task,arm=arm,update=step+1,total_updates=updates,batch_example_ids=ids,
                unique_examples_seen=len(seen),condition=condition,loss=value,**terms,lr=optimizer.param_groups[0]['lr'],
                grad_norm=float(norm),elapsed_seconds=elapsed,eta_train_seconds=elapsed/(step+1)*(updates-step-1),
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30 if device.type=='cuda' else None)
            log.write(json.dumps(row,allow_nan=False)+'\n'); log.flush()
            if step==0 or (step+1)%20==0 or step+1==updates:
                atomic_json(folder.parent/'progress.json',row)
                print('V12 {} update={}/{} loss={:.4f} elapsed={:.1f}s ETAtrain={:.1f}s'.format(arm,step+1,updates,value,elapsed,row['eta_train_seconds']),flush=True)
    audit={}
    for group,prefixes in model.groups().items():
        names=[n for n in active if n.startswith(prefixes)]
        audit[group]=dict(parameters=sum(active[n].numel() for n in names),nonzero_gradient=any(n in gradients for n in names),
            changed=any(not torch.equal(initial[n],active[n].detach().cpu()) for n in names))
    if not all(v['parameters'] and v['nonzero_gradient'] and v['changed'] for v in audit.values()):
        atomic_json(folder/'learning_audit.json',dict(passed=False,groups=audit)); raise RuntimeError('Required codec/adapter group did not learn')
    atomic_json(folder/'learning_audit.json',dict(passed=True,groups=audit,updates=updates,unique_examples_seen=len(seen)))
    checkpoint(folder,model,optimizer,c,task,arm,updates)
    atomic_json(folder/'TRAIN_COMPLETE.json',dict(updates=updates,unique_examples_seen=len(seen),smoke=smoke))
