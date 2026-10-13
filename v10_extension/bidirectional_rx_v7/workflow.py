"""Frozen source/private receivers, differentiable new RX + BART/VAE losses."""
import json
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from bidirectional_rx_v7 import VERSION
from bidirectional_rx_v7.core import array_hash, digest, mask_policy
from clip_bart_joint_v5.data import clean_args
from clip_bart_joint_v6.model import masked_wire
from bart_latent_codec_v4.channel import proxy_tensor
from bart_latent_codec_v4.model import prediction
from bart_latent_codec_v4.workflow import frozen_check
from bart_clip_joint_v1.core import tensor_signature
from bidirectional_rx_v7.features import receiver_features


def args_for(wire,tc,c,step,image_id,phase):
    stats=[]
    if phase=='fso' and step%4!=0:
        wire,stats=proxy_tensor(wire,tc,step,[image_id]); pilot=wire.new_tensor([stats[0]['pilot_snr_db']])
        args=(wire,torch.ones(wire.shape[0],dtype=torch.bool,device=wire.device),pilot,
            torch.sigmoid(pilot.clamp(-40,40)*math.log(10)/10))
    else: args=clean_args(wire)
    policy=mask_policy(c,step,image_id)
    return (masked_wire(args[0],policy),)+args[1:],stats,policy


def train(adapter,base,items,c,tc,vae,scaling,clip,processor,device,folder,phase,epochs,budget,frozen,atomic_json,atomic_checkpoint):
    folder.mkdir(parents=True,exist_ok=False); total=len(items)*epochs; initial=adapter.saved()
    named=dict(adapter.named_parameters()); gradients=set(); optimizer=torch.optim.AdamW(named.values(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    optimizer.zero_grad(set_to_none=True); updates=0; start=time.monotonic(); adapter.eval()
    item=items[0]
    with torch.no_grad(): own=receiver_features(base.receive(*clean_args(item['wire'].to(device))),vae,scaling,clip,processor)
    out=adapter(own,own['latent'].new_full((1,),40),own['latent'].new_ones(1))
    with torch.no_grad(): y=adapter(own,own['latent'].new_full((1,),40),own['latent'].new_ones(1))
    errors={k:float((out['latent']-y['latent']).abs().max()) for k in ('image',)}
    errors['text']=float((out['text']['memory']-y['text']['memory']).abs().max())
    if max(errors.values())>c['forward_tolerance']: raise RuntimeError('Grad/no-grad adapter differs')
    atomic_json(folder/'forward_consistency.json',dict(errors=errors,tolerance=c['forward_tolerance']))
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    with (folder/'history.jsonl').open('x') as log:
        for step in range(total):
            budget(); epoch,pos=divmod(step,len(items))
            if pos==0: order=np.random.default_rng(c['seed']+epoch).permutation(len(items))
            item=items[int(order[pos])]; wire=item['wire'].to(device)
            args,stats,policy=args_for(wire,tc,c,step,item['image_id'],phase)
            with torch.no_grad(): own=receiver_features(base.receive(*args),vae,scaling,clip,processor)
            out=adapter(own,args[2],args[3])
            logits,labels=prediction(base.text.bart,item['input_ids'].to(device),item['mask'].to(device),out['text']['memory'],out['text']['receiver_mask'])
            ce=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),ignore_index=-100)
            latent=F.mse_loss(out['latent'],item['latent'].to(device))
            pixel=(vae.decode(out['latent']/scaling).sample+1)/2
            pix=F.mse_loss(pixel,item['image'].to(device))
            # Frozen clean received memory as a TRAIN anchor, never an RX input.
            anchor=F.mse_loss(out['text']['memory'],item['text_anchor'].to(device))
            loss=ce+c['image_latent_weight']*latent+c['image_pixel_weight']*pix+c['text_anchor_weight']*anchor
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite adapter loss')
            divisor=min(c['grad_accum'],total-(step//c['grad_accum'])*c['grad_accum']); (loss/divisor).backward()
            for n,parameter in named.items():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all(): raise FloatingPointError('Nonfinite gradient '+n)
                    if bool(parameter.grad.abs().sum()>0): gradients.add(n)
            if (step+1)%c['grad_accum']==0 or step+1==total:
                torch.nn.utils.clip_grad_norm_(list(named.values()),1.,error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); updates+=1
            elapsed=time.monotonic()-start
            r=dict(step=step+1,total_steps=total,optimizer_updates=updates,image_id=item['image_id'],phase=phase,
                loss=float(loss.detach()),ce=float(ce.detach()),image_mse=float(latent.detach()),pixel_mse=float(pix.detach()),text_anchor=float(anchor.detach()),
                wire_sha256=array_hash(wire.detach().cpu().numpy()),observed_sha256=array_hash(args[0].detach().cpu().numpy()),
                pilot_snr_db=float(args[2][0]),reliability=float(args[3][0]),private_mask=policy,proxy=stats,
                noisy=bool(stats),elapsed_seconds=elapsed,eta_train_seconds=elapsed/(step+1)*(total-step-1),
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30 if device.type=='cuda' else None)
            log.write(json.dumps(r,allow_nan=False)+'\n'); log.flush()
            if step==0 or (step+1)%20==0 or step+1==total:
                atomic_json(folder.parents[1]/'progress.json',dict(arm=adapter.arm,**r))
                print('RX {} {} {}/{} CE={:.4f} image={:.5f} elapsed={:.1f}s ETAtrain={:.1f}s'.format(adapter.arm,phase,step+1,total,r['ce'],r['image_mse'],elapsed,r['eta_train_seconds']),flush=True)
    saved=adapter.saved(); audit={}
    for group,prefixes in adapter.groups().items():
        ns=[n for n in named if n.startswith(prefixes)]
        audit[group]=dict(tensors=len(ns),nonzero_gradient=any(n in gradients for n in ns),changed=any(not torch.equal(initial[n],saved[n]) for n in ns))
    passed=all(v['tensors'] and v['nonzero_gradient'] and v['changed'] for v in audit.values())
    for module,signatures in frozen: frozen_check(module,signatures)
    atomic_json(folder/'learning_audit.json',dict(passed=passed,groups=audit,frozen_unchanged=True,
        registered_parameters=sum(p.numel() for p in named.values()),gradient_parameters=sum(named[n].numel() for n in gradients),
        comparable_registered_size_not_equal_active_compute=True))
    if not passed: raise RuntimeError('Required adapter groups did not learn; inspect audit')
    atomic_checkpoint(folder/'last.pth',dict(version=VERSION,arm=adapter.arm,phase=phase,config=c,weights=saved,precise_resume_supported=False))
    with torch.no_grad(): before=adapter(own,args[2],args[3])
    loaded=torch.load(str(folder/'last.pth'),map_location='cpu'); adapter.restore(loaded['weights'])
    with torch.no_grad(): after=adapter(own,args[2],args[3])
    if any(not torch.equal(v,adapter.saved()[n]) for n,v in saved.items()): raise RuntimeError('Disk reload weights differ')
    err=max(float((before['latent']-after['latent']).abs().max()),float((before['text']['memory']-after['text']['memory']).abs().max()))
    if err>c['forward_tolerance']: raise RuntimeError('Disk reload outputs differ')
    atomic_json(folder/'reload_audit.json',dict(passed=True,output_max_error=err,actual_disk_reload=True))
    atomic_json(folder/'TRAIN_COMPLETE.json',dict(steps=total,optimizer_updates=updates,checkpoint_sha256=digest(folder/'last.pth'),wall_seconds=time.monotonic()-start,
        fixed_last_checkpoint=True,quality_not_implied=True,initial_hashes={n:tensor_signature(v) for n,v in initial.items()}))
    return saved
