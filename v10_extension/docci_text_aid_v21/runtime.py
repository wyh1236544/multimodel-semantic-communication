"""Real saved-weight GPU work; no downloads, source-cache rewrite or long suite."""
import json
import math
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from docci_text_aid_v21.core import (PARENTS, EVAL_SHA, PILOT_ARMS, DIAG, CONTROLS,
    sha, read, rows, atomic, array_sha, route, order, seed, impairment, eval_controls,check_anchor)
from docci_text_guided_image_v20.data import Items, tensors
from docci_text_guided_image_v20.engine import (make, signatures, check_frozen,
    receive_text, rgb_decode, crop, objective, state_hash)
from docci_text_guided_image_v20.channel import channel

def parent_path(args, arm):
    p = Path(args.v20_root)/('train_'+arm)/'last.pth'
    if sha(p) != PARENTS[arm]:
        raise ValueError('Pinned completed V20 '+arm+' differs BEFORE torch.load')
    return p

class Selected:
    def __init__(self, cache, selected, cohort):
        self.items = Items(cache,cohort)
        by_id = {r['example_id']:i for i,r in enumerate(self.items.entries)}
        self.indices = [by_id[r['example_id']] for r in selected[cohort]]
    def __len__(self):
        return len(self.indices)
    def __getitem__(self,i):
        return self.items[self.indices[i]]

class Trace:
    """Forward-output instrumentation only; never modifies model values."""
    def __init__(self, model):
        self.values = {}
        self.handles = []
        for name in ('tx32','tx16','rx32','rx64'):
            def hook(module,inputs,output,label=name):
                image, side = inputs
                delta = output.detach()-image.detach()
                rms = float(image.detach().square().mean().sqrt())
                self.values[label] = dict(alpha=float(module.alpha.detach()),
                    tanh_alpha=float(module.alpha.detach().tanh()),
                    feature_RMS=rms,residual_RMS=float(delta.square().mean().sqrt()),
                    residual_relative_RMS=float(delta.square().mean().sqrt())/max(rms,1e-12),
                    side_is_self=side is None,
                    side_token_RMS=None if side is None else float(side.detach().square().mean().sqrt()),
                    side_token_variance=None if side is None else float(side.detach().var(1,unbiased=False).mean()))
            self.handles.append(getattr(model,name).register_forward_hook(hook))
    def close(self):
        for h in self.handles:
            h.remove()

def output_row(root,c,item,arm,control,snr,wire,observed,stats,latent,vae,lpips,
               trace=None,donor_id=None,text_stats=None,memory=None,mask=None,save=False):
    from utils.metrics import batch_image_metrics
    with torch.no_grad():
        rgb = rgb_decode(vae,latent).clamp(0,1)
        target = torch.as_tensor(item['rgb'],device=rgb.device).permute(2,0,1)[None].float()/255.
        score = batch_image_metrics(crop(item,rgb),crop(item,target))
        row = dict(arm=arm,control=control,cohort=item['cohort'],example_id=item['example_id'],
            snr_db=snr,donor_id=donor_id,psnr=float(score['psnr'][0]),ssim=float(score['ssim'][0]),
            MSE=float((crop(item,rgb)-crop(item,target)).square().mean()),
            LPIPS=float(lpips(crop(item,rgb),crop(item,target)).mean()),
            image_stats=stats,text_stats=text_stats,source_text_at_RX=False,
            post_noise_renormalization=False,tx_sha256=array_sha(wire.cpu().numpy()),
            observed_sha256=array_sha(observed.cpu().numpy()),
            mask_sha256=array_sha(impairment(c,item['example_id']) if mask is None else mask),
            received_latent_sha256=array_sha(latent.cpu().numpy()),
            source_image_sha256=item['image_sha256'],source_text_sha256=item['text_sha256'],
            memory_sha256=None if memory is None else array_sha(memory.cpu().numpy()),
            trace={} if trace is None else dict(trace.values),
            image_file=None,source_image_file=None)
        # Source occupancy is evaluated AFTER restoration, never passed as a
        # receiver mask/gate. This diagnoses empty-frame dilution only.
        occupied=np.any(~np.isin(item['ids'],(0,1,2)),axis=1)
        slot_rms=None if memory is None else memory.reshape(9,32,768).square().mean((1,2)).sqrt().cpu().numpy()
        row['empty_slots_source_audit_only']=int((~occupied).sum())
        row['empty_slot_memory_RMS']=None if slot_rms is None or occupied.all() else float(slot_rms[~occupied].mean())
        row['content_slot_memory_RMS']=None if slot_rms is None or not occupied.any() else float(slot_rms[occupied].mean())
        row['source_occupancy_passed_to_receiver']=False
        if save:
            name='{}__{}__{}__{}__{}'.format(arm,item['cohort'],item['example_id'],snr,control)
            path=root/'images'/(name+'.png')
            path.parent.mkdir(exist_ok=True)
            Image.fromarray((rgb[0].permute(1,2,0).cpu().numpy()*255).round().clip(0,255).astype(np.uint8)).save(path)
            source=root/'images'/(item['example_id']+'__source.png')
            if not source.exists():
                Image.fromarray(item['rgb']).save(source)
            row.update(image_file=path.relative_to(root).as_posix(),image_sha256=sha(path),
                source_image_file=source.relative_to(root).as_posix(),source_view_sha256=sha(source))
    return row

def memories(text,src,donor,snr,eid,did,phase):
    result = {}
    for label,wire,s,identity in (
        ('received',src['text_wire'],snr,eid),('donor',donor['text_wire'],snr,did),
        ('clean',src['text_wire'],None,eid),('clean_donor',donor['text_wire'],None,did)):
        rx,stats = receive_text(text,wire,s,0,identity,phase)
        result[label] = (rx['memory'].flatten(1,2),stats)
    return result

def anchors(args,c,root,arm,model,vae,text,lpips,device,budget):
    """Reproduce two actual archived 0dB/r0 rows before any new pilot update."""
    old=Path(args.v20_root)/('eval_'+arm)/'rows.jsonl'
    if sha(old)!=EVAL_SHA[arm]:
        raise ValueError('Archived anchor rows differ')
    references={r['example_id']:r for r in rows(old) if r['cohort']=='dev' and
                r['snr_db']==0 and r['repeat']==0 and r['control']=='matched'}
    items=Selected(args.cache,c['selected'],'dev')
    receipts=[]
    with torch.no_grad():
        for i in range(min(2,len(items))):
            budget();item=items[i];src=tensors(item,device);eid=item['example_id']
            rx,ts=receive_text(text,src['text_wire'],0,0,eid,'eval')
            wire=model.encode(src['latent'],None if arm=='A_self' else src['source_semantics'])
            observed,est,rel,stats=channel(wire,0,0,eid,'image','eval')
            memory=rx['memory'].flatten(1,2) if arm=='C_tx_rx_text' else None
            latent=model.receive(observed,memory,est,rel)
            r=output_row(root,c,item,arm,'anchor',0,wire,observed,stats,latent,vae,lpips,memory=memory)
            check_anchor(r,references[eid])
            receipts.append(dict(example_id=eid,actual=r,reference=references[eid]))
    atomic(root/('ANCHOR_'+arm+'.json'),dict(actual_saved_weight_replay=True,
        reference_rows_sha256=EVAL_SHA[arm],metric_tolerance=1e-6,exact_TX_and_observation=True,rows=receipts))
    print('V21 actual saved-parent anchors reproduced: '+arm,flush=True)

def diagnostics(args,c,root,device,budget):
    root.mkdir(exist_ok=True)
    started = time.monotonic()
    count = 0
    with (root/'rows.jsonl').open('x',encoding='utf-8') as log,torch.no_grad():
        for arm in PARENTS:
            model,vae,text,lpips = make(args,device,parent_path(args,arm))
            model.requires_grad_(False)
            initial = state_hash(model)
            frozen = signatures(vae,text,lpips)
            if c['smoke']:
                anchors(args,c,root,arm,model,vae,text,lpips,device,budget)
            trace = Trace(model)
            for cohort in ('train','dev'):
                items = Selected(args.cache,c['selected'],cohort)
                for i in range(len(items)):
                    item, donor_item = items[i],items[(i+1)%len(items)]
                    src, dr = tensors(item,device),tensors(donor_item,device)
                    eid,did = item['example_id'],donor_item['example_id']
                    for snr in c['image_snr_db']:
                        mem = memories(text,src,dr,snr,eid,did,'v21-diagnose')
                        for control in DIAG[arm]:
                            budget()
                            tx,rx = route(arm,control)
                            side = dict(self=None,source=src['source_semantics'],
                                donor=dr['source_semantics'],zero=torch.zeros_like(src['source_semantics']))[tx]
                            memory,ts = (None,None) if rx=='self' else (
                                (torch.zeros_like(mem['received'][0]),mem['received'][1]) if rx=='zero' else mem[rx])
                            wire = model.encode(src['latent'],side)
                            observed,est,rel,stats = channel(wire,snr,0,eid,'image','v21-diagnose')
                            restored = model.receive(observed,memory,est,rel)
                            r = output_row(root,c,item,arm,control,snr,wire,observed,stats,restored,
                                vae,lpips,trace,did if 'shuffled' in control else None,ts,memory,
                                save=cohort=='dev' and i<c['save_examples'])
                            r['source_semantics_RMS_difference_to_donor'] = float(
                                (src['source_semantics']-dr['source_semantics']).square().mean().sqrt())
                            r['received_memory_RMS_difference_to_donor'] = float(
                                (mem['received'][0]-mem['donor'][0]).square().mean().sqrt())
                            log.write(json.dumps(r,allow_nan=False)+'\n'); log.flush()
                            count += 1
                    if i==0 or (i+1)%8==0 or i+1==len(items):
                        atomic(root/'progress.json',dict(arm=arm,cohort=cohort,examples=i+1,
                            rows=count,elapsed_seconds=time.monotonic()-started))
                        print('V21 DIAG {} {} {}/{} rows={}'.format(arm,cohort,i+1,len(items),count),flush=True)
            trace.close()
            if state_hash(model) != initial:
                raise RuntimeError('Frozen diagnostic model changed')
            check_frozen((vae,text,lpips),frozen)
            del model,vae,text,lpips
            torch.cuda.empty_cache()
    atomic(root/'DONE.json',dict(rows=count,rows_sha256=sha(root/'rows.jsonl'),
        frozen_models_unchanged=True,training_updates=0,elapsed_seconds=time.monotonic()-started))

def pilot_evaluate(args,c,root,model,vae,text,lpips,device,budget,arm):
    items = Selected(args.cache,c['selected'],'dev')
    trace = Trace(model)
    count = 0
    with (root/'rows.jsonl').open('x',encoding='utf-8') as log,torch.no_grad():
        for i in range(len(items)):
            item, di = items[i], items[(i+1)%len(items)]
            src,dr = tensors(item,device),tensors(di,device)
            eid,did = item['example_id'],di['example_id']
            wire = model.encode(src['latent'],None)
            mask = impairment(c,eid)
            for snr in c['image_snr_db']:
                observed,est,rel,stats = channel(wire,snr,0,eid,'image','v21-pilot-eval')
                observed = observed*torch.as_tensor(mask,device=device)
                mem = memories(text,src,dr,snr,eid,did,'v21-pilot-eval')
                for control in eval_controls(arm):
                    budget()
                    name = 'clean' if control.startswith('clean') else 'received'
                    if control.endswith('shuffled'):
                        name = 'clean_donor' if name=='clean' else 'donor'
                    memory,ts = mem[name]
                    if control.endswith('zero'):
                        memory = torch.zeros_like(memory)
                    if arm == 'self':
                        memory = None
                    latent = model.receive(observed,memory,est,rel)
                    r = output_row(root,c,item,arm,control,snr,wire,observed,stats,latent,
                        vae,lpips,trace,did if 'shuffled' in control else None,ts,memory,mask,
                        save=i<c['save_examples'])
                    log.write(json.dumps(r,allow_nan=False)+'\n');log.flush();count+=1
            if i==0 or (i+1)%10==0 or i+1==len(items):
                print('V21 EVAL {} {}/{} rows={}'.format(arm,i+1,len(items),count),flush=True)
    trace.close()
    return count

def train_pilot(args,c,root,device,budget):
    root.mkdir(exist_ok=True)
    items = Selected(args.cache,c['selected'],'train')
    for arm in PILOT_ARMS:
        folder = root/arm
        folder.mkdir()
        model,vae,text,lpips = make(args,device,parent_path(args,'A_self'))
        model.requires_grad_(False)
        active = {n:p for n,p in model.named_parameters() if n.startswith(('rx32.','rx64.'))}
        for p in active.values():
            p.requires_grad_(True)
        before = state_hash(model)
        frozen = signatures(vae,text,lpips)
        if c['smoke']:
            anchors(args,c,folder,'A_self',model,vae,text,lpips,device,budget)
        opt = torch.optim.AdamW(list(active.values()),lr=c['learning_rate'],weight_decay=c['weight_decay'])
        atomic(folder/'identity.json',dict(arm=arm,config=c,parent_sha256=PARENTS['A_self'],
            trainable_parameters=sum(p.numel() for p in active.values()),trainable_names=sorted(active),
            frozen_image_TX_RX=True,only_RX_fusion_trains=True,privileged_clean_text=True,
            TX_uses_text=False,new_optimizer=True,exact_resume=False))
        started = time.monotonic()
        grad_seen = set()
        with (folder/'history.jsonl').open('x',encoding='utf-8') as log:
            for step,(epoch,batch) in enumerate(order(c),1):
                budget()
                opt.zero_grad(set_to_none=True)
                lr=c['learning_rate']*(.1+.9*(1-(step-1)/c['updates']))
                for group in opt.param_groups:
                    group['lr']=lr
                observations=[]
                for own,donor in batch:
                    budget()
                    item,di=items[own],items[donor]
                    src,dr=tensors(item,device),tensors(di,device)
                    eid=item['example_id']
                    snr=None if seed(42,'v21-train-snr',epoch,eid)%4==0 else 0
                    with torch.no_grad():
                        wire=model.encode(src['latent'],None)
                        observed,est,rel,stats=channel(wire,snr,epoch,eid,'image','v21-pilot-train')
                        mask=impairment(c,eid)
                        observed=observed*torch.as_tensor(mask,device=device)
                        rx,ts=receive_text(text,dr['text_wire'] if arm=='donor' else src['text_wire'],
                            None,epoch,di['example_id'] if arm=='donor' else eid,'v21-pilot-train')
                        memory=None if arm=='self' else rx['memory'].flatten(1,2)
                    restored=model.receive(observed,memory,est,rel)
                    loss,terms=objective(vae,lpips,item,restored,src['latent'],c)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite pilot loss')
                    (loss/len(batch)).backward()
                    observations.append(dict(example_id=eid,donor_id=di['example_id'],
                        snr_db=snr,image_stats=stats,tx_sha256=array_sha(wire.cpu().numpy()),
                        observed_sha256=array_sha(observed.cpu().numpy()),mask_sha256=array_sha(mask),
                        text_stats=ts,loss=float(loss.detach()),**terms))
                for n,p in active.items():
                    if p.grad is not None:
                        if not torch.isfinite(p.grad).all():
                            raise FloatingPointError('Nonfinite pilot gradient '+n)
                        if bool(p.grad.abs().sum()):
                            grad_seen.add(n)
                norm=torch.nn.utils.clip_grad_norm_(list(active.values()),1.,error_if_nonfinite=True)
                opt.step()
                elapsed=time.monotonic()-started
                r=dict(update=step,total_updates=c['updates'],epoch=epoch,learning_rate=lr,
                    gradient_norm=float(norm),loss=float(np.mean([o['loss'] for o in observations])),
                    observations=observations,elapsed_seconds=elapsed,ETA_this_arm_seconds=elapsed/step*(c['updates']-step))
                log.write(json.dumps(r,allow_nan=False)+'\n');log.flush()
                if step==1 or step%20==0 or step==c['updates']:
                    atomic(folder/'progress.json',r)
                    print('V21 PILOT {} {}/{} loss={:.5f} elapsed={:.1f}s ETA={:.1f}s'.format(
                        arm,step,c['updates'],r['loss'],elapsed,r['ETA_this_arm_seconds']),flush=True)
        after=state_hash(model)
        if any(before[n]!=after[n] for n in before if n not in active):
            raise RuntimeError('Frozen image codec/TX changed')
        if not any(n in grad_seen and before[n]!=after[n] for n in active):
            raise RuntimeError('No real receiver-fusion update')
        check_frozen((vae,text,lpips),frozen)
        with (folder/'last.pth').open('xb') as f:
            torch.save(dict(schema='v21-receiver-checkpoint',arm=arm,state=model.saved(),
                updates=c['updates'],config=c,parent_sha256=PARENTS['A_self']),f)
        saved=model.saved()
        model.restore(torch.load(str(folder/'last.pth'),map_location='cpu')['state'])
        if any(not torch.equal(v,model.saved()[n]) for n,v in saved.items()):
            raise ValueError('Pilot checkpoint roundtrip differs')
        atomic(folder/'TRAIN_COMPLETE.json',dict(updates=c['updates'],checkpoint_sha256=sha(folder/'last.pth'),
            frozen_image_codec_TX_unchanged=True,frozen_backbones_unchanged=True,
            changed=[n for n in active if before[n]!=after[n]],nonzero_gradients=sorted(grad_seen),
            checkpoint_roundtrip=True,training_seconds=time.monotonic()-started))
        atomic(folder/'learning_audit.json',dict(initial=before,final=after,
            frozen_backbones_before=frozen,frozen_backbones_after=signatures(vae,text,lpips)))
        model.requires_grad_(False)
        evaluation_before=state_hash(model)
        n=pilot_evaluate(args,c,folder,model,vae,text,lpips,device,budget,arm)
        if state_hash(model)!=evaluation_before:
            raise RuntimeError('Evaluation changed pilot weights')
        check_frozen((vae,text,lpips),frozen)
        atomic(folder/'DONE.json',dict(rows=n,rows_sha256=sha(folder/'rows.jsonl'),
            actual_evaluation=True,frozen_models_unchanged=True))
        del model,vae,text,lpips,opt,saved,active
        torch.cuda.empty_cache()
