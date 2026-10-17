"""Image-led training. V18 text codec and all pretrained models stay frozen."""
import json
import math
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from docci_text_guided_image_v20.core import (ARMS,IMAGE_SHA,atomic,sha,read,seed,
    batches,train_snr,trial_matrix,GEOMETRY,array_sha)
from docci_text_guided_image_v20.data import parents,text_model,tensors,Items
from docci_text_guided_image_v20.channel import channel
from docci_text_guided_image_v20.model import GuidedImage,select_sides
from docci_text_guided_image_v20.perceptual import Perceptual,ASSETS

def make(args,device,checkpoint=None):
    from docci_modal_v9.backbones import load,historical_backbone
    torch.manual_seed(42)
    model = GuidedImage().to(device).eval()
    spec = parents(args)['image']
    parent = torch.load(spec['path'],map_location='cpu')
    if (parent.get('schema'),parent.get('task'),parent.get('arm'),parent.get('updates')) != (
            'v14-checkpoint','image','I2_shifted',5120):
        raise ValueError('Expected completed expanded V14 image model')
    model.base.restore(parent['state'])
    if checkpoint is not None:
        receipt = read(Path(checkpoint).parent/'TRAIN_COMPLETE.json')
        if sha(checkpoint) != receipt['checkpoint_sha256']:
            raise ValueError('Actual trained checkpoint differs')
        saved = torch.load(str(checkpoint),map_location='cpu')
        if saved.get('schema') != 'v20-image-checkpoint' or saved['image_parent_sha256'] != IMAGE_SHA:
            raise ValueError('Wrong V20 checkpoint')
        model.restore(saved['state'])
    vae,_,identity = load('image',args,device)
    historical_backbone('image',identity['files'])
    text = text_model(args,device)
    lpips = Perceptual(args.lpips_assets).to(device)
    return model,vae,text,lpips

def signatures(vae,text,lpips):
    from docci_longclip_joint_v8.backbones import signature
    return [signature(m) for m in (vae,text,lpips)]

def check_frozen(models,before):
    from docci_longclip_joint_v8.backbones import signature
    if before != [signature(m) for m in models] or any(
            p.requires_grad or p.grad is not None for m in models for p in m.parameters()):
        raise RuntimeError('Frozen VAE/BART/V18/LPIPS changed')

def receive_text(text,wire,snr,epoch,eid,phase,repeat=0):
    with torch.no_grad():
        observed,_,_,stats = channel(wire,snr,epoch,eid,'text',phase,repeat)
        rx = text.receive(observed)
    return rx,stats

def rgb_decode(vae,latent):
    return (vae.decode(latent/float(vae.config.scaling_factor)).sample+1)/2

def crop(item,rgb):
    l,t,r,b = map(int,item['rectangle'])
    return rgb[:,:,t:b,l:r]

def objective(vae,lpips,item,received,source_latent,c):
    rgb = rgb_decode(vae,received)
    target = torch.as_tensor(item['rgb'],device=rgb.device).permute(2,0,1)[None].float()/255.
    pixel = F.mse_loss(crop(item,rgb),crop(item,target))
    latent = F.mse_loss(received,source_latent)
    perception = lpips(crop(item,rgb.clamp(0,1)),crop(item,target)).mean()
    total = pixel+c['latent_weight']*latent+c['lpips_weight']*perception
    return total,dict(pixel_MSE=float(pixel.detach()),latent_MSE=float(latent.detach()),
                      LPIPS=float(perception.detach()))

def state_hash(model):
    return {n:array_sha(p.detach().cpu().contiguous().numpy()) for n,p in model.named_parameters()}

def save_checkpoint(path,model,optimizer,arm,c,step,cache_digest):
    with path.open('xb') as f:
        torch.save(dict(schema='v20-image-checkpoint',arm=arm,config=c,state=model.saved(),
            optimizer=optimizer.state_dict(),updates=step,image_parent_sha256=IMAGE_SHA,
            cache_index_sha256=cache_digest,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state()),f)

def fit(args,c,root,device,budget,smoke):
    root.mkdir(exist_ok=False)
    items = Items(args.cache,'train')
    if any(r['official_split']!='train' or r['cohort']!='train' for r in items.entries):
        raise ValueError('Optimizer may access only selected TRAIN')
    model,vae,text,lpips = make(args,device)
    before = signatures(vae,text,lpips)
    initial = state_hash(model)
    active = dict(model.named_parameters())
    total = 2 if smoke else math.ceil(len(items)/c['accumulation'])*c['epochs']
    atomic(root/'identity.json',dict(arm=args.arm,config=c,smoke=smoke,updates=total,
        train_ids=[r['example_id'] for r in items.entries],cache_index_sha256=sha(Path(args.cache)/'index.json'),
        initial_hashes=initial,trainable_parameters=sum(p.numel() for p in active.values()),
        frozen_models=['VAE','BART','Long-CLIP source cache','V18 text codec','LPIPS'],
        image_TX_and_RX_trainable=True,new_optimizer=True,checkpoint_policy='fixed final',
        exact_resume=False,geometry=GEOMETRY,
        loss='pixel_MSE + {} latent_MSE + {} LPIPS'.format(c['latent_weight'],c['lpips_weight']),
        perceptual_assets=ASSETS,receiver_gold_source=False))
    optimizer = torch.optim.AdamW(list(active.values()),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    grad_seen = set()
    seen = set()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    with (root/'history.jsonl').open('x',encoding='utf-8') as log:
        for step,(epoch,indices) in enumerate(batches(len(items),c)):
            if step>=total:
                break
            budget()
            optimizer.zero_grad(set_to_none=True)
            lr = c['learning_rate']*min(1.,(step+1)/min(200,max(1,total//10)))*(.1+.9*(1-step/total))
            for g in optimizer.param_groups:
                g['lr'] = lr
            loss_value = 0.
            observations = []
            for i in indices:
                budget()
                item = items[i]
                eid = item['example_id']
                seen.add(eid)
                src = tensors(item,device)
                snr = train_snr(c,epoch,eid)
                rx,ts = receive_text(text,src['text_wire'],snr,epoch,eid,'train')
                side_tx,side_rx = select_sides(args.arm,src['source_semantics'],rx['memory'].flatten(1,2))
                wire = model.encode(src['latent'],side_tx)
                observed,estimate,reliability,ims = channel(wire,snr,epoch,eid,'image','train')
                latent = model.receive(observed,side_rx,estimate,reliability)
                loss,terms = objective(vae,lpips,item,latent,src['latent'],c)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite image objective')
                (loss/len(indices)).backward()
                loss_value += float(loss.detach())/len(indices)
                observations.append(dict(example_id=eid,snr_db=snr,image=ims,text=ts,**terms))
            for n,p in active.items():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        raise FloatingPointError('Nonfinite gradient '+n)
                    if bool(p.grad.abs().sum()):
                        grad_seen.add(n)
            norm = torch.nn.utils.clip_grad_norm_(list(active.values()),1.,error_if_nonfinite=True)
            optimizer.step()
            elapsed = time.monotonic()-started
            row = dict(update=step+1,total_updates=total,epoch=epoch+1,loss=loss_value,
                learning_rate=lr,gradient_norm=float(norm),observations=observations,
                elapsed_seconds=elapsed,ETA_this_arm_seconds=elapsed/(step+1)*(total-step-1),
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
            log.write(json.dumps(row,allow_nan=False)+'\n')
            log.flush()
            if step==0 or (step+1)%c['log_every']==0 or step+1==total:
                atomic(root/'progress.json',row)
                print('V20 {} {}/{} epoch={} loss={:.5f} elapsed={:.1f}s ETA_this_arm={:.1f}s'.format(
                    args.arm,step+1,total,epoch+1,loss_value,elapsed,row['ETA_this_arm_seconds']),flush=True)
            if (step+1)%math.ceil(len(items)/c['accumulation'])==0 and not smoke:
                save_checkpoint(root/('epoch_%03d.pth'%(epoch+1)),model,optimizer,args.arm,c,step+1,
                                sha(Path(args.cache)/'index.json'))
    audits = {}
    final = state_hash(model)
    for group,prefixes in model.groups().items():
        names = [n for n in active if n.startswith(prefixes)]
        audits[group] = dict(parameters=sum(active[n].numel() for n in names),
            nonzero_gradient=any(n in grad_seen for n in names),
            changed=any(initial[n]!=final[n] for n in names))
    if not all(v['parameters'] and v['nonzero_gradient'] and v['changed'] for v in audits.values()):
        raise RuntimeError('Image TX/RX/fusion learning audit failed')
    check_frozen((vae,text,lpips),before)
    atomic(root/'learning_audit.json',dict(passed=True,groups=audits,frozen_before=before,frozen_after=signatures(vae,text,lpips)))
    save_checkpoint(root/'last.pth',model,optimizer,args.arm,c,total,sha(Path(args.cache)/'index.json'))
    saved = model.saved()
    model.restore(torch.load(str(root/'last.pth'),map_location='cpu')['state'])
    if any(not torch.equal(v,model.saved()[n]) for n,v in saved.items()):
        raise ValueError('Saved final model roundtrip differs')
    atomic(root/'TRAIN_COMPLETE.json',dict(updates=total,unique_ids_seen=len(seen),
        checkpoint_sha256=sha(root/'last.pth'),actual_disk_roundtrip=True,
        fixed_final=True,training_seconds=time.monotonic()-started))
    if smoke:
        atomic(root/'timing_estimate.json',dict(measured_smoke_seconds_per_update=(time.monotonic()-started)/total,
            warning='Includes initialization/audits; main has different I/O; not a performance guarantee'))

def evaluate(args,c,root,device,budget,smoke):
    from utils.metrics import batch_image_metrics
    from docci_baseline_v14.model import decode
    from docci_text_continue_v11.metrics import metrics
    from transformers import BartTokenizer
    root.mkdir(exist_ok=False)
    (root/'images').mkdir()
    (root/'frames').mkdir()
    model,vae,text,lpips = make(args,device,args.checkpoint)
    model.requires_grad_(False)
    before = state_hash(model)
    frozen = signatures(vae,text,lpips)
    tok = BartTokenizer.from_pretrained(args.bart_path,local_files_only=True)
    matrix = list(trial_matrix(c,args.arm,smoke))
    atomic(root/'identity.json',dict(arm=args.arm,smoke=smoke,config=c,
        checkpoint_sha256=sha(args.checkpoint),cache_index_sha256=sha(Path(args.cache)/'index.json'),
        trials=[list(x) for x in matrix],geometry=GEOMETRY,
        primary_cohort='dev',qual_dev_exploratory=True,text_branch_unchanged=True,
        LPIPS='Torch-only Alex v0.1, original valid rectangle',facts='PENDING',
        oracle_privileged=True,oracle_not_deployment=True))
    text_rows = []
    count = 0
    started = time.monotonic()
    with (root/'rows.jsonl').open('x',encoding='utf-8') as log,torch.no_grad():
        for cohort in ('dev','qual_dev'):
            items = Items(args.cache,cohort)
            for snr,repeat,control in matrix:
                for i in range(len(items)):
                    budget()
                    item = items[i]
                    eid = item['example_id']
                    src = tensors(item,device)
                    rx,ts = receive_text(text,src['text_wire'],snr,0,eid,'eval',repeat)
                    side_tx,side_rx = select_sides(args.arm,src['source_semantics'],rx['memory'].flatten(1,2))
                    wire = model.encode(src['latent'],side_tx)
                    observed,estimate,reliability,ims = channel(wire,snr,0,eid,'image','eval',repeat)
                    donor = None
                    delivered = ts
                    if control=='rx_shuffled':
                        donor_item = items[(i+1)%len(items)]
                        donor = donor_item['example_id']
                        dr = tensors(donor_item,device)
                        donor_rx,delivered = receive_text(text,dr['text_wire'],snr,0,donor,'eval',repeat)
                        side_rx = donor_rx['memory'].flatten(1,2)
                    elif control=='rx_zero':
                        side_rx = torch.zeros_like(side_rx)
                    elif control=='rx_oracle':
                        side_rx = src['oracle_memory']
                    restored = rgb_decode(vae,model.receive(observed,side_rx,estimate,reliability)).clamp(0,1)
                    target = torch.as_tensor(item['rgb'],device=device).permute(2,0,1)[None].float()/255.
                    score = batch_image_metrics(crop(item,restored),crop(item,target))
                    mse = float((crop(item,restored)-crop(item,target)).square().mean())
                    value = float(lpips(crop(item,restored),crop(item,target)).mean())
                    row = dict(example_id=eid,cohort=cohort,official_split=item['official_split'],
                        arm=args.arm,snr_db=snr,repeat=repeat,control=control,donor_id=donor,
                        psnr=float(score['psnr'][0]),ssim=float(score['ssim'][0]),LPIPS=value,MSE=mse,
                        source_image_sha256=item['image_sha256'],source_text_sha256=item['text_sha256'],
                        evaluation_rectangle=item['rectangle'].tolist(),
                        image_stats=ims,text_stats=ts,delivered_text_stats=delivered,
                        tx_image_sha256=array_sha(wire.cpu().numpy()),
                        observed_image_sha256=array_sha(observed.cpu().numpy()),
                        source_text_at_RX=control=='rx_oracle',text_memory_at_RX=control,
                        image_file=None,source_image_file=None,frame_file=None)
                    if i<c['save_examples'] and repeat==0:
                        name='{}__{}__snr{}__{}'.format(cohort,eid,'clean' if snr is None else snr,control)
                        out = (restored[0].permute(1,2,0).cpu().numpy()*255).round().clip(0,255).astype(np.uint8)
                        f = root/'images'/(name+'.png')
                        Image.fromarray(out).save(f)
                        source_file = root/'images'/(cohort+'__'+eid+'__source.png')
                        if not source_file.exists():
                            Image.fromarray(item['rgb']).save(source_file)
                        row.update(image_file=f.relative_to(root).as_posix(),image_sha256=sha(f),
                            source_image_file=source_file.relative_to(root).as_posix(),
                            source_view_sha256=sha(source_file))
                        fp = root/'frames'/(name+'.npz')
                        with fp.open('xb') as handle:
                            np.savez(handle,tx=wire.cpu().numpy(),observed=observed.cpu().numpy())
                        row.update(frame_file=fp.relative_to(root).as_posix(),frame_sha256=sha(fp))
                    # Decode the unchanged text branch ONCE, in C only. These are
                    # actual received texts, NOT outputs improved by images.
                    if args.arm=='C_tx_rx_text' and control=='matched' and repeat==0:
                        generated = decode(text.bart,rx,tok,'greedy')
                        text_rows.append(dict(example_id=eid,cohort=cohort,snr_db=snr,
                            source_text=item['text'],source_text_sha256=item['text_sha256'],
                            image_assisted_text=False,**generated,
                            **metrics(item['text'],generated['received_caption'])))
                    log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                    log.flush()
                    count += 1
                print('V20 EVAL {} {} snr={} r={} control={} finished'.format(
                    args.arm,cohort,snr,repeat,control),flush=True)
                atomic(root/'progress.json',dict(rows=count,elapsed_seconds=time.monotonic()-started,
                    cohort=cohort,snr_db=snr,control=control))
    with (root/'text_rows.jsonl').open('x',encoding='utf-8') as f:
        for row in text_rows:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    if state_hash(model)!=before:
        raise RuntimeError('Evaluation changed image model')
    check_frozen((vae,text,lpips),frozen)
    atomic(root/'EVAL_COMPLETE.json',dict(rows=count,rows_sha256=sha(root/'rows.jsonl'),
        text_rows=len(text_rows),text_rows_sha256=sha(root/'text_rows.jsonl'),
        actual_neural_evaluation=True,unchanged=True,elapsed_seconds=time.monotonic()-started))
