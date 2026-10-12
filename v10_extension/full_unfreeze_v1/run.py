"""Paired VAE-tail vs ALL registered parameters eligible; live end-to-end FSO training."""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXT))
from full_unfreeze_v1 import VERSION, ARMS
from full_unfreeze_v1.core import load, digest, seed_for, paths, transport, schedule, environment, TEMPLATE
from fso_fluctuating_v2.calibrate import read_bank
from full_unfreeze_v1.core import check_bank, check_parent, visual_positions


def main():
    import torch
    import torch.nn.functional as F
    import v10.model as module
    from v10.runtime import add_shared_args, preflight, load_config
    from v10.io import begin_run, atomic_json, atomic_checkpoint
    from v10.data import build_loader
    from v10.model import build_model
    from utils.checkpoint import load_checkpoint
    from utils.experiment import seed_everything
    from utils.metrics import batch_image_metrics, ssim_per_image, token_accuracy, sentence_bleu2
    from token_guidance.model import Guidance
    from final_compare_v1.metrics import caption_metrics
    from optical_realism_v3.constellation import constellation_svg
    from eval_v10 import save_image
    from fso_jscc_v1.model import parameters as semantic_parameters
    from full_unfreeze_v1.model import candidates, stage, gradients, apply_parent, verify_frozen, ENCODER, eligible_names, optimizer_groups, install_live_encoder
    from full_unfreeze_v1.model import ContinuousChannel, Frame, surrogate_context
    from full_unfreeze_v1.report import verify, report

    p=add_shared_args(argparse.ArgumentParser(description=__doc__,allow_abbrev=False))
    p.add_argument('--a-delta',required=True); p.add_argument('--source-run',required=True)
    p.add_argument('--calibration',required=True); p.add_argument('--experiment-config',default=str(TEMPLATE))
    p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    cfg=load(a.experiment_config); bank=read_bank(a.calibration)
    check_bank(cfg,bank)
    if a.smoke: cfg.update(train_samples=64,epochs=1,val_samples=2,primary_validation_offset=0,draws=1,save_images=2,warmup_optimizer_steps=2)
    if not a.smoke and bank['smoke']: raise ValueError('Tiny smoke bank cannot authorize full training')
    if not math.isfinite(a.max_hours) or not 0<a.max_hours<=24: raise ValueError('Invalid wall-time cap')
    a.batch_size,a.num_workers,a.no_amp,a.fixed_mode,a.csi,a.blocks,a.seed=1,0,True,0,'pilot',1,cfg['seed']
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1': raise ValueError('This release uses physical GPU1 only; use launch.py')
    a.min_free_gb=cfg['min_free_gib']; a.min_free_disk_gb=12.
    device=preflight(a); config=load_config(a)
    if device.type!='cuda': raise ValueError('Real screening requires CUDA; CPU toy tests are separate')
    if config['model']['image_size']!=128: raise ValueError('Keep 128x128 resource contract')
    base=load_checkpoint(a.checkpoint); delta=load_checkpoint(a.a_delta)
    hashes=dict(base_checkpoint_sha256=digest(a.checkpoint),a_delta_sha256=digest(a.a_delta),split_sha256=digest(a.split_manifest))
    if not str(base.get('version','')).startswith('V10-') or base.get('split_sha256')!=hashes['split_sha256']:
        raise ValueError('Exact base/split required')
    if (delta.get('version')!='V10-token-guidance-quick-v1' or delta.get('variant')!='A' or
        delta.get('base_checkpoint_sha256')!=hashes['base_checkpoint_sha256'] or delta.get('split_sha256')!=hashes['split_sha256']):
        raise ValueError('Exact original A required')
    config['v10']=copy.deepcopy(base['config']['v10']); config['v10'].update(blocks=1,csi='pilot',fixed_mode=0)
    config['model']['vae_use_fp16']=False; del base
    source=Path(a.source_run).resolve(strict=True)
    # Strict identity/hash/full-matrix check of the user's completed V2 result.
    from fso_fluctuating_v2.report import verify as verify_parent
    verify_parent(source,complete=True)
    parent_design=json.loads((source/'design.json').read_text())
    parent_identity=json.loads((source/'run_identity.json').read_text())
    parent_record=json.loads((source/'train_B2_fso_ft/COMPLETE.json').read_text())
    start_file=source/'train_B2_fso_ft/last_delta.pth'
    if digest(start_file)!=parent_record['checkpoint_sha256']:
        raise ValueError('Changed parent B2 weight')
    start_delta=load_checkpoint(start_file)
    check_parent(start_delta,parent_record,parent_design,parent_identity,hashes,digest(Path(a.calibration)/'bank.json'))
    directory,identity=begin_run(a.output_dir,a,config,EXT)
    identity.update(version=VERSION,**hashes,init_fso_delta_sha256=digest(start_file),calibration_sha256=digest(Path(a.calibration)/'bank.json'),
        experiment_config_sha256=digest(a.experiment_config),new_source_sha256={f.name:digest(f) for f in sorted(Path(__file__).parent.iterdir()) if f.suffix in ('.py','.json','.md')},
        calibration_source_sha256=bank['source_sha256'])
    atomic_json(directory/'run_identity.json',identity)
    with (directory/'calibration_summary.json').open('xb') as stream:
        stream.write((Path(a.calibration)/'bank.json').read_bytes())
    started=time.monotonic()
    def budget():
        if time.monotonic()-started>a.max_hours*3600: raise TimeoutError('Overall cap reached; no COMPLETE, preserve partial artifacts')
    def sync():
        if device.type=='cuda': torch.cuda.synchronize(device)
    try:
        model, exact=build_model(config,device,a.checkpoint)
        if exact!={'v10_exact':True}: raise ValueError('Nonexact base load')
        model.float(); guide=Guidance(model)
        if set(delta['delta'])!={n for n,_ in guide.trainable('A')}: raise ValueError('A tensor key mismatch')
        model.load_state_dict(dict(model.state_dict(),**delta['delta']),strict=True); del delta
        guide.set('A'); model.base.channel=ContinuousChannel(model.base.channel,cfg).to(device)
        semantic=semantic_parameters(model)
        apply_parent(model,semantic,start_delta)
        named,tail_prefix=candidates(model)
        install_live_encoder(model.base.vae_wrapper)
        semantic_count=sum(q.numel() for _,q in semantic)
        tail_count=sum(q.numel() for n,q in named if n.startswith(tail_prefix))
        print('PARAMETERS semantic={} tail={} U1_eligible={} U2_all_registered_eligible={}; inactive branches are NOT activated'.format(semantic_count,tail_count,semantic_count+tail_count,sum(q.numel() for _,q in named)),flush=True)
        del start_delta
        initial={n:q.detach().cpu().clone() for n,q in named}
        initial_all={n:q.detach().cpu().clone() for n,q in model.named_parameters()}
        atomic_json(directory/'checkpoint_load.json',dict(exact_base=True,exact_A=True,exact_parent_B2=True,
            candidate_parameters={n:q.numel() for n,q in named},vae_tail_prefixes=tail_prefix,
            semantic_parameters=semantic_count,vae_tail_parameters=tail_count,
            trainable_per_arm={ARMS[0]:0,ARMS[1]:semantic_count+tail_count,ARMS[2]:sum(q.numel() for _,q in named)},
            semantic_names=[n for n,_ in semantic],all_registered_parameters=sum(q.numel() for _,q in named),
            eligibility_not_observed_gradient_coverage=True,CLIP_and_VAE_encoder_frozen_in_U1=True,**hashes,init_fso_delta_sha256=digest(start_file)))
        def cache(split,limit):
            loader=build_loader(config,a.split_manifest,split,limit,cfg['seed']); loader.dataset.caption_mode='first'
            result=[]
            with torch.no_grad():
                for batch in loader:
                    budget(); image=batch['images']; text=batch['texts'][0]
                    image_id=int(batch['image_ids'][0])
                    generator=torch.Generator().manual_seed(seed_for(cfg['seed'],split,image_id,'posterior')%(2**63-1))
                    epsilon=torch.randn((1,4,image.shape[-2]//8,image.shape[-1]//8),generator=generator)
                    model.base.vae_wrapper.posterior_noise=epsilon
                    latent=model.base.vae_wrapper.encode_images(image.to(device)).detach().cpu()
                    text_reference=model.base.text_encoder([text],device)['pooled'].detach().cpu()
                    result.append(dict(image_id=image_id,text=text,image=image,latent=latent,epsilon=epsilon,text_reference=text_reference,
                        source_sha256=hashlib.sha256(image.numpy().tobytes()+text.encode()).hexdigest()))
            if len(result)!=limit or len({x['image_id'] for x in result})!=limit: raise ValueError('Incomplete/duplicate image cache')
            return result
        model.eval(); train=cache('train',cfg['train_samples']); val=cache('validation',cfg['val_samples'])
        if {x['image_id'] for x in train}&{x['image_id'] for x in val}: raise ValueError('Train/validation overlap')
        total=cfg['epochs']*math.ceil(cfg['train_samples']/cfg['grad_accum'])
        if cfg['warmup_optimizer_steps']>=total: raise ValueError('Joint encoder stage must actually run')
        if not a.smoke and [x['image_id'] for x in val[:32]]!=parent_design['validation_ids']:
            raise ValueError('Expected original V2 validation prefix; cohort identity differs')
        design=dict(version=VERSION,config=cfg,smoke=a.smoke,arms=ARMS,split='validation',
            train_ids=[x['image_id'] for x in train],validation_ids=[x['image_id'] for x in val],
            train_reference_sha256={str(x['image_id']):hashlib.sha256(x['latent'].numpy().tobytes()).hexdigest() for x in train},
            train_posterior_noise_sha256={str(x['image_id']):hashlib.sha256(x['epsilon'].numpy().tobytes()).hexdigest() for x in train},
            validation_source_sha256={str(x['image_id']):x['source_sha256'] for x in val},
            optimizer_steps=total,train_forward_steps=cfg['epochs']*cfg['train_samples'],
            primary_validation_ids=[x['image_id'] for x in val[cfg['primary_validation_offset']:]],
            vae_tail_prefixes=tail_prefix,
            initialization='same completed V2 B2 FSO model for ALL arms',checkpoint_selection='last fixed step, no test',
            environment=environment(cfg),calibration_groups=bank['groups'],calibration_config=bank['config'],
            primary='U2-U1 ALL-FRAME PSNR on validation positions32-63, configured low-quality conditions; BOTH-AVAILABLE and paired noisy-minus-clean margin separately; exploratory screening, no final test',
            training_failure_task_loss='zero task gradient on public-surrogate unavailable frames; PAPR still trained, failed trials retained',
            paired_control='SAME live encoding, frozen reference latent and posterior epsilon, FSO surrogate, noise/order/anchor/task mask, semantic/tail LR/loss/updates; U1 tail vs U2 all registered parameters eligible',
            latent_reference='original frozen B2 posterior with fixed per-image epsilon; cached only for LOSS, NEVER supplied as model input',
            common_metric_reference='same original frozen CLIP projection for every arm, evaluation only; not a moving trained teacher',
            full_scope='all registered parameters eligible, no formerly inactive branch activated; observed gradient coverage separately audited',
            memory_guard='FP32 batch1 sequential arms; minimum free16GiB at start and4GiB sampled reserve; OOM aborts without auto retry',
            clean_anchor='same 15-percent schedule replaces, not adds, steps for both trained arms',
            fixed_budget='C32/I224/T32 width16, 2304 complex symbols, R=1, AWG64GSa/s,4SPS,RRC.1',
            seed_replicates=1,original_text_receiver_side_channel=False,
            limitations='simulator-calibrated stochastic surrogate, no hardware data; same peak/slot caps NOT equal actual energy; conditioned contrast not causal delivered-rate superiority')
        atomic_json(directory/'design.json',design)
        (directory/'images').mkdir(); (directory/'constellations').mkdir()
        def forward(item,context):
            budget()
            with patch.object(module,'LinkFrame',lambda *args,**kw:Frame(*args,context=context,config=cfg,**kw)):
                model.base.vae_wrapper.posterior_noise=item['epsilon']
                outputs=model(item['image'].to(device),[item['text']],'awgn',0.)
                outputs['common_text_semantic_target']=F.normalize(item['text_reference'].to(device).float(),dim=-1)
                return outputs
        fixed_reference={}
        rows_done=0
        with (directory/'samples.jsonl').open('x',encoding='utf-8') as samples, (directory/'events.jsonl').open('x') as events, (directory/'diagnostics.jsonl').open('x') as diagnostics:
            for arm in ARMS:
                with torch.no_grad():
                    for n,q in named: q.copy_(initial[n].to(device))
                target=directory/('train_'+arm); target.mkdir()
                steps=updates=0; counts=dict(source_encoder=0,tx_mapping=0,rx_mapping=0,decoder=0,vae_tail=0,vae_encoder=0,vae_decoder=0,clip=0,text_projection=0)
                covered=set(); live_encoder_gradient_steps=0; memory_samples=[]
                stage(model,False,arm==ARMS[2],tail_prefix,frozen=arm==ARMS[0])
                rejected=anchors=0; arm_start=time.monotonic()
                if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
                if arm!=ARMS[0]:
                    opt=torch.optim.AdamW(optimizer_groups(named,cfg,arm==ARMS[2],tail_prefix))
                    opt.zero_grad(set_to_none=True)
                    with (target/'history.jsonl').open('x') as history:
                        for epoch in range(cfg['epochs']):
                            order=list(range(len(train))); random.Random(seed_for(cfg['seed'],epoch,'order')).shuffle(order)
                            for pos,ix in enumerate(order):
                                budget()
                                if time.monotonic()-arm_start>cfg['arm_max_hours']*3600: raise TimeoutError('Per-arm cap: '+arm)
                                stage(model,updates<cfg['warmup_optimizer_steps'],arm==ARMS[2],tail_prefix)
                                network_seed=seed_for(cfg['seed'],epoch,pos,'network')%(2**31-1)
                                seed_everything(network_seed)
                                index=schedule(bank['config'],epoch,pos,bank)
                                noise_seed=seed_for(cfg['seed'],epoch,pos,'train_noise')
                                if index is None:
                                    ctx=dict(profile='clean',snr=40.,field_seed=0,noise_seed=noise_seed); anchors+=1
                                else: ctx=surrogate_context(bank,index,noise_seed)
                                item=train[ix]; out=forward(item,ctx)
                                if arm==ARMS[2] and not out['vae_latents'].requires_grad: raise RuntimeError('Live VAE encoder detached')
                                valid=bool(model.last_frame.header_ok[0])
                                task_valid=valid and (index is None or bank['records'][index]['available'])
                                rejected+=int(not task_valid)
                                pix=F.mse_loss(out['image'].float(),item['image'].to(device).float())
                                lat=F.mse_loss(out['recovered_vae_latents'].float(),item['latent'].to(device).float())
                                structure=1-ssim_per_image(out['image'],item['image'].to(device)).mean()
                                logits=out['text_logits'].float(); ids=out['text_input_ids']; length=min(logits.shape[1],ids.shape[1])
                                mask=out['text_attention_mask'][:,:length].float()
                                ce=F.cross_entropy(logits[:,:length].reshape(-1,logits.shape[-1]),ids[:,:length].reshape(-1),reduction='none').reshape_as(mask)
                                text=(ce*mask).sum()/mask.sum().clamp_min(1)
                                task=1.5*lat+.75*pix+.35*structure+.5*text
                                loss=task*float(task_valid)+cfg['papr_weight']*model.base.channel.papr_penalty
                                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss; no skipped update')
                                divisor=min(cfg['grad_accum'],len(order)-(pos//cfg['grad_accum'])*cfg['grad_accum'])
                                (loss/divisor).backward(); steps+=1
                                live_encoder_gradient_steps+=int(arm==ARMS[2] and any(q.grad is not None and bool(q.grad.detach().abs().max()>0) for n,q in named if n.startswith('base.vae_wrapper.vae.encoder.')))
                                if (pos+1)%cfg['grad_accum']==0 or pos+1==len(order):
                                    norm=torch.nn.utils.clip_grad_norm_([q for _,q in named],1.)
                                    if not torch.isfinite(norm): raise FloatingPointError('Nonfinite gradient; preserve failed run')
                                    grads=gradients(named,tail_prefix)
                                    for k in counts: counts[k]+=int(grads[k]>0)
                                    covered.update(n for n,q in named if q.grad is not None and bool(q.grad.detach().abs().max()>0))
                                    opt.step(); opt.zero_grad(set_to_none=True); updates+=1
                                rec=dict(model.last_frame.last_stats,arm=arm,epoch=epoch+1,forward_step=steps,optimizer_step=updates,
                                    image_id=item['image_id'],bank_index=index,anchor=index is None,receiver_available=valid,
                                    network_seed=network_seed,noise_seed=noise_seed,vae_tail_trainable=True,full_trainable=arm==ARMS[2],
                                    live_vae_encoder=True,reference_latent_sha256=hashlib.sha256(item['latent'].numpy().tobytes()).hexdigest(),
                                    posterior_noise_sha256=hashlib.sha256(item['epsilon'].numpy().tobytes()).hexdigest(),
                                    task_loss_applied=task_valid,loss=float(loss.detach()),unmasked_task_loss=float(task.detach()),
                                    pixel_mse=float(pix.detach()),text_ce=float(text.detach()),elapsed_seconds=time.monotonic()-arm_start)
                                if steps==1 or steps%20==0 or steps==design['train_forward_steps']:
                                    sync(); free,total_memory=torch.cuda.mem_get_info(device)
                                    rec.update(free_gib=free/2**30,peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                                        peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
                                    memory_samples.append({k:rec[k] for k in ('forward_step','optimizer_step','free_gib','peak_allocated_gib','peak_reserved_gib')})
                                    atomic_json(target/'memory_samples.json',memory_samples)
                                    if free/2**30<cfg['reserve_free_gib']: raise MemoryError('GPU free reserve below configured4GiB; no automatic retry or other-process action')
                                    rec.update(stage='training',projected_arm_training_seconds=rec['elapsed_seconds']*design['train_forward_steps']/steps)
                                    atomic_json(directory/'progress.json',rec)
                                    print('TRAIN {} epoch={}/{} step={}/{} loss={:.4f} masked_tasks={} elapsed={:.1f}s projected={:.1f}min'.format(
                                        arm,epoch+1,cfg['epochs'],steps,design['train_forward_steps'],rec['loss'],rejected,rec['elapsed_seconds'],rec['projected_arm_training_seconds']/60),flush=True)
                                    print('MEMORY peak_allocated={:.2f}GiB peak_reserved={:.2f}GiB free={:.2f}GiB'.format(
                                        rec['peak_allocated_gib'],rec['peak_reserved_gib'],rec['free_gib']),flush=True)
                                history.write(json.dumps(rec,allow_nan=False)+'\n'); history.flush()
                                del out,loss,task,logits,ce,pix,lat,structure,text
                    if updates!=total or steps!=design['train_forward_steps'] or min(counts[k] for k in ('source_encoder','tx_mapping','rx_mapping','decoder'))<=0: raise RuntimeError('Missing optimizer steps/encoder-TX-RX-decoder gradient')
                    changes={n:float((q.detach().cpu()-initial[n]).square().sum()) for n,q in named}
                    for prefix in ('base.multimodal_encoder.','base.channel.tx_map.','base.channel.rx_map.','base.joint_decoder.'):
                        if sum(v for n,v in changes.items() if n.startswith(prefix))<=0: raise RuntimeError('No parameter change '+prefix)
                    tail_change=sum(v for n,v in changes.items() if n.startswith(tail_prefix))
                    if counts['vae_tail']<=0 or tail_change<=0:
                        raise RuntimeError('VAE tail must have real gradient AND parameter change')
                    if arm==ARMS[2]:
                        for key,prefix in [('clip','base.text_encoder.backbone.'),('vae_encoder','base.vae_wrapper.vae.encoder.'),('vae_decoder','base.vae_wrapper.vae.decoder.'),('text_projection','base.text_encoder.projection.')]:
                            if counts[key]<=0 or sum(v for n,v in changes.items() if n.startswith(prefix))<=0:
                                raise RuntimeError('No actual full-backbone gradient/update '+key)
                        if live_encoder_gradient_steps<=0: raise RuntimeError('No live encoder gradient')
                    atomic_checkpoint(target/'last_delta.pth',dict(version=VERSION,arm=arm,config=cfg,**hashes,
                        init_fso_delta_sha256=digest(start_file),calibration_sha256=identity['calibration_sha256'],
                        steps=steps,optimizer_steps=updates,delta={n:q.detach().cpu().clone() for n,q in named}))
                    saved=load_checkpoint(target/'last_delta.pth')
                    if saved['version']!=VERSION or saved['arm']!=arm or set(saved['delta'])!={n for n,_ in named}:
                        raise RuntimeError('New delta roundtrip identity mismatch')
                    if any(not torch.equal(q.detach().cpu(),saved['delta'][n]) for n,q in named):
                        raise RuntimeError('New delta roundtrip tensor mismatch')
                    apply_parent(model,named,saved); del saved
                    del opt
                else: changes={n:0. for n,_ in named}
                allowed=eligible_names(model,arm==ARMS[2],tail_prefix) if arm!=ARMS[0] else set()
                verify_frozen(model,initial_all,allowed)
                peak_allocated=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None
                peak_reserved=torch.cuda.max_memory_reserved(device)/2**30 if device.type=='cuda' else None
                atomic_json(target/'COMPLETE.json',dict(arm=arm,steps=steps,optimizer_steps=updates,gradient_steps=counts,
                    parameter_changes=changes,masked_training_task_frames=rejected,clean_anchor_frames=anchors,
                    frozen_parameters_unchanged=True,vae_tail_gradient_steps=counts['vae_tail'],
                    vae_tail_parameter_change=sum(v for n,v in changes.items() if n.startswith(tail_prefix)),
                    observed_nonzero_gradient_names=sorted(covered),observed_nonzero_gradient_parameters=sum(q.numel() for n,q in named if n in covered),
                    eligible_parameters=sum(q.numel() for n,q in named if n in allowed),
                    live_encoder_gradient_forward_steps=live_encoder_gradient_steps,real_live_encoder_forward=True,
                    full_backbone_gradient_and_update_verified=arm==ARMS[2],
                    minimum_sampled_free_gib=min(r['free_gib'] for r in memory_samples) if memory_samples else None,
                    training_peak_allocated_gib=peak_allocated,training_peak_reserved_gib=peak_reserved,
                    gpu_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
                    checkpoint_roundtrip_verified=arm!=ARMS[0],
                    checkpoint_sha256=digest(target/'last_delta.pth') if arm!=ARMS[0] else None,wall_seconds=time.monotonic()-arm_start))
                stage(model,False,arm==ARMS[2],tail_prefix,frozen=True)
                model.base.channel.papr_penalty=None; model.base.channel.waveform_papr_db=None
                torch.cuda.empty_cache()
                with torch.no_grad():
                    for index,item in enumerate(val):
                        budget()
                        model.base.vae_wrapper.posterior_noise=item['epsilon']
                        direct=model.base.vae_wrapper.decode_latents(item['latent'].to(device))
                        direct_metrics=batch_image_metrics(direct,item['image'].to(device))
                        own_direct=model.base.vae_wrapper.decode_latents(model.base.vae_wrapper.encode_images(item['image'].to(device)))
                        own_metrics=batch_image_metrics(own_direct,item['image'].to(device))
                        if arm==ARMS[0]:
                            ref=forward(item,dict(profile='clean',snr=40.,field_seed=0,noise_seed=0))
                            fixed_reference[item['image_id']]=ref['recovered_vae_latents'].detach().cpu().clone()
                            del ref
                        fixed=model.base.vae_wrapper.decode_latents(fixed_reference[item['image_id']].to(device))
                        fixed_metrics=batch_image_metrics(fixed,item['image'].to(device))
                        diagnostic=dict(arm=arm,image_id=item['image_id'],source_sha256=item['source_sha256'],
                            cached_vae_latent_sha256=hashlib.sha256(item['latent'].numpy().tobytes()).hexdigest(),
                            fixed_reference_latent_sha256=hashlib.sha256(fixed_reference[item['image_id']].numpy().tobytes()).hexdigest(),
                            direct_vae_psnr=float(direct_metrics['psnr'][0]),direct_vae_ssim=float(direct_metrics['ssim'][0]),
                            fixed_U0_semantic_latent_psnr=float(fixed_metrics['psnr'][0]),fixed_U0_semantic_latent_ssim=float(fixed_metrics['ssim'][0]),
                            direct_reference_not_mathematical_upper_bound=True,
                            own_live_vae_psnr=float(own_metrics['psnr'][0]),own_live_vae_ssim=float(own_metrics['ssim'][0]))
                        if index in visual_positions(cfg):
                            for label,picture in [('direct_vae',direct),('fixed_latent',fixed),('own_live_vae',own_direct)]:
                                filename='images/{}__{}__{}.png'.format(item['image_id'],arm,label)
                                save_image(picture[0],directory/filename)
                                diagnostic[label+'_file']=filename; diagnostic[label+'_file_sha256']=digest(directory/filename)
                        diagnostics.write(json.dumps(diagnostic,allow_nan=False)+'\n'); diagnostics.flush()
                        del direct,fixed,own_direct,direct_metrics,fixed_metrics,own_metrics
                    for condition in [dict(id='clean',weather='weak',launch_signal_dbm=0.)]+cfg['conditions']:
                        for draw in range(1 if condition['id']=='clean' else cfg['draws']):
                            trace=paths(cfg,condition['weather'],len(val),('evaluation',condition['weather'],draw))
                            for index,item in enumerate(val):
                                ns=seed_for(cfg['seed'],condition['id'],draw,index,'eval_noise')
                                ctx=dict(profile='clean',snr=40.,field_seed=0,noise_seed=ns) if condition['id']=='clean' else dict(
                                    transport=transport(cfg,condition,trace[index],ns),noise_seed=ns)
                                out=forward(item,ctx); metric=batch_image_metrics(out['image'],item['image'].to(device))
                                caption=model.base.decode_text_logits(out['text_logits'])[0]
                                correct,count=token_accuracy(out['text_logits'],out['text_input_ids'],out['text_attention_mask'])
                                stats=model.last_frame.last_stats; t=model.last_frame.transport
                                row=dict(stats,arm=arm,image_id=item['image_id'],source_sha256=item['source_sha256'],
                                    condition=condition['id'],draw=draw,frame_index=index,noise_seed=ns,
                                    weather_path=None if t is None else trace[index],
                                    business_frame_spacing_s=cfg['frame_spacing_s'],
                                    receiver_available=bool(model.last_frame.header_ok[0]),psnr=float(metric['psnr'][0]),ssim=float(metric['ssim'][0]),
                                    received_caption=caption,source_caption_evaluation_only=item['text'],source_utf8_bytes=len(item['text'].encode()),
                                    token_acc=correct/max(count,1),bleu2=sentence_bleu2(item['text'],caption),
                                    neural_input_capacity_tokens=int(out['text_input_ids'].shape[1]),
                                    common_to_text_cosine=float(F.cosine_similarity(out['common_receiver_semantic'].float(),out['common_text_semantic_target'].float(),dim=-1)[0]),
                                    **caption_metrics(item['text'],caption))
                                if index in visual_positions(cfg):
                                    row['source_image_file']='images/{}__source.png'.format(item['image_id'])
                                    if not (directory/row['source_image_file']).exists(): save_image(item['image'][0],directory/row['source_image_file'])
                                    row['image_file']='images/{}__{}__{}__d{}.png'.format(item['image_id'],arm,condition['id'],draw)
                                    save_image(out['image'][0],directory/row['image_file'])
                                    for k in ('image_file','source_image_file'): row[k+'_sha256']=digest(directory/row[k])
                                if t is not None:
                                    row['event_count']=len(t.events)
                                    for e in t.events: events.write(json.dumps(dict(e,arm=arm,image_id=item['image_id'],condition=condition['id'],draw=draw),allow_nan=False)+'\n')
                                    events.flush()
                                    if index in visual_positions(cfg) and t.constellations:
                                        row['constellation_file']='constellations/{}__{}__{}__d{}.svg'.format(item['image_id'],arm,condition['id'],draw)
                                        constellation_svg(t.constellations,directory/row['constellation_file']); row['constellation_file_sha256']=digest(directory/row['constellation_file'])
                                samples.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); samples.flush(); rows_done+=1
                                if rows_done%20==0:
                                    atomic_json(directory/'progress.json',dict(stage='evaluation',arm=arm,rows=rows_done,elapsed_seconds=time.monotonic()-started))
                                    print('EVAL {} rows={} {} PSNR={:.2f} WER={:.3f} available={}'.format(arm,rows_done,condition['id'],row['psnr'],row['normalized_wer'],row['receiver_available']),flush=True)
                                del out
        if digest(a.checkpoint)!=hashes['base_checkpoint_sha256'] or digest(a.a_delta)!=hashes['a_delta_sha256'] or digest(a.split_manifest)!=hashes['split_sha256'] or digest(start_file)!=identity['init_fso_delta_sha256']:
            raise ValueError('Initialization weight/split changed during run; no complete')
        sync(); verify(directory); report(directory)
        identity['status']='complete'; atomic_json(directory/'run_identity.json',identity)
        result_files=[f for f in directory.rglob('*') if f.is_file() and f.suffix!='.pth']
        atomic_json(directory/'COMPLETE.json',dict(version=VERSION,rows=rows_done,wall_seconds=time.monotonic()-started,
            result_sha256={str(f.relative_to(directory)):digest(f) for f in result_files},
            checkpoint_sha256={str(f.relative_to(directory)):digest(f) for f in directory.glob('train_*/last_delta.pth')}))
        verify(directory,complete=True); print('COMPLETE '+str(directory/'report.html'),flush=True)
    except BaseException as exc:
        identity.update(status='failed',error=repr(exc)); atomic_json(directory/'run_identity.json',identity)
        atomic_json(directory/'FAILED.json',dict(error=repr(exc),partial_artifacts_preserved=True)); raise


if __name__=='__main__': main()
