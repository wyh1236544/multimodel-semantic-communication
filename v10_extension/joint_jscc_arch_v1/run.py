"""Frozen backbones, matched clean->FSO training, actual waveform validation. GPU1."""
import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from joint_jscc_arch_v1 import VERSION,ARMS,TRAINED_ARMS
from joint_jscc_arch_v1.core import load,TEMPLATE,digest,seed_for


def main():
    import torch
    import torch.nn.functional as F
    from v10.runtime import add_shared_args,preflight,load_config
    from v10.io import begin_run,atomic_json,atomic_checkpoint
    from v10.model import build_model
    import v10.model as model_module
    from v10.data import build_loader
    from utils.checkpoint import load_checkpoint
    from utils.metrics import batch_image_metrics,ssim_per_image
    from token_guidance.model import Guidance
    from fso_jscc_v1.model import parameters as semantic_parameters,ContinuousChannel
    from fso_jscc_v1.core import kernels
    from fso_fluctuating_v2.model import Frame,surrogate_context
    from fso_fluctuating_v2.core import paths,transport,schedule,environment
    from fso_fluctuating_v2.calibrate import read_bank
    from fso_fluctuating_v2.report import verify as verify_parent
    from full_unfreeze_v1.core import check_bank,check_parent
    from vae_tail_ablation_v1.model import apply_parent
    from joint_jscc_arch_v1.model import ArchitectureModel,encode_text
    from joint_jscc_arch_v1.fidelity import check as fidelity_check
    from joint_jscc_arch_v1.report import report,verify
    from final_compare_v1.metrics import caption_metrics
    from optical_realism_v3.constellation import constellation_svg
    from eval_v10 import save_image

    p=add_shared_args(argparse.ArgumentParser(description=__doc__,allow_abbrev=False))
    p.add_argument('--a-delta',required=True); p.add_argument('--source-run',required=True)
    p.add_argument('--calibration',required=True); p.add_argument('--experiment-config',default=str(TEMPLATE))
    p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    cfg,cc=load(a.experiment_config); bank=read_bank(a.calibration); check_bank(cc,bank)
    if a.smoke:
        cfg.update(train_samples=8,clean_epochs=1,noise_epochs=1,val_samples=2,draws=1,save_images=2)
    if not math.isfinite(a.max_hours) or not 0<a.max_hours<=24: raise ValueError('Invalid overall time cap')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1': raise ValueError('Physical GPU1 only; use launch.py')
    a.batch_size,a.num_workers,a.no_amp,a.fixed_mode,a.csi,a.blocks,a.seed=1,0,True,0,'pilot',1,cfg['seed']
    a.min_free_gb,a.min_free_disk_gb=cfg['min_free_gib'],12.
    device=preflight(a); config=load_config(a)
    if device.type!='cuda': raise ValueError('Real pretrained experiment requires CUDA; toy CPU tests are separate')
    if (config['model']['image_size'],config['model']['hidden_dim'],config['model']['latent_dim'])!=(128,256,16):
        raise ValueError('Original128/256/16 backbone required')
    config['model']['vae_use_fp16']=False
    parent=Path(a.source_run).resolve(strict=True); verify_parent(parent,complete=True)
    parent_design=json.loads((parent/'design.json').read_text()); parent_identity=json.loads((parent/'run_identity.json').read_text())
    parent_record=json.loads((parent/'train_B2_fso_ft/COMPLETE.json').read_text())
    parent_file=parent/'train_B2_fso_ft/last_delta.pth'
    if digest(parent_file)!=parent_record['checkpoint_sha256']: raise ValueError('Changed B2 initialization')
    base=load_checkpoint(a.checkpoint); delta=load_checkpoint(a.a_delta); parent_delta=load_checkpoint(parent_file)
    hashes=dict(base_checkpoint_sha256=digest(a.checkpoint),a_delta_sha256=digest(a.a_delta),split_sha256=digest(a.split_manifest))
    if base.get('split_sha256')!=hashes['split_sha256'] or not str(base.get('version','')).startswith('V10-'):
        raise ValueError('Exact base/split required')
    if delta.get('version')!='V10-token-guidance-quick-v1' or delta.get('variant')!='A' or any(delta.get(k)!=v for k,v in hashes.items() if k!='a_delta_sha256'):
        raise ValueError('Exact original A delta required')
    check_parent(parent_delta,parent_record,parent_design,parent_identity,hashes,digest(Path(a.calibration)/'bank.json'))
    config['v10']=copy.deepcopy(base['config']['v10']); config['v10'].update(blocks=1,csi='pilot',fixed_mode=0); del base
    directory,identity=begin_run(a.output_dir,a,config,EXT)
    identity.update(version=VERSION,**hashes,init_fso_delta_sha256=digest(parent_file),
        architecture_config_sha256=digest(a.experiment_config),calibration_sha256=digest(Path(a.calibration)/'bank.json'),
        new_source_sha256={f.name:digest(f) for f in sorted(Path(__file__).parent.iterdir()) if f.suffix in ('.py','.json','.md')},
        calibration_source_sha256=bank['source_sha256'])
    atomic_json(directory/'run_identity.json',identity)
    with (directory/'calibration_summary.json').open('xb') as stream: stream.write((Path(a.calibration)/'bank.json').read_bytes())
    started=time.monotonic()
    def budget():
        if time.monotonic()-started>a.max_hours*3600: raise TimeoutError('Overall cap reached; preserve partial output')
    def sync(): torch.cuda.synchronize(device)
    try:
        legacy,exact=build_model(config,device,a.checkpoint)
        if exact!={'v10_exact':True}: raise ValueError('Nonexact base loading')
        legacy.float(); guide=Guidance(legacy)
        if set(delta['delta'])!={n for n,_ in guide.trainable('A')}: raise ValueError('A delta tensor set differs')
        legacy.load_state_dict(dict(legacy.state_dict(),**delta['delta']),strict=True); del delta
        guide.set('A'); legacy.base.channel=ContinuousChannel(legacy.base.channel,cc).to(device)
        apply_parent(legacy,semantic_parameters(legacy),parent_delta); del parent_delta
        legacy.requires_grad_(False); legacy.eval()
        legacy.base.text_encoder.freeze=True; legacy.base.vae_wrapper.freeze=True
        frozen={n:p.detach().cpu().clone() for n,p in legacy.named_parameters()}
        def cache(split,count):
            loader=build_loader(config,a.split_manifest,split,count,cfg['seed']); loader.dataset.caption_mode='first'
            result=[]
            with torch.no_grad():
                for batch in loader:
                    budget(); image=batch['images']; text=batch['texts'][0]; iid=int(batch['image_ids'][0])
                    posterior=legacy.base.vae_wrapper.vae.encode(image.to(device)*2.-1.).latent_dist
                    latent=(posterior.mean*legacy.base.vae_wrapper.scaling_factor).cpu()
                    info=encode_text(legacy.base,text,cfg['max_text_tokens'],device)
                    cids=info['target_ids'][:min(info['target_length'],cfg['common_text_tokens'])]
                    if cids[-1]!=legacy.base.text_encoder.tokenizer.eos_token_id: cids[-1]=legacy.base.text_encoder.tokenizer.eos_token_id
                    tok=legacy.base.text_encoder.tokenizer
                    result.append(dict(image_id=iid,text=text,image=image,latent=latent,text_features=info['features'],
                        text_mask=info['mask'],target_ids=torch.tensor([info['target_ids']]),target_length=torch.tensor([info['target_length']]),
                        common_ids=torch.tensor([cids]),common_length=torch.tensor([len(cids)]),
                        common_reference=tok.decode(cids,skip_special_tokens=True),covered_text=info['covered_text'],
                        source_content_tokens=info['source_content_tokens'],covered_content_tokens=info['covered_content_tokens'],
                        truncated=info['truncated'],source_sha256=hashlib.sha256(image.numpy().tobytes()+text.encode()).hexdigest()))
            if len(result)!=count or len({r['image_id'] for r in result})!=count: raise ValueError('Incomplete/duplicate data')
            return result
        train=cache('train',cfg['train_samples']); val=cache('validation',cfg['val_samples'])
        if {r['image_id'] for r in train}&{r['image_id'] for r in val}: raise ValueError('Train/validation overlap')
        core=ArchitectureModel(legacy.base,cfg,kernels(cc)).to(device).float()
        initial={n:v.detach().cpu().clone() for n,v in core.state_dict().items()}
        named_initial={n:p.detach().cpu().clone() for n,p in core.named_parameters()}
        forward_steps=cfg['train_samples']*(cfg['clean_epochs']+cfg['noise_epochs'])
        optimizer_steps=math.ceil(cfg['train_samples']/cfg['grad_accum'])*(cfg['clean_epochs']+cfg['noise_epochs'])
        physical_conditions=cc['conditions'][:2] if a.smoke else cc['conditions']
        design=dict(version=VERSION,config=cfg,channel_config=cc,arms=list(ARMS),split='validation',smoke=a.smoke,
            train_ids=[r['image_id'] for r in train],validation_ids=[r['image_id'] for r in val],
            validation_source_sha256={str(r['image_id']):r['source_sha256'] for r in val},
            conditions=physical_conditions,train_forward_steps=forward_steps,optimizer_steps=optimizer_steps,
            primary_conditions=[c for c in cc['primary_conditions'] if c in {x['id'] for x in physical_conditions}],
            initialization='same completed B2 FSO weights; all new arms share exact initial tensors',
            latent_mode='deterministic posterior mean for ALL arms; B0 is a rerun control, not historical sampled-VAE score',
            source_text_capacity=cfg['max_text_tokens'],legacy_source_text_capacity=77,
            comparison='B2-B1 interface bundle; B3-B2 Common training; B3-B4 joint vs block-diagonal neural control',
            primary='B3-B4 all-frame PSNR and covered-text WER, paired per-image; exploratory validation screening',
            separate_control='same registered/trainable parameter count, same source/updates/physical budget; masks and modality memories differ; FLOPs are NOT claimed equal',
            common_task='only physically received C slots BEFORE RX private mixing; diagnostic uses full frame resource, NOT a separate Common-only transmission',
            teacher_forcing='TRAIN ONLY; evaluation uses BOS, received memory, own generated tokens and predicted EOS',
            failure_policy='masked task gradients on unavailable surrogate frames; PAPR still trained; all waveform failures retained',
            environment=environment(cc),analog_repeats=1,payload_complex_symbols=2304,raw_utf8_wire_bytes=0,
            original_text_receiver_side_channel=False,seed_replicates=1,
            checkpoint_selection='last fixed step; clean checkpoint diagnostics, no validation selection/automatic enlargement',
            limitations='frozen-backbone architecture screen, single training replicate, fixed weather paths, previously used validation, simulator proxy not hardware, same slot/peak caps not equal actual energy')
        atomic_json(directory/'design.json',design)
        atomic_json(directory/'checkpoint_load.json',dict(exact_base=True,exact_A=True,exact_parent_B2=True,
            frozen_backbones=True,initial_core_sha256={n:hashlib.sha256(v.numpy().tobytes()).hexdigest() for n,v in initial.items()},
            core_registered_parameters=sum(p.numel() for p in core.parameters()),
            parameter_counts={n:p.numel() for n,p in core.named_parameters()},**hashes))
        (directory/'images').mkdir(); (directory/'constellations').mkdir()
        def legacy_forward(item,ctx):
            with patch.object(model_module,'LinkFrame',lambda *args,**kw:Frame(*args,context=ctx,config=cc,**kw)):
                return legacy(item['image'].to(device),[item['text']],'awgn',0.,vae_latents_override=item['latent'].to(device))
        def make_frame(ctx):
            return Frame('awgn',0.,1,device,context=ctx,config=cc,fixed_mode=0,blocks=1)
        def new_forward(item,ctx):
            x=core.source_encode(item['latent'].to(device),item['text_features'].to(device),item['text_mask'].to(device))
            papr,penalty,wave_samples=core.waveform(x,cc); frame=make_frame(ctx)
            received,rel=frame.payload(x)
            out=core.receive(received,rel,frame.routing_snr,frame.header_ok)
            out.update(payload=x,papr=papr,penalty=penalty,wave_samples=wave_samples)
            return out,frame
        def check_fidelity(arm,stage):
            budget(); proof=fidelity_check(core,train,bank,cc,cfg,stage,device)
            atomic_json(directory/('train_'+arm)/('fidelity_'+stage+'.json'),proof)
            if not proof['passed']: raise RuntimeError('CURRENT-symbol surrogate fidelity failed '+arm+'/'+stage+'; no silent tolerance change')
            print('FIDELITY {} {} passed'.format(arm,stage),flush=True)
        for arm in TRAINED_ARMS:
            core.load_state_dict(initial,strict=True); core.configure(arm)
            if any(not torch.equal(v.detach().cpu(),initial[n]) for n,v in core.state_dict().items()):
                raise RuntimeError('New arms do not start from identical core tensors')
            target=directory/('train_'+arm); target.mkdir()
            allowed={n for n,p in core.named_parameters() if p.requires_grad}
            named=list(core.named_parameters()); covered=set(); auxiliary_frames=0
            change_groups=dict(source=('source_transformer.','image_projection.'),text_compression=('text_compressor.',),
                ar=('language.',),tx=('modern_tx.',) if arm!='B1_text' else ('old_tx.',),
                rx=('modern_rx.',) if arm!='B1_text' else ('old_rx.',),
                tx_spatial=('spatial_tx.',),rx_spatial=('spatial_rx.',),
                receiver=('rx_transformer.','image_head.'),conditioning=('rx_condition.',),common_coarse=('common_image_head.',))
            old_prefixes=('source_transformer.','source_norm.','bottleneck.','old_tx.','old_rx.','old_adapter.',
                          'rx_projection.','rx_transformer.','rx_norm.','image_head.')
            groups=[dict(params=[p for n,p in named if p.requires_grad and n.startswith(old_prefixes)],lr=cfg['old_lr']),
                    dict(params=[p for n,p in named if p.requires_grad and not n.startswith(old_prefixes)],lr=cfg['new_lr'])]
            opt=torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'])
            torch.cuda.reset_peak_memory_stats(device); arm_started=time.monotonic(); updates=steps=0
            check_fidelity(arm,'initial')
            with (target/'history.jsonl').open('x',encoding='utf-8') as history:
                for phase,epochs in [('clean',cfg['clean_epochs']),('fso',cfg['noise_epochs'])]:
                    for epoch in range(epochs):
                        order=np.random.default_rng(seed_for(cfg['seed'],phase,epoch,'order')).permutation(len(train))
                        for pos,ix in enumerate(order):
                            budget()
                            if time.monotonic()-arm_started>cfg['arm_max_hours']*3600: raise TimeoutError('Arm training cap reached')
                            item=train[int(ix)]; ns=seed_for(cfg['seed'],phase,epoch,pos,'noise')
                            index=None if phase=='clean' else schedule(bank['config'],epoch,pos,bank)
                            ctx=dict(profile='clean',snr=40.,field_seed=0,noise_seed=ns) if index is None else surrogate_context(bank,index,ns)
                            out,frame=new_forward(item,ctx); valid=bool(frame.header_ok[0])
                            image=legacy.base.vae_wrapper.decode_latents(out['latent'])
                            pix=F.mse_loss(image,item['image'].to(device)); lat=F.mse_loss(out['latent'],item['latent'].to(device))
                            structure=1-ssim_per_image(image,item['image'].to(device)).mean()
                            ce=core.language.teacher_loss(item['target_ids'].to(device),item['target_length'].to(device),out['text_memory'])
                            aux_lat=F.mse_loss(out['common_coarse'],F.adaptive_avg_pool2d(item['latent'].to(device),4))
                            # Compute equal diagnostic forwards in ALL arms; zero aux weights in B1/B2.
                            aux_text=core.language.teacher_loss(item['common_ids'].to(device),item['common_length'].to(device),out['common_text_memory'])
                            task=cfg['latent_weight']*lat+cfg['pixel_weight']*pix+cfg['ssim_weight']*structure+cfg['text_weight']*ce
                            if core.options['auxiliary']:
                                task=task+cfg['common_latent_weight']*aux_lat+cfg['common_text_weight']*aux_text
                                auxiliary_frames+=int(valid)
                            loss=task*float(valid)+cc['papr_weight']*out['penalty']
                            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss; no skipped update')
                            divisor=min(cfg['grad_accum'],len(order)-(pos//cfg['grad_accum'])*cfg['grad_accum'])
                            (loss/divisor).backward(); steps+=1
                            if (pos+1)%cfg['grad_accum']==0 or pos+1==len(order):
                                norm=torch.nn.utils.clip_grad_norm_([p for _,p in named if p.requires_grad],1.)
                                if not torch.isfinite(norm): raise FloatingPointError('Nonfinite gradient')
                                covered.update(n for n,p in named if p.grad is not None and bool(p.grad.detach().abs().max()>0))
                                opt.step(); opt.zero_grad(set_to_none=True); updates+=1
                            rec=dict(arm=arm,phase=phase,epoch=epoch+1,image_id=item['image_id'],forward_step=steps,
                                optimizer_step=updates,bank_index=index,anchor=index is None,task_loss_applied=valid,noise_seed=ns,
                                pixel_mse=float(pix.detach()),latent_mse=float(lat.detach()),text_ce=float(ce.detach()),
                                common_text_ce=float(aux_text.detach()),common_latent_mse=float(aux_lat.detach()),
                                auxiliary_applied=core.options['auxiliary'] and valid,loss=float(loss.detach()),
                                waveform_papr_db=float(out['papr'].detach()[0]),elapsed_seconds=time.monotonic()-arm_started)
                            if steps==1 or steps%20==0 or steps==forward_steps:
                                sync(); free,_=torch.cuda.mem_get_info(device)
                                rec.update(free_gib=free/2**30,peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                                    peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
                                    projected_train_seconds=rec['elapsed_seconds']*forward_steps/steps)
                                if rec['free_gib']<cfg['reserve_free_gib']: raise MemoryError('Free GPU reserve breached; no unrelated process action')
                                atomic_json(directory/'progress.json',dict(stage='training',**rec))
                                print('TRAIN {} {} epoch={}/{} step={}/{} CE={:.4f} pix={:.5f} elapsed={:.1f}s projected_train={:.1f}min peak={:.2f}GiB'.format(
                                    arm,phase,epoch+1,epochs,steps,forward_steps,rec['text_ce'],rec['pixel_mse'],rec['elapsed_seconds'],rec['projected_train_seconds']/60,rec['peak_allocated_gib']),flush=True)
                            history.write(json.dumps(rec,allow_nan=False)+'\n'); history.flush()
                            del out,frame,image,loss,task,pix,lat,structure,ce,aux_lat,aux_text
                    check_fidelity(arm,phase)
                    payload=dict(version=VERSION,arm=arm,phase=phase,steps=steps,optimizer_steps=updates,
                        architecture_config=cfg,channel_config=cc,init_fso_delta_sha256=identity['init_fso_delta_sha256'],
                        calibration_sha256=identity['calibration_sha256'],architecture_config_sha256=identity['architecture_config_sha256'],
                        delta={n:v.detach().cpu().clone() for n,v in core.state_dict().items()},**hashes)
                    atomic_checkpoint(target/(phase+'_delta.pth'),payload)
            changes={n:float((p.detach().cpu()-named_initial[n]).square().sum()) for n,p in named}
            required=['source','text_compression','ar','tx','rx','receiver']+(['conditioning','tx_spatial','rx_spatial'] if core.options['modern'] else [])+(['common_coarse'] if core.options['auxiliary'] else [])
            group_changes={key:sum(v for n,v in changes.items() if n.startswith(prefix)) for key,prefix in change_groups.items()}
            if steps!=forward_steps or updates!=optimizer_steps or any(group_changes[k]<=0 for k in required):
                raise RuntimeError('Missing real update/required module change')
            if any(changes[n]!=0 for n,_ in named if n not in allowed): raise RuntimeError('Ineligible parameter changed')
            # True strict reload, including tied embedding/output tensors, on server.
            checkpoint=target/'fso_delta.pth'; reloaded=load_checkpoint(checkpoint)
            core.load_state_dict(reloaded['delta'],strict=True)
            if any(not torch.equal(v.detach().cpu(),reloaded['delta'][n]) for n,v in core.state_dict().items()): raise RuntimeError('Checkpoint roundtrip mismatch')
            atomic_json(target/'COMPLETE.json',dict(arm=arm,steps=steps,optimizer_steps=updates,wall_seconds=time.monotonic()-arm_started,
                trainable_parameters=sum(p.numel() for n,p in named if n in allowed),registered_parameters=sum(p.numel() for _,p in named),
                observed_nonzero_gradient_parameters=sum(p.numel() for n,p in named if n in covered),
                observed_nonzero_gradient_names=sorted(covered),
                parameter_group_changes=group_changes,parameter_changes=changes,required_groups=required,
                auxiliary_frames=auxiliary_frames,checkpoint_sha256=digest(checkpoint),clean_checkpoint_sha256=digest(target/'clean_delta.pth'),
                checkpoint_roundtrip_verified=True,frozen_backbones=True,
                initial_core_verified=True,
                training_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                training_peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30))
            print('ARM COMPLETE '+arm,flush=True)
            del opt

        def text_fields(item,caption,capacity,covered,common_caption=None):
            full=caption_metrics(item['text'],caption); prefix=caption_metrics(covered,caption)
            result=dict(received_caption=caption,source_caption_evaluation_only=item['text'],source_utf8_bytes=len(item['text'].encode()),
                neural_input_capacity_tokens=capacity,covered_reference_evaluation_only=covered,
                covered_normalized_wer=prefix['normalized_wer'],covered_cer=prefix['cer'],
                input_truncated=item['truncated'] if capacity==cfg['max_text_tokens'] else item['source_content_tokens']>75,
                source_content_tokens=item['source_content_tokens'],covered_content_tokens=min(capacity-2,item['source_content_tokens']),**full)
            if common_caption is not None:
                result.update(common_caption=common_caption,common_reference_evaluation_only=item['common_reference'],
                    common_prefix_wer=caption_metrics(item['common_reference'],common_caption)['normalized_wer'])
            return result
        rows_done=0
        with (directory/'samples.jsonl').open('x',encoding='utf-8') as samples,(directory/'events.jsonl').open('x') as events,(directory/'clean_stage.jsonl').open('x') as clean_rows:
            for arm in ARMS:
                if arm!='B0_legacy':
                    core.load_state_dict(load_checkpoint(directory/('train_'+arm)/'clean_delta.pth')['delta'],strict=True); core.configure(arm)
                    with torch.no_grad():
                        for item in val:
                            out,frame=new_forward(item,dict(profile='clean',snr=40.,field_seed=0,noise_seed=0))
                            image=legacy.base.vae_wrapper.decode_latents(out['latent']); metric=batch_image_metrics(image,item['image'].to(device))
                            caption=legacy.base.text_encoder.tokenizer.batch_decode(core.language.generate(out['text_memory']),skip_special_tokens=True)[0]
                            clean_rows.write(json.dumps(dict(arm=arm,image_id=item['image_id'],psnr=float(metric['psnr'][0]),ssim=float(metric['ssim'][0]),
                                **text_fields(item,caption,cfg['max_text_tokens'],item['covered_text'])),ensure_ascii=False,allow_nan=False)+'\n'); clean_rows.flush()
                            budget()
                    core.load_state_dict(load_checkpoint(directory/('train_'+arm)/'fso_delta.pth')['delta'],strict=True); core.configure(arm)
                for condition in [dict(id='clean',weather='weak',launch_signal_dbm=0.)]+physical_conditions:
                    for draw in range(1 if condition['id']=='clean' else cfg['draws']):
                        trace=paths(cc,condition['weather'],len(val),('architecture-evaluation',condition['weather'],draw))
                        for index,item in enumerate(val):
                            budget(); ns=seed_for(cfg['seed'],condition['id'],draw,index,'architecture-eval-noise')
                            ctx=dict(profile='clean',snr=40.,field_seed=0,noise_seed=ns) if condition['id']=='clean' else dict(transport=transport(cc,condition,trace[index],ns),noise_seed=ns)
                            eval_started=time.monotonic()
                            with torch.no_grad():
                                if arm=='B0_legacy':
                                    out=legacy_forward(item,ctx); image=out['image']; frame=legacy.last_frame
                                    caption=legacy.base.decode_text_logits(out['text_logits'])[0]
                                    ref=legacy.base.text_encoder.tokenizer.batch_decode(out['text_input_ids'],skip_special_tokens=True)[0]
                                    tf=text_fields(item,caption,77,ref)
                                    papr=float(frame.last_stats['waveform_papr_db'])
                                else:
                                    out,frame=new_forward(item,ctx); image=legacy.base.vae_wrapper.decode_latents(out['latent'])
                                    tok=legacy.base.text_encoder.tokenizer
                                    caption=tok.batch_decode(core.language.generate(out['text_memory']),skip_special_tokens=True)[0]
                                    common_caption=tok.batch_decode(core.language.generate(out['common_text_memory'],cfg['common_text_tokens']),skip_special_tokens=True)[0]
                                    tf=text_fields(item,caption,cfg['max_text_tokens'],item['covered_text'],common_caption)
                                    papr=float(out['papr'][0])
                                metric=batch_image_metrics(image,item['image'].to(device))
                                row=dict(frame.last_stats,arm=arm,image_id=item['image_id'],condition=condition['id'],draw=draw,
                                    frame_index=index,noise_seed=ns,source_sha256=item['source_sha256'],weather_path=None if condition['id']=='clean' else trace[index],
                                    psnr=float(metric['psnr'][0]),ssim=float(metric['ssim'][0]),receiver_available=bool(frame.header_ok[0]),
                                    waveform_papr_db=papr,analog_repeats=1,raw_utf8_wire_bytes=0,pre_fec_ber=None,post_fec_ber=None,
                                    payload_complex_symbols=2304,offline_inference_seconds=time.monotonic()-eval_started,**tf)
                                if index<cfg['save_images']:
                                    row['source_image_file']='images/{}__source.png'.format(item['image_id'])
                                    if not (directory/row['source_image_file']).exists(): save_image(item['image'][0],directory/row['source_image_file'])
                                    row['image_file']='images/{}__{}__{}__d{}.png'.format(item['image_id'],arm,condition['id'],draw)
                                    save_image(image[0],directory/row['image_file'])
                                    for key in ('image_file','source_image_file'): row[key+'_sha256']=digest(directory/row[key])
                                t=frame.transport
                                if t is not None:
                                    row['event_count']=len(t.events)
                                    for event in t.events: events.write(json.dumps(dict(event,arm=arm,image_id=item['image_id'],condition=condition['id'],draw=draw),allow_nan=False)+'\n')
                                    events.flush()
                                    if index<cfg['save_images'] and t.constellations:
                                        row['constellation_file']='constellations/{}__{}__{}__d{}.svg'.format(item['image_id'],arm,condition['id'],draw)
                                        constellation_svg(t.constellations,directory/row['constellation_file']); row['constellation_file_sha256']=digest(directory/row['constellation_file'])
                                samples.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); samples.flush(); rows_done+=1
                                if rows_done%10==0:
                                    sync(); atomic_json(directory/'progress.json',dict(stage='evaluation',arm=arm,rows=rows_done,
                                        expected_rows=len(ARMS)*len(val)*(1+len(physical_conditions)*cfg['draws']),elapsed_seconds=time.monotonic()-started))
                                    print('EVAL {} rows={} {} PSNR={:.2f} coveredWER={:.3f} fullWER={:.3f} available={}'.format(
                                        arm,rows_done,condition['id'],row['psnr'],row['covered_normalized_wer'],row['normalized_wer'],row['receiver_available']),flush=True)
                                del out,image,frame
        if any(not torch.equal(p.detach().cpu(),frozen[n]) for n,p in legacy.named_parameters()): raise RuntimeError('Historical/frozen base changed')
        for filename,value in [(a.checkpoint,hashes['base_checkpoint_sha256']),(a.a_delta,hashes['a_delta_sha256']),
                               (a.split_manifest,hashes['split_sha256']),(parent_file,identity['init_fso_delta_sha256'])]:
            if digest(filename)!=value: raise ValueError('Initialization/split changed')
        atomic_json(directory/'frozen_verification.json',dict(all_legacy_parameters_unchanged=True,
                    VAE_CLIP_unchanged=True,no_test_split_used=True))
        sync(); verify(directory); report(directory)
        identity['status']='complete'; atomic_json(directory/'run_identity.json',identity)
        files=[f for f in directory.rglob('*') if f.is_file() and f.suffix!='.pth']
        atomic_json(directory/'COMPLETE.json',dict(version=VERSION,rows=rows_done,wall_seconds=time.monotonic()-started,
            result_sha256={str(f.relative_to(directory)):digest(f) for f in files},
            checkpoint_sha256={str(f.relative_to(directory)):digest(f) for f in directory.glob('train_*/*_delta.pth')}))
        verify(directory,complete=True); print('COMPLETE '+str(directory/'report.html'),flush=True)
    except BaseException as exc:
        identity.update(status='failed',error=repr(exc)); atomic_json(directory/'run_identity.json',identity)
        atomic_json(directory/'FAILED.json',dict(error=repr(exc),partial_artifacts_preserved=True)); raise


if __name__=='__main__': main()
