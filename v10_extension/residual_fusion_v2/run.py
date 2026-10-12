"""GPU1: text localization -> fixed-B4 residual controls -> shared waveform evaluation."""
import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from residual_fusion_v2 import VERSION,ARMS,TRAINED_ARMS
from residual_fusion_v2.core import load,TEMPLATE,digest,seed_for,signature,augmentation,donor_index


def main():
    import torch
    import torch.nn.functional as F
    from v10.runtime import add_shared_args,preflight,load_config
    from v10.io import begin_run,atomic_json,atomic_checkpoint
    from v10.model import build_model
    from v10.data import build_loader
    from utils.checkpoint import load_checkpoint
    from utils.metrics import batch_image_metrics,ssim_per_image
    from token_guidance.model import Guidance
    from fso_jscc_v1.model import parameters as semantic_parameters,ContinuousChannel
    from fso_jscc_v1.core import kernels
    from fso_fluctuating_v2.model import Frame,surrogate_context
    from fso_fluctuating_v2.core import paths,transport,schedule,environment
    from fso_fluctuating_v2.calibrate import read_bank
    from full_unfreeze_v1.core import check_bank
    from vae_tail_ablation_v1.model import apply_parent
    from joint_jscc_arch_v1.model import ArchitectureModel,encode_text
    from joint_jscc_arch_v1.report import verify as verify_source
    from joint_jscc_arch_v1.fidelity import check as check_fidelity
    from final_compare_v1.metrics import caption_metrics
    from optical_realism_v3.constellation import constellation_svg
    from eval_v10 import save_image
    from residual_fusion_v2.model import ResidualFusion,image_side,frozen_receive,generate_record
    from residual_fusion_v2.diagnostics import diagnose
    from residual_fusion_v2.report import report,verify

    p=add_shared_args(argparse.ArgumentParser(description=__doc__,allow_abbrev=False))
    p.add_argument('--a-delta',required=True); p.add_argument('--source-run',required=True)
    p.add_argument('--legacy-run',required=True); p.add_argument('--calibration',required=True)
    p.add_argument('--experiment-config',default=str(TEMPLATE)); p.add_argument('--smoke',action='store_true')
    a=p.parse_args(); cfg=load(a.experiment_config)
    if a.smoke: cfg.update(train_samples=8,clean_epochs=1,noise_epochs=1,val_samples=2,draws=1,save_images=2,diagnostic_samples=2,diagnostic_updates=8)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1': raise ValueError('Use physical GPU1 through launch.py')
    if not math.isfinite(a.max_hours) or not 0<a.max_hours<=24: raise ValueError('Invalid overall budget')
    parent=Path(a.source_run).resolve(strict=True); verify_source(parent,complete=True)
    pd=json.loads((parent/'design.json').read_text()); pi=json.loads((parent/'run_identity.json').read_text())
    if pd['smoke'] or pd['config']['max_text_tokens']!=128: raise ValueError('A completed MAIN B4 with128 tokens is required')
    ac=pd['config']; cc=pd['channel_config']; bank=read_bank(a.calibration); check_bank(cc,bank)
    b4file=parent/'train_B4_separate/fso_delta.pth'; b2file=Path(a.legacy_run)/'train_B2_fso_ft/last_delta.pth'
    hashes=dict(base_checkpoint_sha256=digest(a.checkpoint),a_delta_sha256=digest(a.a_delta),split_sha256=digest(a.split_manifest),
        b4_checkpoint_sha256=digest(b4file),b2_checkpoint_sha256=digest(b2file),calibration_sha256=digest(Path(a.calibration)/'bank.json'))
    if any(pi[k]!=hashes[k] for k in ('base_checkpoint_sha256','a_delta_sha256','split_sha256','calibration_sha256')) or pi['init_fso_delta_sha256']!=hashes['b2_checkpoint_sha256']:
        raise ValueError('Different B4/base/A/B2/split/calibration identity')
    a.batch_size,a.num_workers,a.no_amp,a.fixed_mode,a.csi,a.blocks,a.seed=1,0,True,0,'pilot',1,cfg['seed']
    a.min_free_gb,a.min_free_disk_gb=cfg['min_free_gib'],12.
    device=preflight(a); config=load_config(a)
    if device.type!='cuda': raise ValueError('Real experiment requires CUDA; CPU toy tests are separate')
    if (config['model']['image_size'],config['model']['hidden_dim'],config['model']['latent_dim'])!=(128,256,16): raise ValueError('Original128/256/16 model required')
    base=load_checkpoint(a.checkpoint)
    if base.get('split_sha256')!=hashes['split_sha256'] or not str(base.get('version','')).startswith('V10-'): raise ValueError('Nonexact base split/version')
    config['v10']=copy.deepcopy(base['config']['v10']); config['v10'].update(blocks=1,csi='pilot',fixed_mode=0)
    config['model']['vae_use_fp16']=False; del base
    directory,identity=begin_run(a.output_dir,a,config,EXT); started=time.monotonic()
    identity.update(version=VERSION,**hashes,source_run_complete_sha256=digest(parent/'COMPLETE.json'),
        new_source_sha256={f.name:digest(f) for f in sorted(Path(__file__).parent.iterdir()) if f.suffix in ('.py','.json','.md')})
    atomic_json(directory/'run_identity.json',identity)
    def budget():
        if time.monotonic()-started>a.max_hours*3600: raise TimeoutError('Overall cap reached; preserve partial output')
        free,_=torch.cuda.mem_get_info(device)
        if free/2**30<cfg['reserve_free_gib']: raise MemoryError('GPU reserve breached; unrelated processes untouched')
    try:
        legacy,exact=build_model(config,device,a.checkpoint)
        if exact!={'v10_exact':True}: raise ValueError('Nonexact base loading')
        legacy.float(); guide=Guidance(legacy); delta=load_checkpoint(a.a_delta)
        if delta.get('version')!='V10-token-guidance-quick-v1' or delta.get('variant')!='A' or set(delta['delta'])!={n for n,_ in guide.trainable('A')}:
            raise ValueError('Nonexact original A delta')
        legacy.load_state_dict(dict(legacy.state_dict(),**delta['delta']),strict=True); del delta
        guide.set('A'); legacy.base.channel=ContinuousChannel(legacy.base.channel,cc).to(device)
        apply_parent(legacy,semantic_parameters(legacy),load_checkpoint(b2file)); legacy.requires_grad_(False); legacy.eval()
        legacy.base.text_encoder.freeze=True; legacy.base.vae_wrapper.freeze=True
        core=ArchitectureModel(legacy.base,ac,kernels(cc)).to(device).float(); b4=load_checkpoint(b4file)
        if b4.get('version')!='V10-joint-jscc-architecture-v1' or b4.get('arm')!='B4_separate' or b4.get('phase')!='fso' or b4['architecture_config']!=ac:
            raise ValueError('Nonexact B4 architecture checkpoint')
        core.load_state_dict(b4['delta'],strict=True); del b4
        core.configure('B4_separate'); core.requires_grad_(False); core.eval()
        frozen_core={n:signature(v) for n,v in core.state_dict().items()}
        frozen_legacy={n:signature(v) for n,v in legacy.state_dict().items()}
        tok=legacy.base.text_encoder.tokenizer
        def cache(split,count):
            loader=build_loader(config,a.split_manifest,split,count,cfg['seed']); loader.dataset.caption_mode='first'; result=[]
            with torch.no_grad():
                for batch in loader:
                    budget(); image=batch['images']; text=batch['texts'][0]
                    posterior=legacy.base.vae_wrapper.vae.encode(image.to(device)*2-1).latent_dist
                    info=encode_text(legacy.base,text,128,device)
                    result.append(dict(image_id=int(batch['image_ids'][0]),text=text,image=image,
                        latent=(posterior.mean*legacy.base.vae_wrapper.scaling_factor).cpu(),text_features=info['features'],text_mask=info['mask'],
                        target_ids=torch.tensor([info['target_ids']]),target_length=torch.tensor([info['target_length']]),
                        covered_text=info['covered_text'],source_content_tokens=info['source_content_tokens'],covered_content_tokens=info['covered_content_tokens'],
                        truncated=info['truncated'],tokenizer=tok,source_sha256=hashlib.sha256(image.numpy().tobytes()+text.encode()).hexdigest()))
            if len(result)!=count or len({r['image_id'] for r in result})!=count: raise ValueError('Incomplete/duplicate samples')
            return result
        train=cache('train',cfg['train_samples']); val=cache('validation',cfg['val_samples'])
        if {i['image_id'] for i in train}&{i['image_id'] for i in val}: raise ValueError('Train/validation overlap')
        conditions=cc['conditions'][:2] if a.smoke else cc['conditions']
        design=dict(version=VERSION,config=cfg,architecture_config=ac,channel_config=cc,smoke=a.smoke,
            arms=list(ARMS),train_ids=[r['image_id'] for r in train],validation_ids=[r['image_id'] for r in val],
            validation_source_sha256={str(r['image_id']):r['source_sha256'] for r in val},conditions=conditions,
            primary_conditions=[c for c in cc['primary_conditions'] if c in {r['id'] for r in conditions}],
            split='validation',train_forward_steps=cfg['train_samples']*(cfg['clean_epochs']+cfg['noise_epochs']),
            tokenizer_ids=dict(bos=tok.bos_token_id,eos=tok.eos_token_id,pad=tok.pad_token_id),
            diagnostic_conditions=[r['id'] for r in (conditions if a.smoke else [r for r in conditions if r['id'] in ('moderate_p21','strong_p18')])],
            optimizer_steps=math.ceil(cfg['train_samples']/cfg['grad_accum'])*(cfg['clean_epochs']+cfg['noise_epochs']),
            frozen_B4_all_parameters=True,receiver_only_residual=True,source_text_capacity_tokens=128,
            payload_complex_symbols=2304,analog_repeats=1,raw_utf8_wire_bytes=0,environment=environment(cc),
            text_overfit_weights_are_diagnostic_only=True,main_text_decoder_unchanged=True,
            augmentation='same schedule; R0 image/R1 text memory zero or distinct same-channel donor',
            intervention_scope='receiver fusion memory only; original received caption still from unchanged B4',
            primary='R1_joint-R0_image all-frame PSNR/SSIM; exploratory image-cluster validation',
            limitations='one replicate/fixed weather/previously used validation; no source-joint coding, image-only rate-optimal or digital superiority claim')
        atomic_json(directory/'design.json',design)
        atomic_json(directory/'checkpoint_load.json',dict(exact_base=True,exact_A=True,exact_B2=True,exact_B4=True,
            b4_parameter_sha256=frozen_core,legacy_parameter_sha256=frozen_legacy,**hashes))
        with (directory/'calibration_summary.json').open('xb') as f: f.write((Path(a.calibration)/'bank.json').read_bytes())
        (directory/'images').mkdir(); (directory/'constellations').mkdir()
        @torch.no_grad()
        def receive(item,context):
            x=core.source_encode(item['latent'].to(device),item['text_features'].to(device),item['text_mask'].to(device))
            papr,_,wave_samples=core.waveform(x,cc)
            ctx=context or dict(profile='clean',snr=40.,field_seed=0,noise_seed=0)
            frame=Frame('awgn',0.,1,device,context=ctx,config=cc,fixed_mode=0,blocks=1)
            received,rel=frame.payload(x)
            frame.last_stats.update(waveform_papr_db=float(papr[0]),shaped_waveform_samples=wave_samples,papr_is_payload_only_not_full_frame=True)
            out=frozen_receive(core,received,rel,frame.routing_snr,frame.header_ok)
            out.update(payload_sha256=signature(x),received_sha256=signature(received),reliability=rel,
                pilot_snr=frame.routing_snr,available=frame.header_ok)
            return out,received,rel,frame
        def diagnostic_context(condition,index,count):
            trace=paths(cc,condition['weather'],count,('residual-text-diagnosis',condition['weather']))
            ns=seed_for(cfg['seed'],'text-diagnostic',condition['id'],index)
            return dict(transport=transport(cc,condition,trace[index],ns),noise_seed=ns)
        decision=diagnose(core,train[:cfg['diagnostic_samples']],cfg,device,directory,receive,
            dict(conditions=conditions if a.smoke else [r for r in conditions if r['id'] in ('moderate_p21','strong_p18')],make=diagnostic_context),
            budget,atomic_json,atomic_checkpoint,a.smoke)

        def frozen_check():
            checks=dict(all_B4_parameters_unchanged=all(signature(v)==frozen_core[n] for n,v in core.state_dict().items()),
                all_legacy_parameters_unchanged=all(signature(v)==frozen_legacy[n] for n,v in legacy.state_dict().items()),
                all_B4_gradients_none=all(p.grad is None for p in core.parameters()),
                all_legacy_gradients_none=all(p.grad is None for p in legacy.parameters()),no_test_split_used=True)
            if not all(checks.values()): raise RuntimeError('Frozen source/main decoder changed')
            for f,k in ((a.checkpoint,'base_checkpoint_sha256'),(a.a_delta,'a_delta_sha256'),(a.split_manifest,'split_sha256'),
                    (b4file,'b4_checkpoint_sha256'),(b2file,'b2_checkpoint_sha256')):
                if digest(f)!=hashes[k]: raise RuntimeError('Historical input file changed')
            atomic_json(directory/'frozen_verification.json',checks)
        def finish(status):
            frozen_check(); identity['status']=status; atomic_json(directory/'run_identity.json',identity)
            report(directory,status); verify(directory,complete=False,status=status)
            files=[f for f in directory.rglob('*') if f.is_file() and f.suffix!='.pth']
            marker='COMPLETE.json' if status=='complete' else 'DIAGNOSTIC_COMPLETE.json'
            atomic_json(directory/marker,dict(version=VERSION,status=status,wall_seconds=time.monotonic()-started,
                result_sha256={str(f.relative_to(directory)):digest(f) for f in files},
                checkpoint_sha256={str(f.relative_to(directory)):digest(f) for f in directory.rglob('*.pth')}))
            verify(directory,complete=True,status=status)
        if not decision['proceed']:
            finish('diagnostic_stop')
            print('DIAGNOSTIC STOP: text overfit quality gate not met; no residual main training. Return diagnostic records; do not loosen gate.',flush=True)
            return
        proof=check_fidelity(core,train,bank,cc,dict(ac,seed=cfg['seed']),'fixed_B4',device)
        atomic_json(directory/'fidelity_fixed_B4.json',proof)
        if not proof['passed']: raise RuntimeError('Fixed B4 current-payload surrogate fidelity failed')
        residual=ResidualFusion(256,cfg['residual_hidden']).to(device).float()
        initial={n:v.detach().cpu().clone() for n,v in residual.state_dict().items()}
        atomic_json(directory/'residual_initial.json',dict(parameter_counts={n:p.numel() for n,p in residual.named_parameters()},
            initial_parameter_sha256={n:signature(p) for n,p in residual.named_parameters()}))
        states={}
        for arm in TRAINED_ARMS:
            residual.load_state_dict(initial,strict=True); residual.train(False); folder=directory/('train_'+arm); folder.mkdir()
            torch.cuda.reset_peak_memory_stats(device); arm_started=time.monotonic()
            opt=torch.optim.AdamW(residual.parameters(),lr=cfg['residual_lr'],weight_decay=1e-5)
            covered=set(); steps=updates=0
            with (folder/'history.jsonl').open('x') as history:
                for phase,epochs in [('clean',cfg['clean_epochs']),('fso',cfg['noise_epochs'])]:
                    for epoch in range(epochs):
                        order=__import__('numpy').random.default_rng(seed_for(cfg['seed'],'residual-order',phase,epoch)).permutation(len(train))
                        for pos,ix in enumerate(order):
                            budget(); ix=int(ix); item=train[ix]; ns=seed_for(cfg['seed'],'residual-train-noise',phase,epoch,pos)
                            bi=None if phase=='clean' else schedule(bank['config'],epoch,pos,bank)
                            ctx=None if bi is None else surrogate_context(bank,bi,ns)
                            out,_,_,frame=receive(item,ctx); aug=augmentation(cfg,phase,epoch,pos); donor=None
                            side=image_side(out['image_memory']) if arm=='R0_image' else out['text_memory']
                            if aug=='zero': side=torch.zeros_like(side)
                            if aug=='shuffle':
                                donor=donor_index(ix,len(train)); other,_,_,other_frame=receive(train[donor],ctx)
                                side=image_side(other['image_memory']) if arm=='R0_image' else other['text_memory']; del other,other_frame
                            latent,gate=residual(out['latent'],out['image_memory'],side,out['reliability'],out['pilot_snr'],out['available'])
                            image=legacy.base.vae_wrapper.decode_latents(latent)
                            with torch.no_grad(): base_image=legacy.base.vae_wrapper.decode_latents(out['latent']); base_mse=F.mse_loss(base_image,item['image'].to(device))
                            pix=F.mse_loss(image,item['image'].to(device)); lat=F.mse_loss(latent,item['latent'].to(device))
                            structure=1-ssim_per_image(image,item['image'].to(device)).mean()
                            loss=(.75*pix+.25*lat+.35*structure+cfg['protection_weight']*F.relu(pix-base_mse))*out['available'].to(image).float().mean()
                            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite residual loss')
                            divisor=min(cfg['grad_accum'],len(order)-(pos//cfg['grad_accum'])*cfg['grad_accum'])
                            (loss/divisor).backward(); steps+=1
                            if (pos+1)%cfg['grad_accum']==0 or pos+1==len(order):
                                norm=torch.nn.utils.clip_grad_norm_(residual.parameters(),1.)
                                if not torch.isfinite(norm): raise FloatingPointError('Nonfinite residual gradient')
                                covered.update(n for n,p in residual.named_parameters() if p.grad is not None and bool(p.grad.abs().max()>0))
                                opt.step(); opt.zero_grad(set_to_none=True); updates+=1
                            history.write(json.dumps(dict(arm=arm,phase=phase,epoch=epoch+1,image_id=item['image_id'],
                                forward_step=steps,optimizer_step=updates,noise_seed=ns,bank_index=bi,augmentation=aug,
                                donor_id=None if donor is None else train[donor]['image_id'],task_loss_applied=bool(out['available'][0]),
                                pixel_mse=float(pix.detach()),base_pixel_mse=float(base_mse),latent_mse=float(lat.detach()),loss=float(loss.detach()),
                                gate=float(gate.detach()[0]),elapsed_seconds=time.monotonic()-arm_started))+'\n')
                            if steps%100==0:
                                history.flush(); atomic_json(directory/'progress.json',dict(stage='residual_training',arm=arm,step=steps,
                                    expected_steps=design['train_forward_steps'],elapsed_seconds=time.monotonic()-arm_started))
                                print('TRAIN {} {} step={}/{} pix={:.5f} gate={:.3f} elapsed={:.1f}s'.format(arm,phase,steps,design['train_forward_steps'],float(pix),float(gate[0]),time.monotonic()-arm_started),flush=True)
                            if time.monotonic()-arm_started>cfg['arm_max_hours']*3600: raise TimeoutError('Residual arm cap')
                            del out,frame,latent,image,base_image,loss,pix,lat,structure,side
            changes={n:float((p.detach().cpu()-initial[n]).square().sum()) for n,p in residual.named_parameters()}
            if any(sum(v for n,v in changes.items() if n.startswith(prefix))<=0 for prefix in ('query.','memory.','attention.','gate.','output.')):
                raise RuntimeError('Required residual module did not update')
            checkpoint=folder/'last_delta.pth'; atomic_checkpoint(checkpoint,dict(version=VERSION,arm=arm,b4_checkpoint_sha256=hashes['b4_checkpoint_sha256'],
                delta={n:v.detach().cpu().clone() for n,v in residual.state_dict().items()}))
            payload=load_checkpoint(checkpoint); residual.load_state_dict(payload['delta'],strict=True)
            if any(signature(v)!=signature(payload['delta'][n]) for n,v in residual.state_dict().items()): raise RuntimeError('Residual checkpoint reload differs')
            states[arm]=payload['delta']
            atomic_json(folder/'COMPLETE.json',dict(arm=arm,steps=steps,optimizer_steps=updates,checkpoint_sha256=digest(checkpoint),
                checkpoint_roundtrip_verified=True,parameter_changes=changes,observed_gradient_names=sorted(covered),
                trainable_parameters=sum(p.numel() for p in residual.parameters()),wall_seconds=time.monotonic()-arm_started,
                training_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                training_peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,initialization_sha256={n:signature(initial[n]) for n,_ in residual.named_parameters()}))

        # One actual waveform per physical frame. Every decoder/intervention reuses identical tensors.
        done=0
        with (directory/'samples.jsonl').open('x',encoding='utf-8') as samples,(directory/'events.jsonl').open('x') as events:
            for condition in [dict(id='clean',weather='weak',launch_signal_dbm=0.)]+conditions:
                for draw in range(1 if condition['id']=='clean' else cfg['draws']):
                    trace=paths(cc,condition['weather'],len(val),('residual-evaluation',condition['weather'],draw)); frames=[]
                    for i,item in enumerate(val):
                        budget(); frame_started=time.monotonic(); ns=seed_for(cfg['seed'],'residual-eval-noise',condition['id'],draw,i)
                        ctx=None if condition['id']=='clean' else dict(transport=transport(cc,condition,trace[i],ns),noise_seed=ns)
                        out,received,rel,frame=receive(item,ctx)
                        # CPU storage prevents receiver-cache GPU memory from growing with validation size.
                        packed={k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in out.items()}
                        packed.update(stats=frame.last_stats,noise_seed=ns,weather_path=None if ctx is None else trace[i],
                            events=[] if frame.transport is None else copy.deepcopy(frame.transport.events),
                            generated=generate_record(core.language,out['text_memory'],tok))
                        packed['shared_frame_offline_seconds']=time.monotonic()-frame_started
                        if i<cfg['save_images'] and frame.transport is not None and frame.transport.constellations:
                            packed['constellation_file']='constellations/{}__{}__d{}.svg'.format(item['image_id'],condition['id'],draw)
                            constellation_svg(frame.transport.constellations,directory/packed['constellation_file'])
                        frames.append(packed); del out,received,rel,frame
                    for i,(item,packed) in enumerate(zip(val,frames)):
                        budget(); frame_key='{}__{}__d{}'.format(item['image_id'],condition['id'],draw)
                        out={k:v.to(device) if torch.is_tensor(v) else v for k,v in packed.items()}
                        for event in packed['events']: events.write(json.dumps(dict(event,frame_key=frame_key,image_id=item['image_id'],condition=condition['id'],draw=draw),allow_nan=False)+'\n')
                        events.flush(); full=caption_metrics(item['text'],packed['generated']['received_caption']); covered=caption_metrics(item['covered_text'],packed['generated']['received_caption'])
                        for arm in ARMS:
                            decoder_started=time.monotonic()
                            with torch.no_grad():
                                if arm in ('B4_frozen','R1_off'): latent=out['latent']; gate=latent.new_zeros(1); side=None
                                else:
                                    trained='R0_image' if arm=='R0_image' else 'R1_joint'; residual.load_state_dict(states[trained],strict=True); residual.eval()
                                    side=image_side(out['image_memory']) if arm=='R0_image' else out['text_memory']
                                    if arm=='R1_zero_text': side=torch.zeros_like(side)
                                    if arm=='R1_shuffled_text': side=frames[donor_index(i,len(val))]['text_memory'].to(device)
                                    latent,gate=residual(out['latent'],out['image_memory'],side,out['reliability'],out['pilot_snr'],out['available'])
                                image=legacy.base.vae_wrapper.decode_latents(latent); metric=batch_image_metrics(image,item['image'].to(device))
                            row=dict(packed['stats'],arm=arm,frame_key=frame_key,image_id=item['image_id'],condition=condition['id'],draw=draw,
                                source_sha256=item['source_sha256'],noise_seed=packed['noise_seed'],weather_path=packed['weather_path'],
                                payload_sha256=packed['payload_sha256'],received_sha256=packed['received_sha256'],latent_sha256=signature(latent),
                                fusion_memory_sha256=None if side is None else signature(side),fusion_gate=float(gate[0]),
                                fusion_donor_id=val[donor_index(i,len(val))]['image_id'] if arm=='R1_shuffled_text' else None,
                                shared_frame_offline_seconds=packed['shared_frame_offline_seconds'],
                                decoder_offline_seconds=time.monotonic()-decoder_started,offline_timing_not_hardware_latency=True,
                                receiver_available=bool(out['available'][0]),payload_complex_symbols=2304,analog_repeats=1,raw_utf8_wire_bytes=0,
                                pre_fec_ber=None,post_fec_ber=None,psnr=float(metric['psnr'][0]),ssim=float(metric['ssim'][0]),
                                source_caption_evaluation_only=item['text'],covered_reference_evaluation_only=item['covered_text'],
                                source_content_tokens=item['source_content_tokens'],covered_content_tokens=item['covered_content_tokens'],input_truncated=item['truncated'],
                                covered_normalized_wer=covered['normalized_wer'],main_text_decoder_unchanged=True,neural_input_capacity_tokens=128,
                                **packed['generated'],**full)
                            if i<cfg['save_images']:
                                row['source_image_file']='images/{}__source.png'.format(item['image_id'])
                                if not (directory/row['source_image_file']).exists(): save_image(item['image'][0],directory/row['source_image_file'])
                                row['image_file']='images/{}__{}__{}__d{}.png'.format(item['image_id'],arm,condition['id'],draw)
                                save_image(image[0],directory/row['image_file'])
                                if 'constellation_file' in packed: row['constellation_file']=packed['constellation_file']
                                for name in ('image_file','source_image_file','constellation_file'):
                                    if name in row: row[name+'_sha256']=digest(directory/row[name])
                            samples.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); samples.flush(); done+=1
                        print('EVAL shared frame {} rows={} available={}'.format(frame_key,done,bool(out['available'][0])),flush=True)
                        atomic_json(directory/'progress.json',dict(stage='evaluation',rows=done,expected_rows=len(ARMS)*len(val)*(1+len(conditions)*cfg['draws'])))
                        del out,image,latent
                    del frames
        finish('complete'); print('COMPLETE '+str(directory/'report.html'),flush=True)
    except BaseException as exc:
        identity.update(status='failed',error=repr(exc)); atomic_json(directory/'run_identity.json',identity)
        atomic_json(directory/'FAILED.json',dict(error=repr(exc),partial_artifacts_preserved=True)); raise


if __name__=='__main__': main()
