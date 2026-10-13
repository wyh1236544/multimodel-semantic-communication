"""Bounded receiver screening on physical GPU1; no source/Common retraining."""
import argparse
import math
import os
from pathlib import Path
import sys
import time
EXT=Path(__file__).resolve().parents[1]; PROJECT=EXT.parent; sys.path.insert(0,str(EXT))
from bidirectional_rx_v7 import VERSION,ARMS
from bidirectional_rx_v7.core import load,TEMPLATE,digest,read,plan,geometry,array_hash


def main():
    import torch
    from v10.runtime import add_shared_args,preflight
    from v10.io import atomic_json,atomic_checkpoint
    from bart_clip_joint_v1.backbones import load_bart,load_clip
    from bart_clip_joint_v1.core import tensor_signature
    from bart_latent_codec_v4.workflow import signature,frozen_check
    from text_jscc_v1_fix1.legacy import load_reference
    from clip_bart_joint_v5.data import cache,source,clean_args
    from clip_bart_joint_v6.model import JointCodec
    from clip_bart_joint_v6.report import verify as parent_verify
    from bidirectional_rx_v7.model import ReceiverAdapter
    from bidirectional_rx_v7.workflow import train
    from bidirectional_rx_v7.evaluate import evaluate
    from bidirectional_rx_v7.report import finish
    from bidirectional_rx_v7.features import receiver_features
    p=add_shared_args(argparse.ArgumentParser(description=__doc__,allow_abbrev=False))
    for k in ('bart-path','source-run','legacy-run','a-delta','parent-screen'): p.add_argument('--'+k,required=True)
    p.add_argument('--experiment-config',default=str(TEMPLATE)); p.add_argument('--mode',choices=('smoke','diagnostic','screen','pipeline'),required=True)
    a=p.parse_args(); c=load(a.experiment_config); schedule=plan(c,a.mode)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1': raise ValueError('Use launcher physical GPU1')
    if not math.isfinite(a.max_hours) or not 0<a.max_hours<=24: raise ValueError('Finite run cap')
    if Path(a.output_dir).exists(): raise FileExistsError('Existing output preserved')
    parent=Path(a.parent_screen).resolve(strict=True); proof=parent_verify(parent); pi=read(parent/'run_identity.json'); pd=read(parent/'design.json')
    if proof['smoke'] or not proof['text_prefit_passed'] or pi['text_milestones'][-1]!=9000 or pi['config']['seed']!=c['seed']:
        raise ValueError('Completed V6 formal9000 text milestone required')
    if not all(s['trained'] and s['clean_passed'] for s in proof['arm_status'].values()): raise ValueError('V6 source arms not complete')
    text_path=parent/'text_prefit/clean/last.pth'; joint_path=parent/'J0_separate/clean/last.pth'
    for path,done in ((text_path,parent/'text_prefit/clean/TRAIN_COMPLETE.json'),(joint_path,parent/'J0_separate/clean/TRAIN_COMPLETE.json')):
        if digest(path)!=read(done)['checkpoint_sha256']: raise ValueError('Parent weights drift')
    a.batch_size,a.num_workers,a.no_amp,a.blocks,a.fixed_mode,a.csi,a.seed=1,0,True,1,0,'pilot',c['seed']
    a.min_free_gb,a.min_free_disk_gb=c['min_free_gib'],12.; device=preflight(a)
    if device.type!='cuda': raise RuntimeError('Actual pretrained smoke requires CUDA')
    root=Path(a.output_dir); root.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    folder=Path(__file__).parent; files=sorted(f for f in folder.iterdir() if f.suffix in ('.py','.json','.md'))
    (root/'source_snapshot').mkdir()
    for f in files:
        with (root/'source_snapshot'/f.name).open('xb') as writer: writer.write(f.read_bytes())
    identity=dict(version=VERSION,status='running',mode=a.mode,config=c,schedule=schedule,python=sys.executable,torch=str(torch.__version__),
        cuda=torch.version.cuda,pid=os.getpid(),argv=sys.argv,split_sha256=digest(a.split_manifest),
        source_sha256={f.name:digest(f) for f in files},parent_proof=proof,
        text_checkpoint_sha256=digest(text_path),joint_checkpoint_sha256=digest(joint_path),
        text_global_milestone=9000,TRAIN8_weights_loaded=False,optimizer_resume=False)
    atomic_json(root/'run_identity.json',identity)
    def budget():
        if time.monotonic()-started>a.max_hours*3600: raise TimeoutError('Run time cap exceeded')
        free,_=torch.cuda.mem_get_info(device)
        if free/2**30<c['reserve_free_gib']: raise MemoryError('GPU reserve breached; other processes untouched')
    try:
        budget(); bart,tok,br=load_bart(a.bart_path,device); old,reference,cc,hashes=load_reference(a,device)
        clip,processor,cr=load_clip(a.text_model_path,device); tc=dict(pi['text_config'])
        torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
        base=JointCodec(bart,reference,tc,pi['config'],clip.config.projection_dim).to(device).float().eval()
        j=torch.load(str(joint_path),map_location='cpu'); t=torch.load(str(text_path),map_location='cpu')
        if j['arm']!='J0_separate' or j['phase']!='clean' or j['config']!=pi['config']: raise ValueError('Wrong J0 checkpoint')
        if t['arm']!='P0_ce' or t['width']!=64 or t['updates']!=pi['config']['text_block_updates'] or t['config']!=tc: raise ValueError('Wrong text codec checkpoint')
        base.restore(j['weights']); base.configure('J0_separate'); base.text.restore(t['state'])
        base.requires_grad_(False).eval(); del j,t
        vae=old.base.vae_wrapper.vae.requires_grad_(False).eval(); scaling=float(old.base.vae_wrapper.scaling_factor)
        if abs(scaling-float(vae.config.scaling_factor))>1e-8: raise ValueError('VAE scaling differs')
        frozen=[(base,signature(base)),(old,signature(old)),(clip,signature(clip))]
        atomic_json(root/'checkpoint_load.json',dict(bart=br,clip=cr,**hashes,
            base_recipe='V6 J0 C/I/RX + independent V6 private text codec at global9000',
            historical_V6_replay=False,live_B0_baseline_required=True,all_existing_parameters_frozen=True))
        items,ir=cache(a,c,old,reference,base.text,tok,clip,processor,'train',schedule['train_samples'],device,budget)
        val,iv=cache(a,c,old,reference,base.text,tok,clip,processor,'validation',schedule['val_samples'],device,budget)
        if set(ir['image_ids'])&set(iv['image_ids']): raise ValueError('Train/validation overlap')
        for receipt,key in ((ir,'image_train_view'),(iv,'image_validation_view')):
            oldview=pd[key]
            if receipt['annotation_sha256']!=oldview['annotation_sha256'] or receipt['image_ids']!=oldview['image_ids'][:len(receipt['image_ids'])]:
                raise ValueError('Original COCO selection differs from parent')
        for item in items+val:
            budget()
            with torch.no_grad():
                wire=source(base,item,device); received=base.receive(*clean_args(wire))
            if wire.shape!=(1,6144) or not torch.isfinite(wire).all(): raise ValueError('Wrong source wire')
            if abs(float(wire.square().mean())-1)>1e-4: raise ValueError('Non-unit TX power')
            item['wire']=wire.cpu(); item['text_anchor']=received['text']['memory'].cpu()
        if a.mode=='smoke': cc=dict(cc,conditions=cc['conditions'][:1])
        design=dict(version=VERSION,config=c,schedule=schedule,geometry=geometry(),arms=list(ARMS),image_train_view=ir,image_validation_view=iv,
            channel_config=cc,text_config=tc,training_source='original COCO train2017',validation_scope='reused exploratory validation; NOT independent test',
            source_wire_hashes={str(x['image_id']):array_hash(x['wire'].numpy()) for x in items+val},
            source_frozen=True,common_not_retrained=True,raw_text_not_transmitted=True,analog_repeats=1,
            proxy_scope='symbol Gamma-Gamma + pointing + additive noise + pilot estimate; no waveform sync/rejection/weather correlation during training',
            synthetic_dropout_scope='25% private slots, offline TRAIN + separately labelled diagnostic; full resource still charged',
            actual_receiver_inputs='received C/I/T and pilot/availability only; no source text/images/masks/length',
            receiver_visual_semantics='frozen VAE decode of received latent -> bicubic CLIP size -> frozen CLIP; no clean source feature',
            no_diffusion=True,registered_size_not_equal_active_compute=True)
        atomic_json(root/'design.json',design)
        torch.manual_seed(c['seed']); adapter=ReceiverAdapter(bart.config.d_model,c,clip.config.projection_dim).to(device).float().eval()
        initial=adapter.saved(); initial_hashes={n:tensor_signature(v) for n,v in initial.items()}; states={}
        atomic_json(root/'initial_state_hashes.json',initial_hashes)
        with torch.no_grad():
            original=receiver_features(base.receive(*clean_args(items[0]['wire'].to(device))),vae,scaling,clip,processor)
            z=adapter(original,original['latent'].new_full((1,),40),original['latent'].new_ones(1))
        if not torch.equal(z['latent'],original['latent']) or not torch.equal(z['text']['memory'],original['text']['memory']): raise RuntimeError('Initial adapter must exactly preserve B0')
        atomic_json(root/'initial_preservation.json',dict(passed=True,zero_output_exact_B0=True))
        for phase in schedule['phases']:
            for arm in ARMS:
                adapter.configure(arm); adapter.restore(initial)
                states[(arm,phase)]=train(adapter,base,items,c,tc,vae,scaling,clip,processor,device,root/arm/phase,phase,schedule['epochs'],budget,frozen,atomic_json,atomic_checkpoint)
        evaluate(adapter,base,states,val,c,cc,vae,scaling,clip,processor,tok,device,budget,root/'evaluation',schedule['phases'],atomic_json)
        for module,before in frozen: frozen_check(module,before)
        if digest(text_path)!=identity['text_checkpoint_sha256'] or digest(joint_path)!=identity['joint_checkpoint_sha256'] or digest(a.split_manifest)!=identity['split_sha256']:
            raise RuntimeError('Existing input weights/split changed')
        if any(digest(f)!=identity['source_sha256'][f.name] for f in files): raise RuntimeError('Runtime source changed')
        for view in (ir,iv):
            if digest(view['annotation_path'])!=view['annotation_sha256']: raise RuntimeError('Original captions changed')
        atomic_json(root/'frozen_verification.json',dict(passed=True,backbones_unchanged=True,old_weights_unchanged=True,original_data_unchanged=True,
            base_parameter_count=sum(p.numel() for p in base.parameters()),new_parameter_count=sum(p.numel() for p in adapter.parameters())))
        identity.update(status='complete',wall_seconds=time.monotonic()-started); atomic_json(root/'run_identity.json',identity); finish(root)
        print('V7 finished. Execution completion is NOT evidence of joint benefit.',flush=True)
    except BaseException as exc:
        atomic_json(root/'FAILED.json',dict(type=type(exc).__name__,message=str(exc),wall_seconds=time.monotonic()-started)); raise


if __name__=='__main__': main()
