"""Generalize clean V4 text first, then matched C/I/T joint screening on GPU1."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
EXT=Path(__file__).resolve().parents[1]; PROJECT=EXT.parent; sys.path.insert(0,str(EXT))
from clip_bart_joint_v5 import VERSION,ARMS
from clip_bart_joint_v5.core import load,TEMPLATE,digest,geometry,text_gate


def main():
    import torch
    from v10.runtime import add_shared_args,preflight
    from v10.io import atomic_json,atomic_checkpoint
    from bart_clip_joint_v1.backbones import load_bart,load_clip
    from bart_clip_joint_v1.core import tensor_signature
    from bart_latent_codec_v4.core import load as text_config
    from bart_latent_codec_v4.model import LatentCodec
    from bart_latent_codec_v4.data import cache as text_cache
    from bart_latent_codec_v4.workflow import train as text_train,controls as private_controls,signature,frozen_check
    from bart_latent_codec_v4.report import verify as verify_text_source
    from text_recovery_v3.model import consistency
    from text_jscc_v1_fix1.legacy import load_reference
    from clip_bart_joint_v5.data import cache
    from clip_bart_joint_v5.model import JointCodec
    from clip_bart_joint_v5.workflow import train,text_controls
    from clip_bart_joint_v5.evaluate import evaluate
    from clip_bart_joint_v5.report import finish
    p=add_shared_args(argparse.ArgumentParser(description=__doc__,allow_abbrev=False))
    for k in ('bart-path','source-run','legacy-run','a-delta','text-screen'): p.add_argument('--'+k,required=True)
    p.add_argument('--experiment-config',default=str(TEMPLATE)); p.add_argument('--smoke',action='store_true')
    p.add_argument('--noise',action='store_true',help='Also paired clean continuation/FSO proxy training after own quality gates')
    a=p.parse_args(); c=load(a.experiment_config)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1': raise ValueError('Use launcher, physical GPU1')
    if not math.isfinite(a.max_hours) or not 0<a.max_hours<=24: raise ValueError('Finite stop cap')
    if Path(a.output_dir).exists(): raise FileExistsError('Use NEW output')
    # Real successful screen identity only; TRAIN8 weights NEVER initialize this run.
    source_proof=verify_text_source(a.text_screen)
    if source_proof['smoke'] or not all(source_proof['quality_success'].values()): raise ValueError('Completed V4 text diagnosis required')
    a.batch_size,a.num_workers,a.no_amp,a.blocks,a.fixed_mode,a.csi,a.seed=1,0,True,1,0,'pilot',c['seed']
    a.min_free_gb,a.min_free_disk_gb=c['min_free_gib'],12.; device=preflight(a)
    if device.type!='cuda': raise RuntimeError('Actual saved models require GPU')
    tc=text_config(); tc.update(seed=c['seed'],train_samples=c['text_train_samples'],val_samples=c['val_samples'])
    if a.smoke:
        c.update(text_train_samples=8,text_updates=8,train_samples=8,val_samples=2,clean_epochs=1,continue_epochs=1,joint_warm_steps=2,save_images=2)
        tc.update(train_samples=8,val_samples=2,batch_size=2)
    root=Path(a.output_dir); root.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    files=[f for f in Path(__file__).parent.iterdir() if f.suffix in ('.py','.json','.md')]
    (root/'source_snapshot').mkdir()
    for f in files:
        with (root/'source_snapshot'/f.name).open('xb') as writer: writer.write(f.read_bytes())
    identity=dict(version=VERSION,status='running',python=sys.executable,torch=str(torch.__version__),cuda=torch.version.cuda,
        pid=os.getpid(),argv=sys.argv,config=c,text_config=tc,smoke=a.smoke,noise=a.noise,
        split_sha256=digest(a.split_manifest),source_sha256={f.name:digest(f) for f in files},
        V4_source_screen_proof=source_proof,TRAIN8_weights_loaded=False)
    atomic_json(root/'run_identity.json',identity)
    def budget():
        if time.monotonic()-started>a.max_hours*3600: raise TimeoutError('Finite run cap reached')
        free,_=torch.cuda.mem_get_info(device)
        if free/2**30<c['reserve_free_gib']: raise MemoryError('GPU free reserve breached; no unrelated process action')
    try:
        budget(); bart,tok,br=load_bart(a.bart_path,device); bart_before=signature(bart)
        torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
        text_model=LatentCodec(bart,tc,64).to(device).float().eval()
        texts,tr=text_cache(a,tc,bart,tok,'train',c['text_train_samples'],device,budget)
        val_texts,vr=text_cache(a,tc,bart,tok,'validation',c['val_samples'],device,budget)
        if set(tr['image_ids'])&set(vr['image_ids']): raise ValueError('Text split overlap')
        native=consistency(bart,texts[0],device)
        if max(native['incremental_max_logit_error'],native['cached_source_max_logit_error'])>c['forward_tolerance']: raise RuntimeError('BART cache/interface mismatch')
        atomic_json(root/'bart_interface_audit.json',native)
        (root/'text_prefit').mkdir()
        text_state=text_train(text_model,texts,tc,'P0_ce',device,root/'text_prefit/clean',tok,budget,
            atomic_json,atomic_checkpoint,c['text_updates'],bart_before,stage='clean')
        vg=text_gate(private_controls(text_model,val_texts,tok,device,budget,root/'text_prefit/validation_controls.jsonl','P0_ce'),c)
        atomic_json(root/'text_prefit/validation_gate.json',vg)
        design=dict(version=VERSION,config=c,text_config=tc,arms=list(ARMS),geometry=geometry(),
            text_train_view=tr,text_validation_view=vr,TRAIN8_weights_loaded=False,
            fresh_text_initialization=True,backbones_frozen=True,source_screen_proof=source_proof,
            separate_scope='neural source/RX block isolation in one equally charged physical frame; not two packets or best resource allocation',
            no_clip_scope='remove CLIP TEXT SOURCE only; same CLIP TRAIN loss targets retained',
            inference='actual unrestricted greedy; C/I/T only; no ground-truth receiver inputs',
            stage='text_prefit',proxy_scope='nominal Gamma-Gamma/pilot symbol proxy, NOT full waveform matching',noise=a.noise)
        atomic_json(root/'design.json',design); statuses={}
        if not vg['passed'] and not a.smoke:
            statuses={arm:dict(status='text_prefit_stop',clean_passed=False,trained=False,noise_trained=False) for arm in ARMS}
        else:
            # B4/VAE/CLIP loads only after private text generalization gate.
            old,reference,cc,hashes=load_reference(a,device)
            clip,processor,cr=load_clip(a.text_model_path,device)
            vae=old.base.vae_wrapper.vae.eval().requires_grad_(False); scaling=float(old.base.vae_wrapper.scaling_factor)
            if abs(scaling-float(vae.config.scaling_factor))>1e-8: raise ValueError('VAE scaling differs')
            frozen=[(bart,bart_before),(old,signature(old)),(reference,signature(reference)),(clip,signature(clip))]
            atomic_json(root/'checkpoint_load.json',dict(bart=br,clip=cr,**hashes,full_backbones_frozen=True))
            items,ir=cache(a,c,old,reference,text_model,tok,clip,processor,'train',c['train_samples'],device,budget)
            val,iv=cache(a,c,old,reference,text_model,tok,clip,processor,'validation',c['val_samples'],device,budget)
            if set(ir['image_ids'])&set(iv['image_ids']): raise ValueError('Image/text split overlap')
            if a.smoke: cc=dict(cc,conditions=cc['conditions'][:1])
            design.update(stage='joint',image_train_view=ir,image_validation_view=iv,channel_config=cc)
            atomic_json(root/'design.json',design)
            initial_hashes=None; parameter_count=None
            for arm in ARMS:
                budget(); torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
                model=JointCodec(bart,reference,tc,c,clip.config.projection_dim).to(device).float().eval()
                model.text.restore(text_state); model.configure(arm)
                initial=model.bridge_state(); hashes={n:tensor_signature(v) for n,v in initial.items()}
                count=sum(v.numel() for v in initial.values())
                if initial_hashes is not None and (hashes!=initial_hashes or parameter_count!=count): raise RuntimeError('Paired initialization/registered count differs')
                initial_hashes=hashes; parameter_count=count
                folder=root/arm; folder.mkdir(); atomic_json(folder/'initial_state_hashes.json',hashes)
                clean_state=train(model,items,c,tc,vae,scaling,device,folder/'clean','clean',tok,budget,frozen,atomic_json,atomic_checkpoint)
                gate=text_gate(text_controls(model,val,tok,device,budget,folder/'clean_controls.jsonl'),c)
                atomic_json(folder/'clean_gate.json',gate)
                # Always report clean image/text even if this own text gate fails.
                evaluate(model,val,c,cc,vae,scaling,tok,device,budget,folder/'eval_clean','clean',atomic_json)
                statuses[arm]=dict(status='clean_passed' if gate['passed'] else 'clean_validation_stop',clean_passed=gate['passed'],trained=True,noise_trained=False)
                if a.noise and (gate['passed'] or a.smoke):
                    for phase in ('clean_continue','fso'):
                        model.restore(clean_state)
                        train(model,items,c,tc,vae,scaling,device,folder/phase,phase,tok,budget,frozen,atomic_json,atomic_checkpoint)
                        text_controls(model,val,tok,device,budget,folder/(phase+'_controls.jsonl'))
                        evaluate(model,val,c,cc,vae,scaling,tok,device,budget,folder/('eval_'+phase),phase,atomic_json)
                    statuses[arm].update(status='noise_evaluated',noise_trained=True)
                atomic_json(root/'arm_status.json',statuses); del model,initial; torch.cuda.empty_cache()
            for module,before in frozen: frozen_check(module,before)
        frozen_check(bart,bart_before)
        if digest(a.split_manifest)!=identity['split_sha256'] or any(digest(f)!=identity['source_sha256'][f.name] for f in files): raise RuntimeError('Input/runtime source changed')
        for view in [tr,vr]+[design[k] for k in ('image_train_view','image_validation_view') if k in design]:
            if digest(view['annotation_path'])!=view['annotation_sha256']: raise RuntimeError('Original COCO labels changed')
        atomic_json(root/'arm_status.json',statuses); atomic_json(root/'frozen_verification.json',dict(passed=True,backbones_unchanged=True,original_data_unchanged=True))
        identity.update(status='smoke_complete' if a.smoke else 'screen_complete',wall_seconds=time.monotonic()-started)
        atomic_json(root/'run_identity.json',identity); finish(root)
        print('V5 screen finished; negative gates retained. Send records, not weights.',flush=True)
    except BaseException as exc:
        atomic_json(root/'FAILED.json',dict(type=type(exc).__name__,message=str(exc),wall_seconds=time.monotonic()-started)); raise


if __name__=='__main__': main()
