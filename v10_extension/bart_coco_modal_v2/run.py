"""Independent COCO image/text screens; frozen backbones, actual compact FSO frames."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from bart_coco_modal_v2 import VERSION,MODES
from bart_coco_modal_v2.core import load,TEMPLATE,digest,tensor_signature,quality_gate


def main():
    import torch
    from v10.runtime import add_shared_args,preflight
    from v10.io import begin_run,atomic_json,atomic_checkpoint
    from text_jscc_v1_fix1.legacy import load_reference
    from bart_clip_joint_v1.backbones import load_bart,load_clip
    from bart_coco_modal_v2.model import ModalBridge
    from bart_coco_modal_v2.data import cache
    from bart_coco_modal_v2.workflow import train,text_controls,text_row,frozen_signatures,verify_frozen
    from bart_coco_modal_v2.evaluate import evaluate,common_controls
    from bart_coco_modal_v2.channel import resource
    from bart_coco_modal_v2.report import report,seal,verify
    from utils.checkpoint import load_checkpoint
    p=add_shared_args(argparse.ArgumentParser(description=__doc__,allow_abbrev=False))
    for name in ('source-run','legacy-run','a-delta','bart-path'): p.add_argument('--'+name,required=True)
    p.add_argument('--experiment-config',default=str(TEMPLATE))
    p.add_argument('--modes',nargs='+',choices=MODES)
    p.add_argument('--smoke',action='store_true'); a=p.parse_args(); cfg=load(a.experiment_config)
    modes=a.modes or cfg['default_modes']
    if len(set(modes))!=len(modes): raise ValueError('Repeated mode')
    if a.smoke: cfg.update(train_samples=8,val_samples=2,diagnostic_samples=2,diagnostic_updates=8,feature_warmup_updates=2,clean_epochs=1,save_images=2)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1': raise ValueError('Launcher selects physical GPU1 only')
    if not math.isfinite(a.max_hours) or not 0<a.max_hours<=24: raise ValueError('Finite run budget required')
    a.batch_size,a.num_workers,a.no_amp,a.blocks,a.fixed_mode,a.csi,a.seed=1,0,True,1,0,'pilot',cfg['seed']
    a.min_free_gb,a.min_free_disk_gb=cfg['min_free_gib'],12.
    device=preflight(a)
    if device.type!='cuda': raise ValueError('Real pretrained screen requires GPU')
    root,identity=begin_run(a.output_dir,a,cfg,EXT); start=time.monotonic()
    identity.update(version=VERSION,status='running',modes=modes,smoke=a.smoke,
        new_source_sha256={f.name:digest(f) for f in Path(__file__).parent.iterdir() if f.suffix in ('.py','.json','.md')})
    atomic_json(root/'run_identity.json',identity)
    def budget():
        if time.monotonic()-start>a.max_hours*3600: raise TimeoutError('Finite run cap reached; preserve output')
        free,_=torch.cuda.mem_get_info(device)
        if free/2**30<cfg['reserve_free_gib']: raise MemoryError('Free GPU reserve breached; no unrelated process action')
    try:
        old,reference,cc,hashes=load_reference(a,device)
        vae=old.base.vae_wrapper.vae.eval().requires_grad_(False)
        if abs(float(vae.config.scaling_factor)-float(old.base.vae_wrapper.scaling_factor))>1e-8: raise ValueError('VAE scaling mismatch')
        bart,tokenizer,br=load_bart(a.bart_path,device); clip,processor,cr=load_clip(a.text_model_path,device)
        model=ModalBridge(bart,reference,cfg,clip.config.projection_dim).to(device).float().eval()
        initial=model.bridge_state(); frozen=frozen_signatures(model)
        other={n:tensor_signature(v) for n,v in old.state_dict().items()}; clip_before={n:tensor_signature(v) for n,v in clip.state_dict().items()}
        identity.update(**hashes,bart_snapshot=br,clip_snapshot=cr,initial_bridge_sha256={n:tensor_signature(v) for n,v in initial.items()})
        atomic_json(root/'run_identity.json',identity)
        atomic_json(root/'checkpoint_load.json',dict(exact_historical_reference=True,bart=br,clip=cr,
            registered_bridge_parameters=sum(v.numel() for v in initial.values()),
            frozen_bart_parameters=sum(v.numel() for v in bart.parameters()),
            frozen_reference_parameters=sum(v.numel() for v in reference.parameters()),
            CLIP_BART_VAE_unfrozen=False))
        if a.smoke: cc=dict(cc,conditions=cc['conditions'][:2],primary_conditions=[v['id'] for v in cc['conditions'][:2]])
        design=dict(version=VERSION,config=cfg,modes=modes,smoke=a.smoke,channel_config=cc,
            resources={m:resource(cc,m) for m in modes},caption_views={},
            clean_training_only=True,no_test_source_used=True,analog_repeats=1,
            receiver_inputs='compact features, public predeclared mode, pilot SNR/reliability/lock; no original source/mask/length',
            comparison_scope='single-modality functionality, different actual resources/active parameters; not equal-resource joint superiority',
            private_diagnostic_scope='T32 clean TRAIN diagnostic only; C zero, NOT a compact physical baseline',
            native_scope='uncompressed pretrained BART TRAIN diagnostic; not a rate-matched method',
            reconstruction='BART receives C/T features; VAE receives C/I residual; generated text is NOT fed to image decoder',
            selection='fixed last step; no checkpoint selection/resampling, mode-local text quality stops',
            limitations='small screen, one replicate, reused validation, clean-trained not new noise-trained, no hardware or final test')
        atomic_json(root/'design.json',design); statuses={}; all_common=[]
        (root/'images').mkdir(); (root/'constellations').mkdir()
        with (root/'samples.jsonl').open('x',encoding='utf-8') as writer,(root/'events.jsonl').open('x') as events:
            for mode in modes:
                budget(); folder=root/mode; folder.mkdir(); model.configure_mode(mode); model.load_bridge(initial)
                items,tr=cache(a,cfg,mode,old,reference,bart,tokenizer,clip,processor,'train',cfg['train_samples'],device,budget)
                val,vr=cache(a,cfg,mode,old,reference,bart,tokenizer,clip,processor,'validation',cfg['val_samples'],device,budget)
                if set(tr['image_ids'])&set(vr['image_ids']): raise ValueError('Train/validation overlap')
                design['caption_views'][mode]=dict(train=tr,validation=vr); atomic_json(root/'design.json',design)
                diag=items[:cfg['diagnostic_samples']]
                if mode!='image_only':
                    native=[]
                    with torch.no_grad():
                        for item in diag:
                            budget(); row=text_row(item,mode,'clean',model.generate(item['bart_hidden'].to(device),tokenizer))
                            row.update(diagnostic_uncompressed=True,source_split='train'); native.append(row)
                    with (folder/'pretrained_uncompressed.jsonl').open('x',encoding='utf-8') as f:
                        for row in native: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                    # Private-only diagnosis is additive evidence; it cannot globally block image or C+T.
                    model.configure_mode('text_only',private_diagnostic=True); model.load_bridge(initial)
                    train(model,diag,cfg,cc,device,folder/'private_T32_diagnostic','diagnostic',budget,atomic_json,atomic_checkpoint)
                    groups=text_controls(model,diag,tokenizer,device,budget,folder,'private_T32',atomic_json)
                    private=quality_gate(groups,cfg); private['smoke_quality_bypass']=a.smoke
                    atomic_json(folder/'private_T32_gate.json',private)
                    model.configure_mode(mode); model.load_bridge(initial)
                    train(model,diag,cfg,cc,device,folder/'mode_diagnostic','diagnostic',budget,atomic_json,atomic_checkpoint,vae)
                    groups=text_controls(model,diag,tokenizer,device,budget,folder,'mode_overfit',atomic_json)
                    gate=quality_gate(groups,cfg); gate['smoke_quality_bypass']=a.smoke
                    atomic_json(folder/'mode_overfit_gate.json',gate)
                    if not gate['passed'] and not a.smoke:
                        statuses[mode]=dict(status='overfit_stop',evaluated=False,private_T32_passed=private['passed'],gate=gate)
                        atomic_json(root/'mode_status.json',statuses); continue
                model.configure_mode(mode); model.load_bridge(initial)
                train(model,items,cfg,cc,device,folder/'clean','clean',budget,atomic_json,atomic_checkpoint,vae)
                if mode!='image_only':
                    groups=text_controls(model,val,tokenizer,device,budget,folder,'clean_validation',atomic_json)
                    gate=quality_gate(groups,cfg,diagnostic=False); gate['smoke_quality_bypass']=a.smoke
                    atomic_json(folder/'clean_validation_gate.json',gate)
                    if not gate['passed'] and not a.smoke:
                        statuses[mode]=dict(status='clean_validation_stop',evaluated=False,gate=gate)
                        atomic_json(root/'mode_status.json',statuses); continue
                ck=load_checkpoint(folder/'clean/last_bridge.pth')
                if ck['version']!=VERSION or ck['mode']!=mode or ck['config']!=cfg or ck['private_diagnostic']: raise ValueError('Wrong saved checkpoint')
                model.load_bridge(ck['weights']); del ck
                all_common.extend(common_controls(model,val,cfg,device,root,tokenizer,vae,budget))
                count=evaluate(model,val,cfg,cc,device,root,tokenizer,vae,budget,atomic_json,writer,events)
                statuses[mode]=dict(status='smoke_evaluated' if a.smoke else 'evaluated',evaluated=True,rows=count)
                atomic_json(root/'mode_status.json',statuses)
        with (root/'common_controls.jsonl').open('x',encoding='utf-8') as f:
            for row in all_common: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        if any(p.grad is not None for n,p in model.named_parameters() if n.startswith(('bart.','image_reference.'))): raise RuntimeError('Frozen gradient leakage')
        model.configure_mode('joint'); verify_frozen(model,frozen)
        if other!={n:tensor_signature(v) for n,v in old.state_dict().items()} or clip_before!={n:tensor_signature(v) for n,v in clip.state_dict().items()}: raise RuntimeError('Frozen backbone changed')
        if any(p.grad is not None for obj in (old,clip) for p in obj.parameters()): raise RuntimeError('Frozen gradient leakage')
        if digest(a.split_manifest)!=identity['split_sha256']: raise ValueError('Source split changed during run')
        for views in design['caption_views'].values():
            for receipt in views.values():
                if digest(receipt['annotation_path'])!=receipt['annotation_sha256']: raise ValueError('Official caption JSON changed during run')
        atomic_json(root/'frozen_verification.json',dict(passed=True,BART_reference_VAE_CLIP_unchanged=True,no_test_source_used=True))
        status='smoke_complete' if a.smoke else 'complete' if all(v['evaluated'] for v in statuses.values()) else 'partial_complete'
        identity.update(status=status,wall_seconds=time.monotonic()-start); atomic_json(root/'run_identity.json',identity)
        report(root); seal(root); verify(root)
        print('STATUS {} wall={:.1f}s modes={}'.format(status,identity['wall_seconds'],statuses),flush=True)
    except BaseException as exc:
        atomic_json(root/'FAILED.json',dict(version=VERSION,type=type(exc).__name__,message=str(exc),wall_seconds=time.monotonic()-start)); raise


if __name__=='__main__': main()
