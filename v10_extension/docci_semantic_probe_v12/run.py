"""Actual offline GPU experiments. Main never reads smoke weights or test inputs."""
import argparse
import gc
import os
from pathlib import Path
import shutil
import sys
import time
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT)); sys.path.insert(0,str(EXT.parent))
from docci_semantic_probe_v12.core import FOLDER, PROJECT, SCHEMA, load, read, safe, parent, check_sources, atomic_json, digest


def text_task(args, c, root, info, device, budget):
    import numpy as np
    import torch
    from docci_modal_v9.backbones import load as backbone_load,signature,verify_frozen,historical_backbone
    from docci_reconstruction_v10.data import source_view,cache
    from docci_reconstruction_v10.model import TextBranch
    from docci_reconstruction_v10.core import state_hash
    from docci_text_continue_v11.evaluate import evaluate
    from docci_text_continue_v11.diagnostic import diagnose
    from docci_semantic_probe_v12.train import fit
    root.mkdir(exist_ok=False); oldc=dict(info['config'])
    oldc.update(train_samples=c['train_samples'],val_samples=c['val_samples'],text_batch_size=c['text_batch_size'])
    mode='smoke' if args.mode=='smoke' else 'screen'
    backbone,tok,br=backbone_load('text',args,device); historical_backbone('text',br['files']); frozen=signature(backbone)
    model=TextBranch(backbone,oldc,'T1_chunks').to(device)
    saved=torch.load(str(info['checkpoint']),map_location='cpu')
    if saved['schema']!='docci-text-continue-v11-checkpoint' or saved['global_update']!=20480 or saved['config']!=info['config']:
        raise ValueError('Actual approved V11 checkpoint metadata differs')
    model.restore(saved['state']); del saved
    initial=model.saved()
    if state_hash(initial)!=info['identity']['final_state_hashes']: raise ValueError('Loaded V11 final tensors differ')
    tr,vr,source=source_view(args,oldc,mode)
    if source!=info['data']['source']: raise ValueError('Official source identity changed from approved V11')
    train,tc=cache(args,oldc,tr,'text',backbone,tok,br,device,budget,'TRAIN')
    val,vc=cache(args,oldc,vr,'text',backbone,tok,br,device,budget,'qual_dev')
    atomic_json(root/'data_identity.json',dict(source=source,train=tc,validation=vc))
    arms={}
    for arm in ('B0_ce','B1_ce_kd'):
        budget(); model.restore(initial); torch.manual_seed(c['seed']); np.random.seed(c['seed'])
        folder=root/arm; folder.mkdir(); print('STAGE '+arm+' clean matched training',flush=True)
        fit(model,train,c,'text',arm,device,folder/'train',budget,smoke=args.mode=='smoke')
        selected=train[:2 if args.mode=='smoke' else c['train_eval_samples']]
        evaluate(model,selected,oldc,tok,device,budget,folder/'train_evaluation',arm,'diagnostic_train')
        rows=evaluate(model,val,oldc,tok,device,budget,folder/'validation',arm,'validation')
        diagnose(model,val[:min(c['diagnostic_samples'],len(val))],oldc,tok,device,budget,folder/'prefix_diagnostic',rows)
        arms[arm]=dict(initial_state_hashes=state_hash(initial),final_state_hashes=state_hash(model.saved()),
            active_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            parent_optimizer_restored=False,backbone_frozen=True,teacher_weight=c['kd_weight'] if arm=='B1_ce_kd' else 0.)
    verify_frozen((backbone,),(frozen,))
    if digest(Path(args.data_root)/'docci_descriptions.jsonlines')!=source['annotation_sha256']: raise ValueError('Source annotation changed during run')
    atomic_json(root/'identity.json',dict(task='text',config=oldc,arms=arms,parent_checkpoint_sha256=info['spec']['checkpoint_sha256'],
        source_signature_before=frozen,source_signature_after=signature(backbone),
        text_complex_symbols=9216,analog_repeats=1,FSO=False,Common_CLIP_image_diffusion=False,
        teacher_TRAIN_loss_only=True,actual_RX_source_text_memory_mask_or_length=False,
        controls=['matched','shuffled','zero'],checkpoint_policy='fixed final; no validation-best selection'))


def image_task(args, c, root, info, device, budget):
    import numpy as np
    import torch
    from docci_modal_v9.backbones import load as backbone_load,signature,verify_frozen,historical_backbone
    from docci_reconstruction_v10.data import source_view,cache
    from docci_reconstruction_v10.model import ImageBranch
    from docci_reconstruction_v10.core import state_hash
    from docci_semantic_probe_v12.data import load_longclip,semantic_cache,image_wires
    from docci_semantic_probe_v12.model import ImageProbe
    from docci_semantic_probe_v12.train import fit
    from docci_semantic_probe_v12.evaluate import frames,images
    root.mkdir(exist_ok=False); oldc=dict(info['config']); oldc.update(train_samples=c['train_samples'],val_samples=c['val_samples'])
    vae,_,br=backbone_load('image',args,device); historical_backbone('image',br['files']); frozen_vae=signature(vae)
    base=ImageBranch(oldc,'I2_shifted').to(device)
    saved=torch.load(str(info['checkpoint']),map_location='cpu')
    if saved['version']!='docci-reconstruction-v10' or saved['arm']!='I2_shifted' or saved['updates']!=512 or saved['config']!=info['config']:
        raise ValueError('Actual approved I2 checkpoint metadata differs')
    base.restore(saved['state']); del saved
    if state_hash(base.saved())!=info['identity']['final_state_hashes']: raise ValueError('Loaded I2 tensors differ')
    base.requires_grad_(False).eval(); frozen_base=signature(base)
    tr,vr,source=source_view(args,oldc,'smoke' if args.mode=='smoke' else 'screen')
    if source!=info['data']['source']: raise ValueError('Official source identity changed from approved I2')
    train,tc=cache(args,oldc,tr,'image',vae,None,br,device,budget,'TRAIN')
    val,vc=cache(args,oldc,vr,'image',vae,None,br,device,budget,'qual_dev')
    clip,tok,cr=load_longclip(args,device); frozen_clip=signature(clip)
    train_sem,ts=semantic_cache(tr,clip,tok,cr,device,budget,'TRAIN')
    val_sem,vs=semantic_cache(vr,clip,tok,cr,device,budget,'qual_dev')
    verify_frozen((clip,),(frozen_clip,)); clip_after=signature(clip)
    del clip,tok; gc.collect(); torch.cuda.empty_cache()
    for records,items,semantics in ((tr,train,train_sem),(vr,val,val_sem)):
        if [r['example_id'] for r in records]!=[i['example_id'] for i in items]: raise ValueError('Image/text pairing differs')
        for r,item in zip(records,items):
            item.update(text=r['text'],text_sha256=r['text_sha256'],semantic=semantics[r['example_id']])
    tw=image_wires(train,base,device,budget,'TRAIN'); vw=image_wires(val,base,device,budget,'qual_dev')
    atomic_json(root/'data_identity.json',dict(source=source,train=tc,validation=vc,train_semantics=ts,validation_semantics=vs,
        train_wires=tw,validation_wires=vw,longclip=cr,longclip_signature_before=frozen_clip,longclip_signature_after=clip_after))
    received=frames(val,c,root/'frames')
    images(None,base,val,vae,c,device,budget,root,received,'P0_parent')
    torch.manual_seed(c['seed']); probe=ImageProbe(base,'A0_image_only').to(device); initial=probe.saved(); arms={}
    for arm in ('A0_image_only','A1_oracle_text'):
        budget(); probe.arm=arm; probe.restore(initial); torch.manual_seed(c['seed']); np.random.seed(c['seed'])
        folder=root/arm; folder.mkdir(); print('STAGE '+arm+' receiver-only matched training',flush=True)
        fit(probe,train,c,'image',arm,device,folder/'train',budget,backbone=vae,smoke=args.mode=='smoke')
        # images() owns the evaluation directory; keep training beside it, not inside a reused path.
        images(probe,base,val,vae,c,device,budget,root,received,arm+'_evaluation')
        arms[arm]=dict(initial_state_hashes=state_hash(initial),final_state_hashes=state_hash(probe.saved()),
            active_parameters=sum(p.numel() for p in probe.parameters() if p.requires_grad),backbone_frozen=True,
            effective_context='image-derived context only' if arm=='A0_image_only' else 'same image-derived context + ORACLE text vector')
    verify_frozen((base,vae),(frozen_base,frozen_vae))
    if digest(Path(args.data_root)/'docci_descriptions.jsonlines')!=source['annotation_sha256']: raise ValueError('Source annotation changed during run')
    atomic_json(root/'identity.json',dict(task='image',config=oldc,arms=arms,parent_checkpoint_sha256=info['spec']['checkpoint_sha256'],
        frozen_image_codec_before=frozen_base,frozen_image_codec_after=signature(base),frozen_VAE_before=frozen_vae,frozen_VAE_after=signature(vae),
        image_complex_symbols=4096,analog_repeats=1,text_overhead_unmeasured=True,FSO=False,Common=False,diffusion=False,
        ORACLE_text_is_privileged_NOT_actual_communication=True,LPIPS=None,LPIPS_status='NOT_MEASURED',
        primary_metric='PSNR',primary_condition=c['image_primary_condition'],
        scope='SSA-inspired receiver probe, NOT a reproduction of HiFiC/GAN or proof of equal-budget joint transmission'))


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--mode',choices=('smoke','train'),required=True); p.add_argument('--task',choices=('text','image','both'),required=True)
    p.add_argument('--output',required=True); p.add_argument('--config'); p.add_argument('--text-parent'); p.add_argument('--image-parent')
    p.add_argument('--max-hours',type=float,default=4)
    for name,path in dict(data_root=PROJECT,bart_path=PROJECT/'pretrained/bart-base',vae_path=PROJECT/'pretrained/sd-vae-ft-mse',
        longclip_path=PROJECT/'longclip-B.pt',clip_tokenizer_path=PROJECT/'pretrained/clip-vit-base-patch32').items():
        p.add_argument('--'+name.replace('_','-'),default=str(path))
    args=p.parse_args(); c=load(args.config); root=safe(args.output)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1' or not 0<args.max_hours<=24: raise ValueError('Physical GPU1 and finite cap required')
    if root.exists(): raise FileExistsError('Existing output preserved; use NEW attempt')
    if shutil.disk_usage(PROJECT).free<12*2**30: raise RuntimeError('Need12GiB free project disk')
    manifest=check_sources(); tasks=['text','image'] if args.task=='both' else [args.task]
    approved={task:parent(task,getattr(args,task+'_parent')) for task in tasks}
    root.mkdir(parents=True,exist_ok=False); started=time.monotonic(); atomic_json(root/'config.json',c)
    snapshot=root/'source_snapshot'; snapshot.mkdir()
    for name in list(manifest['files'])+['MANIFEST.json']: shutil.copyfile(FOLDER/name,snapshot/name)
    try:
        import torch
        import numpy as np
        if not torch.cuda.is_available(): raise RuntimeError('Authentic GPU required; no CPU substitute')
        device=torch.device('cuda:0'); free,total=torch.cuda.mem_get_info(device)
        print('GPU1 free={:.2f}GiB total={:.2f}GiB'.format(free/2**30,total/2**30),flush=True)
        if free<c['min_free_gib']*2**30: raise RuntimeError('Need16GiB free; never stop other users')
        torch.set_num_threads(4)
        def budget():
            if time.monotonic()-started>args.max_hours*3600: raise TimeoutError('Owned experiment budget exhausted')
            if torch.cuda.mem_get_info(device)[0]<c['reserve_free_gib']*2**30: raise RuntimeError('Free GPU reserve crossed; no automatic batch changes')
        for task in tasks:
            torch.manual_seed(c['seed']); np.random.seed(c['seed'])
            atomic_json(root/'progress.json',dict(stage=task,mode=args.mode,elapsed_seconds=time.monotonic()-started))
            (text_task if task=='text' else image_task)(args,c,root/task,approved[task],device,budget)
            reference=root/task/'parent_reference'; reference.mkdir()
            for name in approved[task]['spec']['files']:
                target=reference/name; target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(approved[task]['root']/name,target)
            gc.collect(); torch.cuda.empty_cache()
            if digest(approved[task]['checkpoint'])!=approved[task]['spec']['checkpoint_sha256']: raise ValueError('Historical checkpoint changed')
        check_sources()
        identity=dict(schema=SCHEMA,mode=args.mode,tasks=tasks,config=c,gpu_physical=1,smoke_weights_reused=False,
            FSO=False,no_automatic_promotion=True,source_snapshot_hashes=manifest['files'],
            wall_seconds=time.monotonic()-started,last_training_phase_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
            last_training_phase_peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
            memory_scope='reset per fit; history retains each phase peak; NOT whole-launch peak')
        atomic_json(root/'run_identity.json',identity)
        atomic_json(root/'RUN_FINISHED.json',dict(schema=SCHEMA,run_identity_sha256=digest(root/'run_identity.json'),independent_verification_pending=True))
        print('GPU_PROGRAM_FINISHED; independent collection still required. No performance claim.',flush=True)
    except BaseException as exc:
        atomic_json(root/'FAILED.json',dict(exception=type(exc).__name__,message=str(exc),elapsed_seconds=time.monotonic()-started)); raise


if __name__=='__main__': main()
