"""One owned process, one modality, fixed checkpoints, independent quality stop."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time
EXT=Path(__file__).resolve().parents[1]; PROJECT=EXT.parent
sys.path.insert(0,str(PROJECT)); sys.path.insert(0,str(EXT))
from docci_modal_v9.core import atomic_json,digest,load,safe_write,schedule,initialization,gate,state_hash,geometry,read


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--mode',choices=('smoke','diagnostic','screen','pipeline'),required=True)
    p.add_argument('--branch',choices=('text','image'),required=True); p.add_argument('--output-dir',required=True)
    p.add_argument('--experiment-config'); p.add_argument('--image-size',type=int,choices=(256,512),default=512)
    p.add_argument('--max-text-tokens',type=int,choices=(512,1024),default=512); p.add_argument('--max-hours',type=float,default=4)
    for key,value in dict(data_root=PROJECT,bart_path=PROJECT/'pretrained/bart-base',vae_path=PROJECT/'pretrained/sd-vae-ft-mse').items(): p.add_argument('--'+key.replace('_','-'),default=str(value))
    for branch in ('text','image'):
        p.add_argument('--'+branch+'-init',default='history'); p.add_argument('--'+branch+'-init-sha256')
    a=p.parse_args(); c=load(a.experiment_config,a.image_size,a.max_text_tokens); plan=schedule(c,a.mode,a.branch)
    if not 0<a.max_hours<=24: raise ValueError('Finite time cap0..24h')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1': raise ValueError('Use physical GPU1 only; logical cuda:0')
    root=safe_write(a.output_dir)
    if root.exists(): raise FileExistsError('Existing output preserved')
    if shutil.disk_usage(PROJECT).free<8*2**30: raise RuntimeError('At least8GiB project disk reserve required')
    info=initialization(a.branch,getattr(a,a.branch+'_init'),getattr(a,a.branch+'_init_sha256'))
    root.mkdir(parents=True,exist_ok=False); atomic_json(root/'config.json',c)
    started=time.monotonic()
    import numpy as np
    import torch
    from docci_modal_v9.backbones import load as load_backbone,signature,verify_frozen,historical_backbone
    from docci_modal_v9.model import TextBranch,ImageBranch,warm
    from docci_modal_v9.data import source_view,cache,batch
    from docci_modal_v9.workflow import fit
    from docci_modal_v9.evaluate import native,evaluate
    from docci_modal_v9.channel import clean
    from docci_modal_v9.report import finish
    if not torch.cuda.is_available(): raise RuntimeError('Authentic GPU smoke required; no CPU fallback')
    device=torch.device('cuda:0'); free,total=torch.cuda.mem_get_info(device)
    print('GPU free={:.2f}GiB total={:.2f}GiB'.format(free/2**30,total/2**30),flush=True)
    if free<c['min_free_gib']*2**30: raise RuntimeError('Insufficient GPU free reserve; do not touch unrelated processes')
    torch.manual_seed(c['seed']); np.random.seed(c['seed']); torch.set_num_threads(4)
    def budget():
        if time.monotonic()-started>a.max_hours*3600: raise TimeoutError('Owned branch cap reached; preserve evidence')
        if torch.cuda.mem_get_info(device)[0]<c['reserve_free_gib']*2**30: raise RuntimeError('GPU free reserve crossed; no silent batch reduction')
    manifest=read(Path(__file__).with_name('MANIFEST.json'))
    if any(digest(PROJECT/n)!=sha for n,sha in manifest['required_sources'].items()): raise ValueError('Historical prerequisite changed before execution')
    snapshots={n:digest(Path(__file__).parent/n) for n in manifest['files']}
    if snapshots!=manifest['files']: raise ValueError('Published source differs before execution')
    source_folder=root/'source_snapshot'; source_folder.mkdir()
    for n in list(manifest['files'])+['MANIFEST.json']: shutil.copyfile(Path(__file__).parent/n,source_folder/n)
    backbone,tok,backbone_record=load_backbone(a.branch,a,device); frozen=signature(backbone)
    if info['checkpoint']: historical_backbone(a.branch,backbone_record['files'])
    model=TextBranch(backbone,c) if a.branch=='text' else ImageBranch(c)
    model=model.to(device); init_record=warm(model,info); initial=model.saved()
    train_records,val_records,receipt=source_view(a,c,a.mode)
    # Smoke has genuine real backbones/shape/optimizer/checkpoint, not quality claims.
    train,train_cache=cache(a,c,train_records,a.branch,backbone,tok,backbone_record,device,budget,'train')
    val,val_cache=cache(a,c,val_records,a.branch,backbone,tok,backbone_record,device,budget,'validation')
    diag=train[:plan['diagnostic_samples']]
    atomic_json(root/'data_identity.json',dict(source=receipt,train=train_cache,validation=val_cache,diagnostic_ids=[i['example_id'] for i in diag]))
    identity=dict(schema='docci-modal-v9',mode=a.mode,branch=a.branch,config=c,schedule=plan,initialization=init_record,
        backbone=backbone_record,backbone_signature_before=frozen,initial_state_hashes=state_hash(initial),geometry=geometry(c,a.branch),
        gpu_physical=1,logical_device='cuda:0',source_snapshot_sha256=snapshots,opposite_modality_model_loaded=False,
        Common_used=False,raw_UTF8_transmitted=False,diffusion_used=False,quality_selection='fixed last; no best validation checkpoint',stages=[],evaluations=[])
    def ev(items,stage,cohort,name,physical=False,save_images=True):
        rows=evaluate(model,items,c,backbone,tok,device,budget,root/name,stage,cohort,physical,save_images)
        identity['evaluations'].append(name); return rows
    native(model,diag,backbone,tok,device,budget,root/'native_train','diagnostic_train')
    native(model,val,backbone,tok,device,budget,root/'native_validation','validation')
    ev(val,'initial','validation','initial_validation')
    def snapshot(update):
        ev(diag,'overfit','diagnostic_train','snapshot_{:05d}'.format(update),save_images=False)
    overfit=fit(model,diag,c,backbone,device,root/'train_overfit','overfit',plan['diagnostic_updates'],budget,snapshot)
    identity['stages'].append('train_overfit')
    rows=ev(diag,'overfit','diagnostic_train','overfit_train'); quality=gate(rows,a.branch,c,True)
    ev(val,'overfit_DIAGNOSTIC_WEIGHTS','validation','overfit_validation')
    identity['overfit_gate']=quality; states={}
    if a.mode=='smoke': identity['status']='SMOKE_COMPLETE'; identity['smoke_quality_not_required']=True
    elif a.mode=='diagnostic': identity['status']='DIAGNOSTIC_COMPLETE' if quality['passed'] else 'QUALITY_STOP'
    elif not quality['passed']: identity['status']='QUALITY_STOP'
    else:
        model.restore(initial)
        if state_hash(model.saved())!=identity['initial_state_hashes']: raise RuntimeError('Production reset inherited fitted diagnostic weights')
        identity['clean_reset_from_history_not_overfit']=True
        states['clean']=fit(model,train,c,backbone,device,root/'train_clean','clean',plan['clean_updates'],budget)
        identity['stages'].append('train_clean')
        clean_rows=ev(val,'clean','validation','clean_validation'); clean_gate=gate(clean_rows,a.branch,c,False)
        identity['clean_gate']=clean_gate
        if not clean_gate['passed']: identity['status']='QUALITY_STOP'
        elif a.mode=='screen': identity['status']='SCREEN_COMPLETE'
        else:
            key='memory' if a.branch=='text' else 'latent'
            with torch.no_grad():
                for i,item in enumerate(train):
                    budget(); wire=model.encode(batch(train,[i],a.branch,device))
                    item['clean_teacher']=model.receive(*clean(wire))[key][0].cpu().numpy()
            identity['teacher_scope']='TRAIN only; own clean branch received memory/latent; no validation teacher'
            for stage in ('clean_continue','fso_continue'):
                model.restore(states['clean'])
                states[stage]=fit(model,train,c,backbone,device,root/('train_'+stage),stage,plan['continue_updates'],budget)
                identity['stages'].append('train_'+stage)
            for stage,state in states.items():
                model.restore(state); ev(val,stage,'validation','physical_'+stage,physical=True)
            identity['status']='COMPLETE'
    verify_frozen((backbone,),(frozen,)); identity['backbone_signature_after']=signature(backbone)
    if snapshots!={n:digest(Path(__file__).parent/n) for n in snapshots}: raise RuntimeError('Source changed during execution')
    if any(digest(PROJECT/n)!=sha for n,sha in manifest['required_sources'].items()): raise RuntimeError('Historical prerequisite changed during execution')
    if info['checkpoint'] and digest(info['checkpoint'])!=info['sha256']: raise RuntimeError('Historical initializer changed')
    if digest(Path(a.data_root)/'docci_descriptions.jsonlines')!=receipt['annotation_sha256']: raise RuntimeError('DOCCI annotation changed during execution')
    identity.update(wall_seconds=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,actual_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        final_state_hashes=state_hash(model.saved()))
    atomic_json(root/'run_identity.json',identity); finish(root)
    print('{} {} overfit_gate={} wall={:.1f}s'.format(a.branch,identity['status'],quality['passed'],identity['wall_seconds']),flush=True)


if __name__=='__main__': main()
