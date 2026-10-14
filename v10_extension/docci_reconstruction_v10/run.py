"""One owned GPU process per arm. No auto promotion or old-result rewriting."""
import argparse
import os
from pathlib import Path
import shutil
import sys
import time
EXT=Path(__file__).resolve().parents[1]; PROJECT=EXT.parent
sys.path.insert(0,str(PROJECT)); sys.path.insert(0,str(EXT))
from docci_reconstruction_v10.core import (load,plan,atomic_json,safe_write,read,digest,
    initialization,state_hash,geometry,gate,TEXT_ARMS,IMAGE_ARMS)


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--mode',choices=('smoke','diagnostic','screen','pipeline'),required=True)
    p.add_argument('--arm',choices=TEXT_ARMS+IMAGE_ARMS,required=True); p.add_argument('--output-dir',required=True)
    p.add_argument('--experiment-config'); p.add_argument('--image-size',type=int,choices=(256,512),default=512)
    p.add_argument('--image-width',type=int,choices=(8,16,32),default=8); p.add_argument('--text-width',type=int,choices=(16,32,64),default=64)
    p.add_argument('--max-hours',type=float,default=4)
    for k,v in dict(data_root=PROJECT,bart_path=PROJECT/'pretrained/bart-base',vae_path=PROJECT/'pretrained/sd-vae-ft-mse').items(): p.add_argument('--'+k.replace('_','-'),default=str(v))
    p.add_argument('--init',choices=('history','fresh'),default='history')
    a=p.parse_args(); c=load(a.experiment_config,a.image_size,a.image_width,a.text_width)
    branch='text' if a.arm in TEXT_ARMS else 'image'; schedule=plan(c,a.mode,branch)
    if not 0<a.max_hours<=24 or os.environ.get('CUDA_VISIBLE_DEVICES')!='1': raise ValueError('Finite cap and physical GPU1 required')
    root=safe_write(a.output_dir)
    if root.exists(): raise FileExistsError('Existing result preserved; NEW output required')
    if shutil.disk_usage(PROJECT).free<12*2**30: raise RuntimeError('12GiB project free disk reserve required')
    info=initialization(branch,'fresh' if a.arm=='I2_shifted' else a.init)
    if info['checkpoint'] and ((branch=='text' and c['text_width']!=64) or (branch=='image' and c['image_width']!=8)): raise ValueError('Rate change needs explicit --init fresh')
    root.mkdir(parents=True,exist_ok=False); atomic_json(root/'config.json',c); started=time.monotonic()
    try:
        import numpy as np
        import torch
        from docci_modal_v9.backbones import load as load_backbone,signature,verify_frozen,historical_backbone
        from docci_reconstruction_v10.model import TextBranch,ImageBranch,warm
        from docci_reconstruction_v10.data import source_view,cache,batch
        from docci_reconstruction_v10.workflow import fit
        from docci_reconstruction_v10.evaluate import native,native_chunks,evaluate
        from docci_reconstruction_v10.channel import clean
        from docci_reconstruction_v10.report import finish
        if not torch.cuda.is_available(): raise RuntimeError('Actual GPU required; no CPU replacement')
        device=torch.device('cuda:0'); free,total=torch.cuda.mem_get_info(device)
        print('GPU free={:.2f}GiB total={:.2f}GiB'.format(free/2**30,total/2**30),flush=True)
        if free<c['min_free_gib']*2**30: raise RuntimeError('Insufficient free GPU; no other-process action')
        torch.manual_seed(c['seed']); np.random.seed(c['seed']); torch.set_num_threads(4)
        def budget():
            if time.monotonic()-started>a.max_hours*3600: raise TimeoutError('Owned arm time cap reached')
            if torch.cuda.mem_get_info(device)[0]<c['reserve_free_gib']*2**30: raise RuntimeError('GPU reserve crossed; no silent batch change')
        folder=Path(__file__).parent; manifest=read(folder/'MANIFEST.json')
        if any(digest(PROJECT/n)!=sha for n,sha in manifest['required_sources'].items()): raise ValueError('Historical source changed')
        if any(digest(folder/n)!=sha for n,sha in manifest['files'].items()): raise ValueError('Published V10 source differs')
        snapshot=root/'source_snapshot'; snapshot.mkdir()
        for n in list(manifest['files'])+['MANIFEST.json']: shutil.copyfile(folder/n,snapshot/n)
        backbone,tok,backbone_record=load_backbone(branch,a,device); frozen=signature(backbone)
        historical_backbone(branch,backbone_record['files'])
        model=(TextBranch(backbone,c,a.arm) if branch=='text' else ImageBranch(c,a.arm)).to(device)
        init_record=warm(model,info); initial=model.saved()
        tr,vr,receipt=source_view(a,c,a.mode)
        train,tr_cache=cache(a,c,tr,branch,backbone,tok,backbone_record,device,budget,'train')
        val,va_cache=cache(a,c,vr,branch,backbone,tok,backbone_record,device,budget,'validation')
        diagnostic=train[:schedule['gate_train_samples']]
        atomic_json(root/'data_identity.json',dict(source=receipt,train=tr_cache,validation=va_cache,diagnostic_ids=[i['example_id'] for i in diagnostic]))
        identity=dict(schema='docci-reconstruction-v10',arm=a.arm,branch=branch,mode=a.mode,config=c,schedule=schedule,
            initialization=init_record,backbone=backbone_record,backbone_signature_before=frozen,initial_state_hashes=state_hash(initial),
            geometry=geometry(c,a.arm),gpu_physical=1,source_snapshot_sha256=manifest['files'],Common_used=False,
            diffusion_used=False,opposite_modality_loaded=False,raw_UTF8_transmitted=False,source_mask_or_length_at_RX=False,
            checkpoint_policy='fixed last; no best-validation selection; no smoke reuse',stages=[],evaluations=[])
        def ev(items,stage,cohort,name,physical=False,preview=True):
            rows=evaluate(model,items,c,backbone,tok,device,budget,root/name,stage,cohort,physical,preview)
            identity['evaluations'].append(name); return rows
        native(model,diagnostic,backbone,tok,device,budget,root/'native_train','diagnostic_train')
        native(model,val,backbone,tok,device,budget,root/'native_validation','validation')
        if a.arm=='T1_chunks': native_chunks(model,val,backbone,tok,device,budget,root/'native_chunk_validation')
        ev(val,'initial','validation','initial_validation')
        fit(model,train,c,backbone,device,root/'train_clean','clean',schedule['updates'],budget)
        identity['stages'].append('train_clean')
        train_rows=ev(diagnostic,'clean','diagnostic_train','clean_train'); val_rows=ev(val,'clean','validation','clean_validation')
        identity['train_gate']=gate(train_rows,branch,c,True); identity['validation_gate']=gate(val_rows,branch,c,False)
        quality=identity['train_gate']['passed'] and identity['validation_gate']['passed']
        if a.mode=='smoke': identity['status']='SMOKE_COMPLETE'
        elif not quality: identity['status']='QUALITY_STOP'
        elif a.mode!='pipeline': identity['status']='CLEAN_READY'
        else:
            clean_state=model.saved(); key='memory' if branch=='text' else 'latent'
            with torch.no_grad():
                for i,item in enumerate(train):
                    budget(); wire=model.encode(batch(train,[i],branch,device)); item['clean_teacher']=model.receive(*clean(wire))[key][0].cpu().numpy()
            states={'clean':clean_state}; identity['teacher_scope']='own clean TRAIN receiver feature; no validation teacher'
            for stage in ('clean_continue','fso_continue'):
                model.restore(clean_state)
                states[stage]=fit(model,train,c,backbone,device,root/('train_'+stage),stage,c[branch+'_continue_updates'],budget)
                identity['stages'].append('train_'+stage)
            for stage,state in states.items():
                model.restore(state); ev(val,stage,'validation','physical_'+stage,physical=True)
            identity['status']='PIPELINE_COMPLETE'
        verify_frozen((backbone,),(frozen,)); identity['backbone_signature_after']=signature(backbone)
        if any(digest(PROJECT/n)!=sha for n,sha in manifest['required_sources'].items()): raise RuntimeError('Historical source mutated during run')
        if any(digest(folder/n)!=sha for n,sha in manifest['files'].items()): raise RuntimeError('Published source changed during run')
        if info['checkpoint'] and digest(info['checkpoint'])!=info['sha256']: raise RuntimeError('Own historical checkpoint changed')
        if digest(Path(a.data_root)/'docci_descriptions.jsonlines')!=receipt['annotation_sha256']: raise RuntimeError('Source annotation changed')
        identity.update(wall_seconds=time.monotonic()-started,actual_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
            final_state_hashes=state_hash(model.saved()))
        atomic_json(root/'run_identity.json',identity); finish(root)
        print('{} {} train_gate={} val_gate={} wall={:.1f}s'.format(a.arm,identity['status'],identity['train_gate']['passed'],identity['validation_gate']['passed'],identity['wall_seconds']),flush=True)
    except BaseException as exc:
        atomic_json(root/'FAILED.json',dict(exception=type(exc).__name__,message=str(exc),arm=a.arm,wall_seconds=time.monotonic()-started,not_a_quality_stop=True))
        raise


if __name__=='__main__': main()
