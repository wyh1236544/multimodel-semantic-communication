"""Owned GPU1 process; approved weights-only warm start, pure CE to cumulative20epochs."""
import argparse
import os
from pathlib import Path
import shutil
import sys
import time
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT)); sys.path.insert(0,str(EXT.parent))
from docci_text_continue_v11.core import (FOLDER,PROJECT,load,parent,check_sources,schedule,read,digest,
    safe_write,atomic_json,state_hash,geometry,gate,jsonlines)


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--output',required=True); p.add_argument('--parent-run',required=True)
    p.add_argument('--phase-config'); p.add_argument('--data-root',default=str(PROJECT))
    p.add_argument('--bart-path',default=str(PROJECT/'pretrained/bart-base'))
    p.add_argument('--max-hours',type=float,default=3); p.add_argument('--smoke-only',action='store_true')
    a=p.parse_args(); phase=load(a.phase_config); approved=parent(a.parent_run,True); oldc=approved['config']
    planned=schedule(phase); root=safe_write(a.output)
    if root.exists(): raise FileExistsError('Existing output preserved; use a NEW attempt')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1' or not 0<a.max_hours<=24: raise ValueError('PhysicalGPU1 and finite cap required')
    if shutil.disk_usage(PROJECT).free<12*2**30: raise RuntimeError('Need12GiB free project disk')
    manifest=check_sources(); root.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    atomic_json(root/'phase_config.json',phase); atomic_json(root/'config.json',oldc)
    snapshot=root/'source_snapshot'; snapshot.mkdir()
    for name in list(manifest['files'])+['MANIFEST.json']: shutil.copyfile(FOLDER/name,snapshot/name)
    # Exact immutable approved evidence, not a substitute new experiment or imported code.
    reference=root/'parent_reference'; reference.mkdir()
    for name in read(FOLDER/'parent_contract.json')['files']:
        dest=reference/name; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(approved['root']/name,dest)
    try:
        import numpy as np
        import torch
        from docci_modal_v9.backbones import load as backbone_load,signature,verify_frozen,historical_backbone
        from docci_reconstruction_v10.model import TextBranch
        from docci_reconstruction_v10.data import source_view,cache
        from docci_text_continue_v11.workflow import fit
        from docci_text_continue_v11.evaluate import evaluate
        from docci_text_continue_v11.diagnostic import diagnose
        if not torch.cuda.is_available(): raise RuntimeError('Actual GPU required, no CPU fallback')
        device=torch.device('cuda:0'); free,total=torch.cuda.mem_get_info(device)
        print('GPU1 free={:.2f}GiB total={:.2f}GiB'.format(free/2**30,total/2**30),flush=True)
        if free<oldc['min_free_gib']*2**30: raise RuntimeError('Need16GiB free; do not stop other processes')
        def budget():
            if time.monotonic()-started>a.max_hours*3600: raise TimeoutError('Owned run cap reached')
            if torch.cuda.mem_get_info(device)[0]<oldc['reserve_free_gib']*2**30: raise RuntimeError('GPU reserve crossed')
        def progress(stage):
            atomic_json(root/'progress.json',dict(stage=stage,elapsed_seconds=time.monotonic()-started)); print('STAGE '+stage,flush=True)
        torch.manual_seed(oldc['seed']); np.random.seed(oldc['seed']); torch.set_num_threads(4)
        progress('loading approved offline BART and actual parent checkpoint')
        backbone,tok,br=backbone_load('text',a,device); frozen=signature(backbone); historical_backbone('text',br['files'])
        model=TextBranch(backbone,oldc,'T1_chunks').to(device)
        record=torch.load(str(approved['checkpoint']),map_location='cpu')
        if record['version']!='docci-reconstruction-v10' or record['arm']!='T1_chunks' or record['updates']!=10000 or record['config']!=oldc:
            raise ValueError('Actual parent checkpoint metadata differs')
        model.restore(record['state']); parent_state=model.saved()
        if state_hash(parent_state)!=approved['identity']['final_state_hashes']: raise ValueError('Loaded parent tensors differ')
        del record
        tr,vr,receipt=source_view(a,oldc,'screen')
        progress('TRAIN2048/qual_dev100 cached source identity check (no test)')
        train,tc=cache(a,oldc,tr,'text',backbone,tok,br,device,budget,'train')
        val,vc=cache(a,oldc,vr,'text',backbone,tok,br,device,budget,'validation')
        expected=approved['data']
        if receipt!=expected['source'] or tc['records']!=expected['train']['records'] or vc['records']!=expected['validation']['records']:
            raise ValueError('Source/cache identity changed from approved parent')
        atomic_json(root/'data_identity.json',dict(source=receipt,train=tc,validation=vc,diagnostic_ids=expected['diagnostic_ids']))
        # Genuine2-step training/serialization smoke; restore the UNCHANGED parent afterwards.
        progress('authentic saved-parent GPU smoke; its weights will be discarded')
        smoke=root/'smoke'; smoke.mkdir()
        fit(model,train[:8],oldc,phase,planned,device,smoke/'train',budget,lambda u:None,True)
        smoke_rows=evaluate(model,val[:2],oldc,tok,device,budget,smoke/'evaluation','smoke','validation')
        model.restore(parent_state); torch.manual_seed(oldc['seed']); np.random.seed(oldc['seed'])
        if state_hash(model.saved())!=state_hash(parent_state): raise ValueError('Smoke contaminated main initialization')
        atomic_json(smoke/'SMOKE_COMPLETE.json',dict(actual_parent_loaded=True,actual_backward_and_disk_reload=True,
            saved_smoke_weights_NEVER_reused=True,parent_restored_state_hashes=state_hash(model.saved()),rows=len(smoke_rows)))
        if a.smoke_only:
            atomic_json(root/'SMOKE_ONLY_COMPLETE.json',dict(schema='docci-text-continue-v11',no_main_training=True,
                parent_checkpoint_sha256=approved['checkpoint_sha256'])); print('SMOKE_ONLY_COMPLETE',flush=True); return
        probes=[]; evaluations=[]; diagnostics=[]
        def probe(global_update,final=False):
            stage='g'+str(global_update); progress('free generation and privileged prefix diagnosis '+stage)
            ti=train[:32] if final else train[:phase['probe_samples']]
            vi=val if final else val[:phase['probe_samples']]
            for cohort,items,tag in (('diagnostic_train',ti,'train'),('validation',vi,'validation')):
                name=stage+'_'+tag
                rows=evaluate(model,items,oldc,tok,device,budget,root/name,stage,cohort)
                evaluations.append(dict(folder=name,stage=stage,cohort=cohort,example_ids=[x['example_id'] for x in items],global_update=global_update))
                if global_update==10000:
                    original={r['example_id']:r for r in jsonlines(
                        approved['root']/('clean_train' if tag=='train' else 'clean_validation')/'rows.jsonl') if r['control']=='matched'}
                    for r in rows:
                        if r['control']=='matched' and (r['generated_token_ids']!=original[r['example_id']]['generated_token_ids'] or
                                r['tx_wire_sha256']!=original[r['example_id']]['tx_wire_sha256']):
                            raise ValueError('Actual loaded-parent baseline differs from approved generation')
                if tag=='validation':
                    dname=stage+'_prefix_diagnostic'; selected=val[:phase['probe_samples']]
                    diagnose(model,selected,oldc,tok,device,budget,root/dname,rows)
                    diagnostics.append(dict(folder=dname,global_update=global_update,example_ids=[x['example_id'] for x in selected]))
            probes.append(global_update)
        probe(10000)
        progress('pure CE new optimizer phase; global cursor starts at10000')
        fit(model,train,oldc,phase,planned,device,root/'train_ce',budget,probe)
        probe(planned['target_updates'],True)
        progress('read-only historical sources, backbone and checkpoint final audit')
        verify_frozen((backbone,),(frozen,)); check_sources()
        if digest(approved['checkpoint'])!=approved['checkpoint_sha256']: raise RuntimeError('Parent weight changed during continuation')
        if digest(Path(a.data_root)/'docci_descriptions.jsonlines')!=receipt['annotation_sha256']: raise RuntimeError('Original annotation changed')
        # No automatic FSO or best-checkpoint selection, regardless of quality.
        identity=dict(schema='docci-text-continue-v11',status='TRAIN_AND_EVAL_FINISHED',schedule=planned,phase=phase,
            config=oldc,gpu_physical=1,smoke_weights_reused=False,optimizer_restored=False,
            initial_state_hashes=state_hash(parent_state),final_state_hashes=state_hash(model.saved()),
            parent_checkpoint_sha256=approved['checkpoint_sha256'],backbone_signature_before=frozen,backbone_signature_after=signature(backbone),
            geometry=geometry(oldc,'T1_chunks'),evaluations=evaluations,diagnostics=diagnostics,probes=probes,
            gold_prefix_diagnostics_NOT_actual_recovery=True,source_at_actual_RX=False,FSO_training=False,
            CLIP_Common_image_diffusion_used=False,loss='pure CE',checkpoint_policy='fixed final, no validation selection',
            active_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),wall_seconds=time.monotonic()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
        atomic_json(root/'run_identity.json',identity)
        atomic_json(root/'RUN_FINISHED.json',dict(run_identity_sha256=digest(root/'run_identity.json'),independent_CPU_verification_PENDING=True))
        print('TRAIN_AND_EVAL_FINISHED; independent collection/verification still pending',flush=True)
    except BaseException as exc:
        atomic_json(root/'FAILED.json',dict(exception=type(exc).__name__,message=str(exc),elapsed_seconds=time.monotonic()-started))
        raise


if __name__=='__main__': main()
