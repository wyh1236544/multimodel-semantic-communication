"""GPU1 diagnostic -> clean text-only training -> matched clean/FSO continuation."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from text_jscc_v1_fix1 import VERSION,ARMS
from text_jscc_v1_fix1.core import load,TEMPLATE,digest,text_records,overfit_gate,seed_for


def main():
    import torch
    from v10.runtime import add_shared_args,preflight
    from v10.io import atomic_json,atomic_checkpoint,begin_run
    from joint_jscc_arch_v1.model import encode_text
    from text_jscc_v1_fix1.legacy import load_reference
    from text_jscc_v1_fix1.model import TextJSCC,tensors,generate,legacy_wire,legacy_receive
    from text_jscc_v1_fix1.workflow import (state,signature,row,native_memories,memory_tests,
        decoder_diagnostics,train,SemanticScorer)
    from text_jscc_v1_fix1.channel import calibrate,check_current,wave,short_resource
    from fso_fluctuating_v2.core import paths
    from fso_jscc_v1.core import kernels
    from text_jscc_v1_fix1.report import report,verify
    from optical_realism_v3.constellation import constellation_svg
    p=add_shared_args(argparse.ArgumentParser(description=__doc__,allow_abbrev=False))
    p.add_argument('--source-run',required=True); p.add_argument('--legacy-run',required=True); p.add_argument('--a-delta',required=True)
    p.add_argument('--experiment-config',default=str(TEMPLATE)); p.add_argument('--smoke',action='store_true')
    a=p.parse_args(); cfg=load(a.experiment_config)
    if a.smoke: cfg.update(train_samples=8,val_samples=2,diagnostic_samples=2,diagnostic_updates=8,clean_epochs=1,continue_epochs=1,batch_size=2,bank_train_frames=8,bank_holdout_frames=4)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1': raise ValueError('Use launch.py, physical GPU1')
    if not math.isfinite(a.max_hours) or not 0<a.max_hours<=24: raise ValueError('Invalid stop budget')
    a.batch_size,a.num_workers,a.no_amp,a.blocks,a.fixed_mode,a.csi=1,0,True,1,0,'pilot'
    a.min_free_gb,a.min_free_disk_gb=cfg['min_free_gib'],12.
    device=preflight(a)
    if device.type!='cuda': raise ValueError('Real saved-checkpoint experiment requires GPU')
    old,core,cc,hashes=load_reference(a,device)
    if a.smoke: cc=dict(cc,conditions=cc['conditions'][:2],primary_conditions=[q['id'] for q in cc['conditions'][:2]])
    folder,identity=begin_run(a.output_dir,a,cfg,EXT); start=time.monotonic()
    identity.update(version=VERSION,**hashes,new_source_sha256={f.name:digest(f) for f in Path(__file__).parent.iterdir() if f.suffix in ('.py','.json','.md')})
    atomic_json(folder/'run_identity.json',identity)
    tok=old.base.text_encoder.tokenizer
    frozen_old={n:signature(v) for n,v in old.state_dict().items()}; frozen_core={n:signature(v) for n,v in core.state_dict().items()}
    def budget():
        if time.monotonic()-start>a.max_hours*3600: raise TimeoutError('Overall text screen cap reached; preserve outputs')
        free,_=torch.cuda.mem_get_info(device)
        if free/2**30<cfg['reserve_free_gib']: raise MemoryError('GPU reserve breached; no unrelated process action')
    def record_lines(path,rows):
        with path.open('x',encoding='utf-8') as f:
            for r in rows: f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    design=None
    def finish(status):
        if any(signature(v)!=frozen_old[n] for n,v in old.state_dict().items()) or any(signature(v)!=frozen_core[n] for n,v in core.state_dict().items()):
            raise RuntimeError('Historical frozen model changed')
        if any(p.grad is not None for p in old.parameters()) or any(p.grad is not None for p in core.parameters()): raise RuntimeError('Historical gradients leaked')
        atomic_json(folder/'frozen_verification.json',dict(old_model_unchanged=True,B4_unchanged=True,old_gradients_none=True,no_test_source_used=True))
        identity['status']=status; atomic_json(folder/'run_identity.json',identity); report(folder)
        verify(folder,complete=False)
        marker='COMPLETE.json' if status=='complete' else 'DIAGNOSTIC_COMPLETE.json'
        files=[f for f in folder.rglob('*') if f.is_file()]
        atomic_json(folder/marker,dict(version=VERSION,status=status,wall_seconds=time.monotonic()-start,
            result_sha256={str(f.relative_to(folder)):digest(f) for f in files if f.suffix!='.pth' and f.name!='bank.npz'},
            checkpoint_sha256={str(f.relative_to(folder)):digest(f) for f in files if f.suffix=='.pth'},
            bank_sha256={str(f.relative_to(folder)):digest(f) for f in files if f.name=='bank.npz'}))
        verify(folder)
    try:
        train_items=text_records(a.data_root,a.split_manifest,'train',cfg['train_samples'],cfg['seed'],tok,128)
        val=text_records(a.data_root,a.split_manifest,'validation',cfg['val_samples'],cfg['seed'],tok,128)
        if {r['image_id'] for r in train_items}&{r['image_id'] for r in val}: raise ValueError('Source split overlap')
        # No image file is opened/encoded. Frozen CLIP features only serve legacy diagnostic/reference.
        with torch.no_grad():
            for item in train_items[:cfg['diagnostic_samples']]+val:
                budget(); enc=encode_text(old.base,item['text'],128,device)
                item.update(text_features=enc['features'],text_mask=enc['mask'])
        design=dict(version=VERSION,smoke=a.smoke,config=cfg,channel_config=cc,arms=list(ARMS),
            tokenizer_ids=dict(bos=tok.bos_token_id,eos=tok.eos_token_id,pad=tok.pad_token_id),
            train_ids=[r['image_id'] for r in train_items],validation_ids=[r['image_id'] for r in val],
            train_source_sha256={str(r['image_id']):r['source_sha256'] for r in train_items},
            validation_source_sha256={str(r['image_id']):r['source_sha256'] for r in val},
            payload_complex_symbols=384,source_capacity_tokens=128,raw_utf8_wire_bytes=0,analog_repeats=1,
            slot_layout='48 text-only slots; frozen B4 Ctext16+T32 or new learned48; no image payload',
            source_case_note='Existing CLIP tokenizer normalizes case/spacing; not byte-lossless text transmission',
            resource=short_resource(cc),split='validation',no_image_source_read=True,
            primary='T2_fso minus T1_clean_continue, covered WER averaged per image over predeclared primary conditions',
            limits='screen, one training replicate/fixed weather/historical validation; no matched digital baseline or hardware/publication superiority')
        atomic_json(folder/'design.json',design)
        diag=train_items[:cfg['diagnostic_samples']]
        decoder_diagnostics(core,diag,tok,cfg,device,folder/'legacy_diagnostics',budget,atomic_json,atomic_checkpoint)
        model=TextJSCC(old.base,cfg,kernels(cc)).to(device).float().eval(); initial=state(model)
        atomic_checkpoint(folder/'initial.pth',dict(version=VERSION,weights=initial,config=cfg))
        trained,_=train(model,diag,cfg,device,folder/'native_overfit',1,cfg['diagnostic_lr'],'native_overfit',budget,
            atomic_json,atomic_checkpoint,updates=cfg['diagnostic_updates'])
        rr,groups=memory_tests(model.language,native_memories(model,diag,device),diag,tok,device,'native_overfit',budget)
        record_lines(folder/'native_overfit/samples.jsonl',rr); gate=overfit_gate(groups,cfg)
        gate.update(groups=groups,smoke_bypasses_quality_only=a.smoke); atomic_json(folder/'native_overfit/gate.json',gate)
        print('NATIVE OVERFIT gate={} actualWER={:.3f} margins={}'.format(gate['passed'],gate['matched_wer'],gate['memory_margins']),flush=True)
        if not gate['passed'] and not a.smoke: finish('overfit_stop'); return
        # Production starts again from initialization, NOT fitted32-sentence diagnostic weights.
        model.load_state_dict(initial,strict=True)
        t0,_=train(model,train_items,cfg,device,folder/'train_T0_clean',cfg['clean_epochs'],cfg['learning_rate'],'T0_clean',budget,atomic_json,atomic_checkpoint)
        rr,groups=memory_tests(model.language,native_memories(model,val,device),val,tok,device,'T0_clean_validation',budget)
        record_lines(folder/'clean_validation_samples.jsonl',rr)
        passed=groups['matched']['covered_wer']<=cfg['clean_validation_wer_gate'] and min(groups[k]['covered_wer']-groups['matched']['covered_wer'] for k in ('zero','shuffled'))>=cfg['memory_margin_gate']
        atomic_json(folder/'clean_validation_gate.json',dict(passed=bool(passed),groups=groups,quality_limit=cfg['clean_validation_wer_gate'],memory_margin_limit=cfg['memory_margin_gate'],smoke_bypasses_quality_only=a.smoke,
            exploratory_validation_not_final_test=True))
        if not passed and not a.smoke: finish('clean_validation_stop'); return
        bank=calibrate(cc,cfg,folder/'short_calibration',atomic_json,budget)
        proof=check_current(model,train_items,bank,cc,cfg,'T0_clean',device,budget); atomic_json(folder/'fidelity_T0_clean.json',proof)
        if not proof['passed']: raise RuntimeError('Current clean text payload fidelity failed; NO noisy training')
        for arm,noisy in (('T1_clean_continue',False),('T2_fso',True)):
            model.load_state_dict(t0,strict=True)
            train(model,train_items,cfg,device,folder/('train_'+arm),cfg['continue_epochs'],cfg['fine_learning_rate'],arm,budget,
                atomic_json,atomic_checkpoint,bank=bank if noisy else None)
            proof=check_current(model,train_items,bank,cc,cfg,arm,device,budget); atomic_json(folder/('fidelity_'+arm+'.json'),proof)
            if not proof['passed']: raise RuntimeError('Current learned short payload fidelity failed at '+arm)
        # T1/T2 are paired continuation from sameT0, source order/update count match; network sizes match.
        scorer=SemanticScorer(a.text_model_path,device,tok)
        (folder/'constellations').mkdir(); rows=[]
        with (folder/'samples.jsonl').open('x',encoding='utf-8') as samples,(folder/'events.jsonl').open('x') as events:
            for arm in ARMS:
                if arm!='L0_frozen_B4':
                    from utils.checkpoint import load_checkpoint
                    saved=load_checkpoint(folder/('train_'+arm)/'last.pth')
                    if saved['arm']!=arm or saved['config']!=cfg: raise ValueError('Wrong text arm checkpoint')
                    model.load_state_dict(saved['weights'],strict=True); del saved
                for condition in [None]+cc['conditions']:
                    cid='clean' if condition is None else condition['id']
                    weather=None if condition is None else paths(cc,condition['weather'],len(val),('text-validation',condition['weather']))
                    for i,item in enumerate(val):
                        budget(); started=time.monotonic(); ns=seed_for(cfg['seed'],'text-eval',cid,item['image_id'])
                        with torch.no_grad():
                            if arm=='L0_frozen_B4': x=legacy_wire(core,item,device)
                            else:
                                ids,lens=tensors([item],device); x=model.encode(ids,lens)
                            tx=x[0].cpu().numpy(); t=None
                            if condition is None:
                                y=x; rel,snr,ok=1.,40.,True
                                stats=dict(receiver_available=True,payload_complex_symbols=384,analog_repeats=1,raw_utf8_wire_bytes=0,
                                    physical_symbol_snr_db=None,pilot_snr_db=None,scheduled_seconds=None,launch_signal_energy_j=None,
                                    clean_control_no_physical_transmission=True,pre_fec_ber=None,post_fec_ber=None)
                            else:
                                yy,rel,snr,stats,t=wave(cc,condition,weather[i],ns,tx)
                                y=torch.as_tensor(yy[None],device=device,dtype=x.dtype); ok=stats['analog_receiver_available']
                                stats.update(receiver_available=ok,pilot_snr_db=snr,clean_control_no_physical_transmission=False)
                            memory=legacy_receive(core,y,rel,y.new_tensor([snr]),y.new_tensor([ok],dtype=torch.bool)) if arm=='L0_frozen_B4' else model.receive(y,y.new_tensor([rel]),y.new_tensor([snr]),y.new_tensor([ok]))
                            output=generate(core.language if arm=='L0_frozen_B4' else model.language,memory,tok,ok)
                        key='{}__{}__{}'.format(arm,item['image_id'],cid)
                        r=row(item,arm,cid,output,stats); r.update(frame_key=key,noise_seed=ns,
                            payload_sha256=signature(x),received_sha256=signature(y),offline_seconds=time.monotonic()-started,
                            semantic_chunked_clip_cosine=scorer.score(item['covered_text'],output['received_caption']))
                        if t is not None:
                            for event in t.events: events.write(json.dumps(dict(event,frame_key=key),allow_nan=False)+'\n')
                            if i<4:
                                name='constellations/'+key+'.svg'; constellation_svg(t.constellations,folder/name); r['constellation_file']=name
                        rows.append(r); samples.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n'); samples.flush(); events.flush()
                    print('TEXT EVAL {} {} complete={}/{}'.format(arm,cid,len(rows),len(ARMS)*len(val)*(len(cc['conditions'])+1)),flush=True)
        finish('complete')
    except BaseException as exc:
        atomic_json(folder/'FAILED.json',dict(version=VERSION,type=type(exc).__name__,message=str(exc),wall_seconds=time.monotonic()-start))
        raise


if __name__=='__main__': main()
