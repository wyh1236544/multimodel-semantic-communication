"""Inference-only paired validation of the four completed FSO-JSCC V1 arms."""
import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
EXT = Path(__file__).resolve().parents[1]
PROJECT = EXT.parent
sys.path.insert(0, str(EXT))
from fso_jscc_v1 import ARMS
from fso_jscc_eval_v1 import VERSION
from fso_jscc_eval_v1.core import load, settings, proxy_conditions, select_ids, expected_counts, replay_check, state_fingerprint, digest, seed_for, TEMPLATE


def main():
    import torch
    import torch.nn.functional as F
    import v10.model as module
    from v10.runtime import add_shared_args, preflight, load_config
    from v10.io import begin_run, atomic_json
    from v10.data import build_loader, validate_manifest
    from v10.model import build_model
    from utils.checkpoint import load_checkpoint
    from utils.metrics import batch_image_metrics, token_accuracy, sentence_bleu2
    from token_guidance.model import Guidance
    from final_compare_v1.metrics import caption_metrics
    from optical_realism_v3.spec import resources
    from optical_realism_v3.transport import RealisticTransport
    from optical_realism_v3.constellation import constellation_svg
    from eval_v10 import save_image
    from fso_jscc_v1.model import ContinuousChannel, Frame, parameters
    from fso_jscc_v1.report import verify as verify_parent, read_rows
    from fso_jscc_eval_v1.checkpoint import apply_delta, advance_parent_vae_rng
    from fso_jscc_eval_v1.report import verify, write_report

    p = add_shared_args(argparse.ArgumentParser(description=__doc__, allow_abbrev=False))
    p.add_argument('--a-delta', required=True); p.add_argument('--source-run', required=True)
    p.add_argument('--experiment-config', default=str(TEMPLATE)); p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if not math.isfinite(a.max_hours) or not 0 < a.max_hours <= 12:
        p.error('Budget must be >0 and <=12h, not an ETA')
    if a.device.startswith('cuda') and os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('Use launch.py: this task is scoped to physical GPU1')
    started = time.monotonic()
    def budget():
        if time.monotonic()-started > a.max_hours*3600:
            raise TimeoutError('Evaluation wall-time budget reached; partial evidence preserved')
    c = settings(load(a.experiment_config), a.smoke)
    source = Path(a.source_run).resolve(strict=True)
    source.relative_to(PROJECT)
    parent, original_rows = verify_parent(source, complete=True)
    if parent['smoke'] or parent['config']['val_samples'] != 32 or len(parent['arms']) != 4:
        raise ValueError('Use the completed formal 32-image direction screen, not smoke weights')
    manifest = json.loads(Path(a.split_manifest).read_text(encoding='utf-8'))
    validate_manifest(manifest)
    ids, old, new = select_ids(manifest, parent, c, a.smoke)
    source_identity = json.loads((source/'run_identity.json').read_text())
    # Pin the actually executed old/new/physical sources, not only version strings.
    for field, prefix in [('source_sha256', EXT), ('new_source_sha256', EXT/'fso_jscc_v1'), ('physical_source_sha256', EXT)]:
        for name, expected in source_identity[field].items():
            if digest(prefix/name) != expected:
                raise ValueError('Parent source differs: '+name+'; do not silently reinterpret old weights')
    weight_paths = {'base': Path(a.checkpoint), 'A': Path(a.a_delta)}
    weight_paths.update({arm:source/('train_'+arm)/'last_delta.pth' for arm in ARMS[1:]})
    before = {key:digest(path) for key,path in weight_paths.items()}
    if before['base'] != source_identity['checkpoint_sha256'] or before['A'] != source_identity['a_delta_sha256'] or digest(a.split_manifest) != source_identity['split_sha256']:
        raise ValueError('Base/A/frozen split differs from the completed source run')
    source_done = json.loads((source/'COMPLETE.json').read_text())
    for arm in ARMS[1:]:
        if before[arm] != source_done['checkpoint_sha256']['train_'+arm+'/last_delta.pth']:
            raise ValueError('Source trained checkpoint missing/changed: '+arm)
    a.batch_size, a.num_workers, a.no_amp, a.fixed_mode, a.csi, a.blocks, a.seed = 1, 0, True, 0, 'pilot', 1, parent['config']['seed']
    device = preflight(a); config = load_config(a)
    base, delta = load_checkpoint(a.checkpoint), load_checkpoint(a.a_delta)
    if (base.get('split_sha256') != source_identity['split_sha256'] or delta.get('version') != 'V10-token-guidance-quick-v1' or
            delta.get('variant') != 'A' or delta.get('base_checkpoint_sha256') != before['base'] or delta.get('split_sha256') != source_identity['split_sha256']):
        raise ValueError('Exact original base/A checkpoint identities required')
    config['v10'] = copy.deepcopy(base['config']['v10']); config['v10'].update(blocks=1,csi='pilot',fixed_mode=0)
    config['model']['vae_use_fp16'] = False; del base
    physical_plan, physical_cases = copy.deepcopy(parent['physical_plan']), copy.deepcopy(parent['physical_cases'])
    known = {x['id']:x for x in physical_plan['conditions']}
    if any(x not in known for x in c['physical_conditions']):
        raise ValueError('Unknown fixed parent V3 physical workpoint')
    conditions = [known[x] for x in c['physical_conditions']]
    resource = resources(physical_cases[conditions[0]['case']],physical_plan)
    if resource != parent['physical_resource'] or physical_plan['analog_repeats'] != 1:
        raise ValueError('Changed R=1 physical resource plan')
    directory, identity = begin_run(a.output_dir,a,config,EXT)
    identity.update(version=VERSION,training_performed=False,a_delta_sha256=before['A'],gpu_name=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        new_source_sha256={f.name:digest(f) for f in sorted(Path(__file__).parent.iterdir()) if f.suffix in ('.py','.json','.md')})
    atomic_json(directory/'run_identity.json',identity)
    try:
        evidence = ['COMPLETE.json','design.json','run_identity.json','checkpoint_load.json']
        evidence += ['train_'+arm+'/COMPLETE.json' for arm in ARMS]
        evidence += ['train_'+arm+'/history.jsonl' for arm in ARMS[1:]]
        for name in evidence:
            target = directory/'source_evidence'/name; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(str(source/name),str(target))
        shutil.copyfile(a.split_manifest,str(directory/'split_evidence.json'))
        receipt = dict(source_run=str(source),source_copy_sha256={name:digest(source/name) for name in evidence},
            weights_before=before,weights_after={},parent_verified_at_start_and_end=False,large_weights_not_copied=True)
        atomic_json(directory/'input_receipt.json',receipt)
        print('Loading exact original A and THREE saved deltas. INFERENCE ONLY; no optimizer or training.',flush=True)
        model, exact = build_model(config,device,a.checkpoint)
        if exact != {'v10_exact':True}:
            raise ValueError('Exact base architecture/load required')
        model.float(); guide = Guidance(model)
        if set(delta['delta']) != {n for n,_ in guide.trainable('A')}:
            raise ValueError('A delta key mismatch')
        model.load_state_dict(dict(model.state_dict(),**delta['delta']),strict=True); del delta
        guide.set('A'); model.base.channel = ContinuousChannel(model.base.channel,parent['config']).to(device)
        named = parameters(model); initial = {n:q.detach().cpu().clone() for n,q in named}
        model.requires_grad_(False); model.eval()
        print('Replaying {} omitted VAE cache RNG draws; NO training images, optimizer or weight update.'.format(parent['config']['train_samples']),flush=True)
        zero=torch.zeros((1,3,int(config['model']['image_size']),int(config['model']['image_size'])),device=device)
        rng_receipt=advance_parent_vae_rng(model.base.vae_wrapper,zero,parent['config']['train_samples'],budget)
        atomic_json(directory/'latent_rng_replay.json',rng_receipt); del zero
        cached = []
        print('Caching {} validation inputs/VAE latents; no train/test data.'.format(c['samples']),flush=True)
        with torch.no_grad():
            loader = build_loader(config,a.split_manifest,'validation',c['samples'],a.seed)
            loader.dataset.caption_mode = 'first'
            for batch in loader:
                budget(); image = batch['images']; text = batch['texts'][0]; image_id = int(batch['image_ids'][0])
                source_hash = hashlib.sha256(image.numpy().tobytes()+text.encode()).hexdigest()
                if image_id in old and source_hash != parent['validation_source_sha256'][str(image_id)]:
                    raise ValueError('Changed pilot image/caption preprocessing')
                latent=model.base.vae_wrapper.encode_images(image.to(device)).detach().cpu()
                cached.append(dict(image_id=image_id,text=text,images=image,source_sha256=source_hash,
                    latent=latent,source_latent_sha256=hashlib.sha256(latent.numpy().tobytes()).hexdigest()))
        if [x['image_id'] for x in cached] != ids:
            raise ValueError('Cached loader order differs from frozen split')
        n_old = min(len(old),max(1,c['save_images']//2)) if c['save_images'] else 0
        examples = (old[:n_old]+new[:c['save_images']-n_old])[:c['save_images']]
        if len(examples)<min(c['save_images'],len(ids)):
            examples += [i for i in ids if i not in examples][:c['save_images']-len(examples)]
        design = dict(version=VERSION,config=c,smoke=a.smoke,split='validation',arms=list(ARMS),
            validation_ids=ids,pilot_ids=old,new_ids=new,training_performed=False,seed_replicates=1,
            validation_source_sha256={str(x['image_id']):x['source_sha256'] for x in cached},
            validation_latent_sha256={str(x['image_id']):x['source_latent_sha256'] for x in cached},
            proxy_conditions=proxy_conditions(c),physical_conditions=conditions,physical_resource=resource,
            physical_plan=physical_plan,physical_cases=physical_cases,example_ids=examples,
            primary='NEW images: M3-M2 PSNR averaged fixed low-power conditions and independent draws per image',
            expected_counts=expected_counts(c),source_split_sha256=source_identity['split_sha256'],
            no_original_caption_at_receiver=True,all_inputs_and_channel_draws_paired=True)
        atomic_json(directory/'design.json',design); (directory/'images').mkdir(); (directory/'constellations').mkdir()
        loads = dict(base_exact=True,a_exact=True,base_sha256=before['base'],a_delta_sha256=before['A'],arms={})
        rows, timings = [], {e:dict(rows=0,seconds=0.) for e in ('proxy','waveform_v3')}
        expected = sum(design['expected_counts'].values())
        with (directory/'samples.jsonl').open('x',encoding='utf-8') as stream, (directory/'events.jsonl').open('x',encoding='utf-8') as events:
            def emit(item,engine,condition,draw,context):
                tick=time.monotonic(); budget()
                with patch.object(module,'LinkFrame',lambda *args,**kw:Frame(*args,context=context,config=parent['config'],**kw)):
                    out=model(item['images'].to(device),[item['text']],'awgn',0.,vae_latents_override=item['latent'].to(device))
                metrics=batch_image_metrics(out['image'],item['images'].to(device))
                caption=model.base.decode_text_logits(out['text_logits'])[0]
                correct,count=token_accuracy(out['text_logits'],out['text_input_ids'],out['text_attention_mask'])
                row=dict(model.last_frame.last_stats,arm=arm,engine=engine,condition=condition,draw=draw,
                    image_id=item['image_id'],image_subset='pilot' if item['image_id'] in old else 'new',source_sha256=item['source_sha256'],
                    source_latent_sha256=item['source_latent_sha256'],
                    field_seed=context['field_seed'],noise_seed=context['noise_seed'],psnr=float(metrics['psnr'][0]),ssim=float(metrics['ssim'][0]),
                    token_acc=correct/max(count,1),token_count=count,bleu2=sentence_bleu2(item['text'],caption),
                    source_caption_evaluation_only=item['text'],received_caption=caption,neural_input_capacity_tokens=int(out['text_input_ids'].shape[1]),
                    source_utf8_bytes=len(item['text'].encode()),source_rgb8_bytes=item['images'].numel(),raw_utf8_wire_bytes=0,
                    analog_repeats=1,pre_fec_ber=None,post_fec_ber=None,**caption_metrics(item['text'],caption))
                row['receiver_available']=bool(model.last_frame.header_ok[0])
                row['diagnostic_joint_success']=bool(row['receiver_available'] and row['psnr']>=c['image_psnr_target_db'] and row['normalized_wer']<=c['normalized_wer_target'])
                target=out['common_text_semantic_target'].float()
                row['common_to_source_text_cosine']=float(F.cosine_similarity(out['common_receiver_semantic'].float(),target,dim=-1)[0])
                row['receiver_text_to_source_text_cosine']=float(F.cosine_similarity(out['receiver_text_semantic'].float(),target,dim=-1)[0])
                row['semantic_metric_note']='inherited CLIP-family alignment, NOT independent semantic correctness'
                if item['image_id'] in examples:
                    source_file=directory/'images'/('{}__source.png'.format(item['image_id']))
                    if not source_file.exists(): save_image(item['images'][0],source_file)
                    row['source_image_file']='images/'+source_file.name
                    row['image_file']='images/{}__{}__{}__{}__d{}.png'.format(item['image_id'],arm,engine,condition,draw)
                    save_image(out['image'][0],directory/row['image_file'])
                    for k in ('source_image_file','image_file'): row[k+'_sha256']=digest(directory/row[k])
                transport=model.last_frame.transport
                if transport is not None:
                    row['event_count']=len(transport.events)
                    for e in transport.events:
                        events.write(json.dumps(dict(e,arm=arm,engine=engine,condition=condition,image_id=item['image_id'],draw=draw),allow_nan=False)+'\n')
                    events.flush()
                    if item['image_id'] in examples and transport.constellations:
                        row['constellation_file']='constellations/{}__{}__{}__d{}.svg'.format(item['image_id'],arm,condition,draw)
                        constellation_svg(transport.constellations,directory/row['constellation_file'])
                        row['constellation_file_sha256']=digest(directory/row['constellation_file'])
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); stream.flush(); rows.append(row)
                timings[engine]['rows']+=1; timings[engine]['seconds']+=time.monotonic()-tick
                if len(rows)==1 or len(rows)%20==0:
                    elapsed=time.monotonic()-started
                    remaining=sum((design['expected_counts'][e]-t['rows'])*t['seconds']/t['rows'] for e,t in timings.items() if t['rows'])
                    progress=dict(stage='evaluation',arm=arm,rows=len(rows),expected_rows=expected,engine=engine,
                        elapsed_seconds=elapsed,projected_remaining_seconds=remaining,estimate_excludes_unmeasured_engines_and_reporting=True)
                    atomic_json(directory/'progress.json',progress)
                    print('EVAL {} {}/{} {} {} PSNR={:.2f} WER={:.3f} elapsed={:.1f}s'.format(arm,len(rows),expected,engine,condition,row['psnr'],row['normalized_wer'],elapsed),flush=True)
                del out
            with torch.no_grad():
                for arm in ARMS:
                    budget()
                    for n,q in named: q.copy_(initial[n].to(device))
                    if arm!=ARMS[0]:
                        record=json.loads((source/('train_'+arm)/'COMPLETE.json').read_text())
                        payload=load_checkpoint(weight_paths[arm])
                        apply_delta(model,named,payload,record,parent,before['base'],before['A'],source_identity['split_sha256']); del payload
                    model.requires_grad_(False); model.eval()
                    fingerprint=state_fingerprint(model)
                    loads['arms'][arm]=dict(exact=True,checkpoint_sha256=before.get(arm),all_parameters_frozen=True,state_before=fingerprint,state_after=None)
                    atomic_json(directory/'checkpoint_load.json',loads)
                    for item in cached:
                        for condition in design['proxy_conditions']:
                            pro,snr=condition['profile'],condition['snr']
                            for draw in range(1 if pro=='clean' else c['proxy_draws']):
                                context=dict(profile=pro,snr=snr,field_seed=seed_for(a.seed,item['image_id'],pro,draw,'valfield'),
                                    noise_seed=seed_for(a.seed,item['image_id'],condition['id'],draw,'valnoise'))
                                emit(item,'proxy',condition['id'],draw,context)
                        for condition in conditions:
                            for draw in range(c['physical_draws']):
                                field=seed_for(a.seed,item['image_id'],condition['case'],draw,'physicalfield')
                                noise=seed_for(a.seed,item['image_id'],condition['case'],draw,'physicalnoise')
                                transport=RealisticTransport(physical_cases[condition['case']],field,noise,condition['launch_signal_dbm'],
                                    resource['slot_samples'],physical_plan['laser_max_dbm'],physical_plan['hardware'])
                                emit(item,'waveform_v3',condition['id'],draw,dict(field_seed=field,noise_seed=noise,transport=transport))
                    after_fingerprint=state_fingerprint(model)
                    if fingerprint!=after_fingerprint: raise RuntimeError('Frozen model state changed')
                    loads['arms'][arm]['state_after']=after_fingerprint; atomic_json(directory/'checkpoint_load.json',loads)
        verify_parent(source,complete=True)
        receipt.update(weights_after={k:digest(v) for k,v in weight_paths.items()},parent_verified_at_start_and_end=True)
        if receipt['weights_before']!=receipt['weights_after']: raise RuntimeError('Original input checkpoint changed')
        atomic_json(directory/'input_receipt.json',receipt)
        atomic_json(directory/'replay.json',replay_check(rows,original_rows))
        atomic_json(directory/'timing.json',dict(engines=timings,formal_wall_seconds=time.monotonic()-started,estimate_not_promise=True))
        budget(); verify(directory); write_report(directory)
        identity['status']='complete'; atomic_json(directory/'run_identity.json',identity)
        files=[f for f in directory.rglob('*') if f.is_file() and f.name!='COMPLETE.json' and f.suffix not in ('.pth','.pt')]
        # The original COMPLETE.json inside source_evidence is required provenance.
        files += [directory/'source_evidence'/n for n in evidence if Path(n).name=='COMPLETE.json']
        atomic_json(directory/'COMPLETE.json',dict(version=VERSION,rows=len(rows),training_performed=False,wall_seconds=time.monotonic()-started,
            result_sha256={str(f.relative_to(directory)):digest(f) for f in sorted(set(files))}))
        verify(directory,complete=True)
        print('COMPLETE expanded evaluation: '+str(directory/'report.html'),flush=True)
    except BaseException as exc:
        identity.update(status='failed',error=repr(exc)); atomic_json(directory/'run_identity.json',identity)
        atomic_json(directory/'FAILED.json',dict(error=repr(exc),partial_evidence_preserved=True,no_training=True)); raise


if __name__=='__main__':
    main()
