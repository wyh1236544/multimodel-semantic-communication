"""Independent CPU evidence check/paired aggregation. Not new neural inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT),str(EXT.parent)]
from docci_text_guided_image_v20.core import (read,rows,sha,atomic,load,batches,train_snr,
    trial_matrix,bootstrap,GEOMETRY,ARMS,IMAGE_SHA,TEXT_SHA,DOCCI_SHA,Budget,seed,array_sha)
from docci_text_guided_image_v20.channel import sample

def close(a,b):
    return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)

def telemetry(s,kind,eid,snr,repeat):
    n = 8192 if kind=='image' else 18432
    value = seed(42,'v20-AWGN','eval',0,eid,kind,snr,repeat)
    epsilon = sample(n,snr,value)
    pilot = sample(64,snr,seed(value,'pilot'))
    pn = float(np.mean(epsilon.astype(np.float64)**2))
    n0 = float(np.mean(pilot.astype(np.float64)**2))
    if (s['noise_seed']!=value or s['noise_sha256']!=array_sha(epsilon)
            or s['pilot_sha256']!=array_sha(pilot) or s['nominal_snr_db']!=snr
            or not close(s['noise_power'],pn) or not close(s['pilot_noise_variance'],n0)
            or abs(s['signal_power']-1)>.0002 or s['post_noise_renormalization']):
        raise ValueError('Deterministic telemetry differs')
    if snr is None:
        if s['estimated_snr_db'] is not None or s['realized_snr_db'] is not None:
            raise ValueError('clean is not 0dB or measured40dB')
    else:
        est = max(-40.,min(40.,-10*math.log10(max(n0,1e-12))))
        if not close(s['estimated_snr_db'],est) or not close(s['realized_snr_db'],10*math.log10(s['signal_power']/pn)):
            raise ValueError('Reported receive SNR differs')

def check_cache(folder,c,smoke,full,budget):
    receipt = read(folder/'CACHE_COMPLETE.json')
    if sha(folder/'index.json')!=receipt['index_sha256']:
        raise ValueError('Cache index changed')
    if (receipt['source_sha256']!=DOCCI_SHA or
            {k:v['sha256'] for k,v in receipt['parents'].items()}!={'image':IMAGE_SHA,'text':TEXT_SHA}):
        raise ValueError('Wrong data/checkpoint parent')
    entries = read(folder/'index.json')['records']
    groups = {k:[r for r in entries if r['cohort']==k] for k in ('train','dev','qual_dev')}
    expected = dict(train=c['train_samples'],dev=c['dev_samples'],qual_dev=2 if smoke else 100)
    if {k:len(v) for k,v in groups.items()}!=expected or len({r['example_id'] for r in entries})!=len(entries):
        raise ValueError('Wrong cache cohort/duplicate ID')
    protected = read(folder/'protected_split.json')
    if any(protected[k]!=[r['example_id'] for r in groups[k]] for k in groups):
        raise ValueError('Protected split differs')
    if set(protected['heldout']) & set(protected['train']):
        raise ValueError('Held-out development leaked into training')
    for a,b in (('train','dev'),('train','qual_dev'),('dev','qual_dev')):
        for field in ('text_sha256','image_sha256'):
            if {r[field] for r in groups[a]} & {r[field] for r in groups[b]}:
                raise ValueError('Source duplicate across cohorts')
    for r in entries:
        budget()
        if r['official_split']!=('qual_dev' if r['cohort']=='qual_dev' else 'train'):
            raise ValueError('Official split differs')
        if hashlib.sha256(r['text'].encode()).hexdigest()!=r['text_sha256'] or r['input_truncated']:
            raise ValueError('Source text changed/truncated')
        if full:
            if sha(Path(r['source_image_path']))!=r['image_sha256']:
                raise ValueError('Original image bytes changed after source audit')
            p = folder/r['cache_file']
            if sha(p)!=r['cache_sha256']:
                raise ValueError('Actual cache file changed')
            with np.load(p,allow_pickle=False) as z:
                if set(z.files)!=set(r['array_hashes']):
                    raise ValueError('Cache geometry keys differ')
                shapes = dict(rgb=(512,512,3),rectangle=(4,),latent=(4,64,64),
                    source_semantics=(32,512),text_wire=(18432,),ids=(9,64),oracle_memory=(288,768))
                for k,shape in shapes.items():
                    x = z[k]
                    if x.shape!=shape or not np.isfinite(x).all() or array_sha(x)!=r['array_hashes'][k]:
                        raise ValueError('Invalid cache array '+k)
                    expected_dtype = np.uint8 if k=='rgb' else np.int64 if k in ('rectangle','ids') else np.float32
                    if x.dtype!=expected_dtype:
                        raise ValueError('Cache array dtype differs '+k)
                if abs(float(np.mean(z['text_wire']**2))-1)>.0002:
                    raise ValueError('V18 fixed text unit power differs')
    return groups

def check_train(folder,arm,c,records,full,budget):
    ident = read(folder/'identity.json')
    total = 2 if ident['smoke'] else math.ceil(len(records)/c['accumulation'])*c['epochs']
    history = rows(folder/'history.jsonl')
    done = read(folder/'TRAIN_COMPLETE.json')
    if (ident['arm']!=arm or ident['geometry']!=GEOMETRY or ident['updates']!=total
            or len(history)!=total or done['updates']!=total or not done['fixed_final']
            or not done['actual_disk_roundtrip']):
        raise ValueError('Incomplete training schedule/identity')
    if ident['train_ids']!=[r['example_id'] for r in records]:
        raise ValueError('TRAIN cohort differs')
    seen = set()
    for step,((epoch,indices),r) in enumerate(zip(batches(len(records),c),history),1):
        budget()
        ids = [records[i]['example_id'] for i in indices]
        if (r['update']!=step or r['total_updates']!=total or
                [x['example_id'] for x in r['observations']]!=ids or r['epoch']!=epoch+1):
            raise ValueError('Training sample order/update differs')
        seen.update(ids)
        if not all(math.isfinite(r[k]) for k in ('loss','gradient_norm','learning_rate')):
            raise ValueError('Nonfinite training record')
        objectives = []
        for item in r['observations']:
            if item['snr_db']!=train_snr(c,epoch,item['example_id']):
                raise ValueError('Arm-specific channel selection')
            if abs(item['image']['signal_power']-1)>.0002 or abs(item['text']['signal_power']-1)>.0002:
                raise ValueError('Transmit energy differs')
            objectives.append(item['pixel_MSE']+c['latent_weight']*item['latent_MSE']+c['lpips_weight']*item['LPIPS'])
        if not math.isclose(r['loss'],float(np.mean(objectives)),rel_tol=2e-6,abs_tol=2e-7):
            raise ValueError('Training loss does not match declared components')
    if done['unique_ids_seen']!=len(seen):
        raise ValueError('Training coverage differs')
    audit = read(folder/'learning_audit.json')
    if (set(audit['groups'])!={'image_encoder','image_decoder','tx_fusion','rx_fusion'}
            or not audit['passed'] or audit['frozen_before']!=audit['frozen_after']
            or not all(v['nonzero_gradient'] and v['changed'] and v['parameters'] for v in audit['groups'].values())
            or sum(v['parameters'] for v in audit['groups'].values())!=ident['trainable_parameters']):
        raise ValueError('Real learning/frozen groups audit failed')
    if full and sha(folder/'last.pth')!=done['checkpoint_sha256']:
        raise ValueError('Actual saved final weight differs')
    return ident

def check_evaluation(folder,arm,c,groups,smoke,budget):
    identity = read(folder/'identity.json')
    receipt = read(folder/'EVAL_COMPLETE.json')
    data = rows(folder/'rows.jsonl')
    if (identity['arm']!=arm or identity['geometry']!=GEOMETRY or
            receipt['rows']!=len(data) or sha(folder/'rows.jsonl')!=receipt['rows_sha256']
            or sha(folder/'text_rows.jsonl')!=receipt['text_rows_sha256']):
        raise ValueError('Evaluation seal/geometry differs')
    expected = {(r['example_id'],cohort,s,repeat,control):r
                for cohort in ('dev','qual_dev') for r in groups[cohort]
                for s,repeat,control in trial_matrix(c,arm,smoke)}
    seen = {}
    for r in data:
        budget()
        key = (r['example_id'],r['cohort'],r['snr_db'],r['repeat'],r['control'])
        if key in seen or key not in expected:
            raise ValueError('Unexpected/duplicate evaluation case')
        source = expected[key]
        if (r['official_split']!=source['official_split'] or
                r['source_text_sha256']!=source['text_sha256'] or r['source_image_sha256']!=source['image_sha256']
                or r['source_text_at_RX']!=(r['control']=='rx_oracle') or r['arm']!=arm):
            raise ValueError('Source identity or privileged-data label differs')
        if not (0<=r['MSE']<=1 and math.isfinite(r['LPIPS']) and r['LPIPS']>=0 and -1<=r['ssim']<=1):
            raise ValueError('Invalid reconstruction metric')
        if abs(r['psnr']+10*math.log10(max(r['MSE'],1e-12)))>.002:
            raise ValueError('PSNR disagrees with saved MSE')
        for kind in ('image','text'):
            telemetry(r[kind+'_stats'],kind,r['example_id'],r['snr_db'],r['repeat'])
        if r['control']=='rx_shuffled':
            cohort_ids = [x['example_id'] for x in groups[r['cohort']]]
            donor = cohort_ids[(cohort_ids.index(r['example_id'])+1)%len(cohort_ids)]
            if r['donor_id']!=donor:
                raise ValueError('Wrong/identical/cross-cohort donor')
            telemetry(r['delivered_text_stats'],'text',donor,r['snr_db'],r['repeat'])
        elif r['donor_id'] is not None:
            raise ValueError('Unexpected donor')
        for name,hash_field in (('image_file','image_sha256'),('source_image_file','source_view_sha256')):
            if r[name] is not None and sha(folder/r[name])!=r[hash_field]:
                raise ValueError('Saved actual image changed')
        seen[key] = r
    if set(seen)!=set(expected):
        raise ValueError('Missing evaluation cases')
    if arm=='C_tx_rx_text':
        for key,r in seen.items():
            if r['control']!='matched':
                anchor = seen[(r['example_id'],r['cohort'],r['snr_db'],0,'matched')]
                if any(r[k]!=anchor[k] for k in ('tx_image_sha256','observed_image_sha256','image_stats','text_stats')):
                    raise ValueError('RX control changed own TX/image noise')
    text_data = rows(folder/'text_rows.jsonl')
    text_expected = {(r['example_id'],cohort,s):r for cohort in ('dev','qual_dev')
                     for r in groups[cohort] for s in ([None,0] if smoke else c['evaluation_snr_db'])}
    if arm!='C_tx_rx_text':
        text_expected = {}
    text_seen = set()
    from docci_text_continue_v11.metrics import metrics
    from docci_reconstruction_v10.core import joined_ids
    for r in text_data:
        key = (r['example_id'],r['cohort'],r['snr_db'])
        if key in text_seen or key not in text_expected:
            raise ValueError('Duplicate/unexpected text output')
        origin = text_expected[key]
        if (r['source_text']!=origin['text'] or r['source_text_sha256']!=origin['text_sha256']
                or r['image_assisted_text'] is not False):
            raise ValueError('Source text differs or false bidirectional claim')
        if len(r['chunk_generated_token_ids'])!=9:
            raise ValueError('Incomplete public text frame generation')
        joined,capacity = joined_ids(r['chunk_generated_token_ids'],0,2,1)
        if joined!=r['generated_token_ids'] or capacity!=r['stopped_by_capacity']:
            raise ValueError('Generated public text framing differs')
        for metric,value in metrics(origin['text'],r['received_caption']).items():
            if isinstance(value,(int,float)) and not close(r[metric],value):
                raise ValueError('Saved text metric differs '+metric)
        text_seen.add(key)
    if text_seen!=set(text_expected) or receipt['text_rows']!=len(text_data):
        raise ValueError('Missing text outputs')
    return data

def summary(data):
    result = []
    keys = sorted({(r['cohort'],r['snr_db'],r['control']) for r in data},key=str)
    for cohort,snr,control in keys:
        selected = [r for r in data if (r['cohort'],r['snr_db'],r['control'])==(cohort,snr,control)]
        result.append(dict(cohort=cohort,snr_db=snr,control=control,rows=len(selected),
            means={k:float(np.mean([r[k] for r in selected])) for k in ('psnr','ssim','LPIPS')}))
    return result

def paired(left,right,metric,cohort='dev',snr=0,left_control='matched',right_control='matched'):
    # left is the reference; positive = lower LPIPS / higher PSNR in right.
    def means(data,control):
        result = {}
        for r in data:
            if (r['cohort'],r['snr_db'],r['control'])==(cohort,snr,control):
                result.setdefault(r['example_id'],[]).append(r[metric])
        return {k:float(np.mean(v)) for k,v in result.items()}
    a,b = means(left,left_control),means(right,right_control)
    if set(a)!=set(b):
        raise ValueError('Unpaired description set')
    values = [(a[k]-b[k])*(1 if metric=='LPIPS' else -1) for k in sorted(a)]
    return bootstrap(values)

def verify(root,full=False,max_hours=1):
    root = Path(root)
    budget = Budget(max_hours)
    pipeline = read(root/'PIPELINE.json')
    config = read(root/'config.json')
    result = dict(status=pipeline['status'],stages={},comparisons=None,facts='PENDING',
        full_semantic_success=False,FSO='NOT_RUN',image_to_text='NOT_IMPLEMENTED_THIS_VERSION',
        numeric_evaluation_only=True,full_arrays_and_weights_verified=full,
        LPIPS_not_recomputed_from_all_neural_outputs=True)
    for prefix in ('smoke_',''):
        smoke = bool(prefix)
        cache = root/(prefix+'cache')
        if not (cache/'CACHE_COMPLETE.json').is_file():
            continue
        c = load(root/'config.json',smoke)
        groups = check_cache(cache,c,smoke,full,budget)
        result['stages'][prefix+'cache'] = 'VERIFIED'
        identities = []
        evaluations = {}
        for arm in ARMS:
            train = root/(prefix+'train_'+arm)
            ev = root/(prefix+'eval_'+arm)
            if (train/'TRAIN_COMPLETE.json').is_file():
                ident = check_train(train,arm,c,groups['train'],full,budget)
                if ident['cache_index_sha256']!=sha(cache/'index.json'):
                    raise ValueError('Training used a different joint cache')
                identities.append(ident)
                result['stages'][train.name] = 'VERIFIED'
            if (ev/'EVAL_COMPLETE.json').is_file():
                if not (train/'TRAIN_COMPLETE.json').is_file():
                    raise ValueError('Evaluation without verified training')
                identity = read(ev/'identity.json')
                if (identity['checkpoint_sha256']!=read(train/'TRAIN_COMPLETE.json')['checkpoint_sha256']
                        or identity['cache_index_sha256']!=sha(cache/'index.json')):
                    raise ValueError('Evaluation used a different checkpoint')
                evaluations[arm] = check_evaluation(ev,arm,c,groups,smoke,budget)
                result['stages'][ev.name] = 'VERIFIED'
        if identities and (len({x['trainable_parameters'] for x in identities})!=1 or
                any(x['initial_hashes']!=identities[0]['initial_hashes'] for x in identities)):
            raise ValueError('Arms did not share initial parameters/capacity')
        if len(evaluations)==3:
            agg = {k:summary(v) for k,v in evaluations.items()}
            pairs = {}
            for a,b in (('A_self','B_tx_text'),('B_tx_text','C_tx_rx_text'),('A_self','C_tx_rx_text')):
                pairs[b+'_over_'+a] = {m:paired(evaluations[a],evaluations[b],m) for m in ('LPIPS','psnr')}
            # Compare donor vs matched with IDENTICAL r0 noise, not donor r0 vs
            # the mean of three matched draws.
            actual = [r for r in evaluations['C_tx_rx_text'] if r['repeat']==0]
            pairs['correct_over_received_donor'] = {m:paired(actual,actual,m,
                left_control='rx_shuffled',right_control='matched') for m in ('LPIPS','psnr')}
            numerical = (pairs['C_tx_rx_text_over_B_tx_text']['LPIPS']['ci'][0]>0
                and pairs['C_tx_rx_text_over_A_self']['LPIPS']['ci'][0]>0
                and pairs['correct_over_received_donor']['LPIPS']['ci'][0]>0
                and pairs['C_tx_rx_text_over_B_tx_text']['psnr']['ci'][0]>=-.2)
            value = dict(aggregates=agg,paired_dev_awgn0=pairs,
                numerical_screen_passed=numerical,primary='dev AWGN0 LPIPS C over B',
                pointwise_CI=True,multiple_comparisons_corrected=False,training_seeds=1,
                repeats_averaged_within_image=True,facts='PENDING',
                text_not_helped_by_image=True,full_semantic_success=False)
            if smoke:
                result['smoke_numerical_values_NOT_quality_claim'] = value
            else:
                result['comparisons'] = value
    completed = {k for k,v in pipeline['stages'].items() if v=='VERIFIED'}
    if not completed.issubset(result['stages']):
        raise ValueError('Pipeline falsely marks an unverified stage complete')
    if pipeline['status'] in ('COMPLETE','SMOKE_COMPLETE') and set(pipeline['stages'])!=set(result['stages']):
        raise ValueError('Completion marker with missing stages')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root')
    p.add_argument('--full',action='store_true')
    p.add_argument('--max-hours',type=float,default=1)
    a=p.parse_args()
    value=verify(a.root,a.full,a.max_hours)
    print(json.dumps(value,ensure_ascii=False,indent=2))
