"""Independent CPU verification, no torch imports or checkpoint deserialization."""
import argparse
import hashlib
import math
from pathlib import Path
import statistics
import sys
import time
import numpy as np
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_text_channel_v18.core import (FOLDER,ARMS,PARENT_SHA,PARAMETERS,read,sha,lines,
    load,atomic,batches,channel_for,trial_seed,noise,trials)
from docci_rx_bridge_v16.report import verify_metrics,paired
from docci_text_continue_v11.metrics import summary

def close(a,b): return math.isfinite(a) and math.isclose(a,b,rel_tol=3e-6,abs_tol=3e-7)
def seal_check(root,budget):
    if read(root/'SEALED.json')['manifest_sha256']!=sha(root/'LIGHT_MANIFEST.json'): raise ValueError('Stage seal differs')
    for n,h in read(root/'LIGHT_MANIFEST.json').items():
        budget();p=Path(n)
        if p.is_absolute() or '..' in p.parts or not p.parts: raise ValueError('Unsafe light member')
        if sha(root/p)!=h: raise ValueError('Sealed result changed: '+n)
def source_check(root):
    m=read(FOLDER/'MANIFEST.json')
    if sha(root/'source_snapshot/MANIFEST.json')!=sha(FOLDER/'MANIFEST.json'): raise ValueError('Wrong source release')
    for base,spec in [('source_snapshot',m['files']),('dependency_snapshot',m['dependencies'])]:
        for n,h in spec.items():
            if sha(root/base/n)!=h: raise ValueError('Run source/dependency differs: '+n)
def verify_inputs(root,full,budget):
    seal_check(root,budget);spec=read(root/'INPUTS.json')
    expected=read(FOLDER/'inputs.json')
    if spec['pinned_metadata']!=expected or not spec['arrays_verified'] or not spec['no_cache_regeneration']: raise ValueError('Input audit identity differs')
    for n,h in expected.items():
        if sha(root/'pinned'/n)!=h: raise ValueError('Pinned historical metadata changed')
    out={}
    for split,folder,count in [('train','cache_train',8 if spec['smoke'] else 2048),('qual_dev','cache_validation',2 if spec['smoke'] else 100)]:
        rs=read(root/(split+'.json'));old=read(root/'pinned'/(folder+'_identity.json'))['records'][:count]
        if len(rs)!=count: raise ValueError('Input cohort incomplete')
        for a,b in zip(rs,old):
            if {k:v for k,v in a.items() if k!='cache_file'}!={k:v for k,v in b.items() if k!='cache_file'}: raise ValueError('Input original source identity differs')
            if a['official_split']!=split or hashlib.sha256(a['text'].encode()).hexdigest()!=a['text_sha256']: raise ValueError('Source text/split mismatch')
        if full:
            from docci_text_channel_v18.data import Items
            items=Items(rs,1)
            for i in range(len(items)):budget();items[i]
        out[split]=rs
    return out
def stats_check(s,snr,value):
    epsilon=noise(18432,snr,value);pn=float(np.mean(epsilon.astype(np.float64)**2))
    if s['nominal_snr_db']!=snr or s['noise_seed']!=value or s['epsilon_sha256']!=hashlib.sha256(epsilon.tobytes()).hexdigest() or s['post_noise_normalization']: raise ValueError('Noise identity/protocol differs')
    if not close(s['noise_power'],pn) or abs(s['signal_power']-1)>.0002: raise ValueError('Power accounting differs')
    if snr is None:
        if s['realized_snr_db'] is not None or s['observed_sha256']!=s['tx_sha256']: raise ValueError('Clean is not identity')
    elif not close(s['realized_snr_db'],10*math.log10(s['signal_power']/pn)): raise ValueError('Realized SNR differs')
def verify_train(root,rs,c,smoke,full,budget):
    seal_check(root,budget);identity=read(root/'identity.json');arm=identity['arm']
    ids=[r['example_id'] for r in rs];source={r['example_id']:r for r in rs}
    n=2 if smoke else math.ceil(len(rs)/c['accumulation'])*c['epochs']
    if arm not in ARMS or identity['config']!=c or identity['parent_sha256']!=PARENT_SHA or identity['trainable_parameters']!=PARAMETERS or identity['TRAIN_ids']!=ids or identity['updates']!=n or not identity['two_forward_backward_views_per_description']: raise ValueError('Train identity/budget differs')
    rows=lines(root/'history.jsonl');schedule=list(batches(len(rs),c['accumulation'],c['epochs'],c['seed']))[:n]
    if len(rows)!=n: raise ValueError('Partial training not complete')
    for step,(r,(epoch,ix)) in enumerate(zip(rows,schedule)):
        budget();wanted=[ids[i] for i in ix];denom=sum(source[e]['source_tokens']+16 for e in wanted)
        lr=c['learning_rate']*min(1.,(step+1)/min(100,max(1,n//10)))*(.1+.9*(1-step/n))
        if r['update']!=step+1 or r['total_updates']!=n or r['epoch']!=epoch+1 or r['batch_ids']!=wanted or r['valid_tokens_per_view']!=denom or not close(r['lr'],lr): raise ValueError('Schedule/denominator differs')
        if not close(r['loss'],c['clean_weight']*r['clean_CE']+(1-c['clean_weight'])*r['second_CE']) or not math.isfinite(r['gradient_norm']): raise ValueError('Loss/gradient invalid')
        if [x['example_id'] for x in r['observations']]!=wanted: raise ValueError('Missing per-view observations')
        for entry in r['observations']:
            eid=entry['example_id'];channel=None if arm=='clean_continue' else channel_for(c,epoch,eid)
            if len(entry['views'])!=2: raise ValueError('Two matched-compute views required')
            for s,(view,snr,weight) in zip(entry['views'],[('clean',None,.25),('second',channel,.75)]):
                if s['view']!=view or s['weight']!=weight: raise ValueError('Replay view changed')
                stats_check(s,snr,trial_seed(c,'train',epoch,eid,snr))
    learning=read(root/'learning_audit.json');backbone=read(root/'backbone_audit.json');done=read(root/'TRAIN_COMPLETE.json')
    if set(learning['groups'])!={'compressor','receiver'} or not learning['passed'] or not all(v['nonzero_gradient'] and v['changed'] for v in learning['groups'].values()) or sum(v['parameters'] for v in learning['groups'].values())!=PARAMETERS: raise ValueError('TX/RX learning contract failed')
    if not backbone['unchanged'] or backbone['before']!=backbone['after'] or not done['actual_disk_roundtrip'] or not done['fixed_final'] or done['updates']!=n or done['unique_ids_seen']!=len({eid for r in rows for eid in r['batch_ids']}): raise ValueError('Frozen/checkpoint audit differs')
    if full and sha(root/'last.pth')!=done['checkpoint_sha256']: raise ValueError('Final checkpoint changed')
    return dict(kind='train',arm=arm,updates=n,smoke=smoke,arrays_and_weights_verified_here=full,quality_success=False)
def verify_eval(root,rs,c,smoke,budget):
    seal_check(root,budget);identity=read(root/'identity.json');arm=identity['arm']
    ids=[r['example_id'] for r in rs];src={r['example_id']:r for r in rs};schedule=list(trials(c,smoke))
    prefix='smoke_' if smoke else ''
    checkpoint_sha=PARENT_SHA if arm=='frozen' else read(root.parent/(prefix+'train_'+arm)/'TRAIN_COMPLETE.json')['checkpoint_sha256']
    if identity['checkpoint_sha256']!=checkpoint_sha or identity['smoke']!=smoke: raise ValueError('Evaluation checkpoint provenance differs')
    if identity['config']!=c or identity['ids']!=ids or identity['trials']!=[list(x) for x in schedule] or identity['parent_sha256']!=PARENT_SHA or identity['arm'] not in ('frozen',)+ARMS: raise ValueError('Eval identity differs')
    rows=lines(root/'rows.jsonl');expected={(eid,s,r,k) for eid in ids for s,r,k in schedule}
    old={r['example_id']:r for r in lines(root.parent/(prefix+'inputs')/'pinned/released64/validation/rows.jsonl') if r['strategy']=='greedy' and r['control']=='matched'}
    if len(rows)!=len(expected) or {(r['example_id'],r['snr_db'],r['repeat'],r['control']) for r in rows}!=expected: raise ValueError('Incomplete evaluation matrix')
    for r in rows:
        budget();eid=r['example_id'];s=src[eid];snr=r['snr_db'];repeat=r['repeat'];control=r['control']
        if r['official_split']!='qual_dev' or r['arm']!=arm or r['source_text']!=s['text'] or r['source_text_sha256']!=s['text_sha256'] or r['source_tokens']!=s['source_tokens'] or r['input_truncated']: raise ValueError('Evaluation original identity differs')
        stats_check(r['own_stats'],snr,trial_seed(c,'validation',0,eid,snr,repeat))
        donor=ids[(ids.index(eid)+1)%len(ids)] if control=='shuffled' else None
        if r['donor_id']!=donor: raise ValueError('Invalid real donor')
        if control=='shuffled': stats_check(r['delivered_stats'],snr,trial_seed(c,'validation',0,donor,snr,repeat))
        if control=='matched' and r['received_payload_sha256']!=r['own_stats']['observed_sha256']: raise ValueError('Delivered symbols changed')
        if control=='zero' and r['received_payload_sha256']!=hashlib.sha256(np.zeros((1,18432),np.float32).tobytes()).hexdigest(): raise ValueError('Zero payload differs')
        if arm=='frozen' and snr is None and control=='matched' and r['chunk_generated_token_ids']!=old[eid]['chunk_generated_token_ids']: raise ValueError('Original baseline output changed')
        verify_metrics(r)
    if read(root/'EVAL_COMPLETE.json')['rows']!=len(expected): raise ValueError('Eval completion differs')
    return dict(kind='evaluation',arm=arm,rows=len(rows),smoke=smoke,facts='PENDING',quality_success=False)
def aggregate(root,c):
    allrows={a:lines(root/('eval_'+a)/'rows.jsonl') for a in ('frozen',)+ARMS}
    scores={};effects={}
    for snr in (None,)+tuple(c['evaluation_snr_db']):
        key='clean' if snr is None else str(snr);cohorts={}
        scores[key]={}
        for arm,rows in allrows.items():
            matched=[r for r in rows if r['snr_db']==snr and r['control']=='matched']
            byid={}
            for r in matched:byid.setdefault(r['example_id'],[]).append(r)
            # Repeated noise draws are averaged WITHIN description for uncertainty.
            cohorts[arm]=[dict(example_id=eid,normalized_wer=statistics.mean(r['normalized_wer'] for r in rs)) for eid,rs in byid.items()]
            values=np.array([r['normalized_wer'] for r in sorted(cohorts[arm],key=lambda r:r['example_id'])])
            rng=np.random.default_rng(42);samples=rng.integers(0,len(values),(c['bootstrap_draws'],len(values)))
            mean_ci=np.quantile(values[samples].mean(1),[.025,.975]).tolist()
            from docci_text_continue_v11.metrics import words,body
            empty=[(r,j) for r in matched for j in range(math.ceil((r['source_tokens']-2)/62),9)]
            scores[key][arm]=dict(mean_WER=statistics.mean(r['normalized_wer'] for r in cohorts[arm]),mean95CI_WER=mean_ci,descriptions=len(byid),noise_draws=1 if snr is None else c['evaluation_repeats'],generation_rows=len(matched),
                detailed_text_metrics=summary(matched),
                empty_false_activation=sum(bool(body(r['chunk_generated_token_ids'][j])[0]) for r,j in empty),empty_slots=len(empty),
                adjacent_word_repeat=statistics.mean(sum(a==b for a,b in zip(v,v[1:]))/max(1,len(v)-1) for v in [words(r['received_caption']) for r in matched]))
        effects[key]=dict(noise_vs_clean=paired(cohorts['noise_replay'],cohorts['clean_continue'],c['bootstrap_draws']),noise_vs_frozen=paired(cohorts['noise_replay'],cohorts['frozen'],c['bootstrap_draws']))
    gain=effects['0']['noise_vs_clean'];clean=effects['clean']['noise_vs_clean']
    numerical_screen=(gain['gain_pp']>=2 and gain['paired95CI_pp'][0]>0 and clean['paired95CI_pp'][0]>=-1)
    return dict(scores=scores,effects=effects,primary='noise_replay vs clean_continue at 0dB',
        numerical_screen_pass=numerical_screen,
        predeclared_screen='0dB improvement>=2pp and pairedCI lower>0; clean gain CI lower>=-1pp. Screening only, not factual success.',
        secondary='clean regression reported, not silently traded away',facts='PENDING',digital='NOT_RUN',FSO='NOT_RUN',
        image='UNCHANGED_NOT_EVALUATED',full_semantic_success=False,
        uncertainty='description-level paired bootstrap10000; repeats averaged within description; one training seed; reused qual_dev; pointwise unadjusted CIs')

def pipeline(root,full=False,max_minutes=60):
    start=time.monotonic()
    def budget():
        if time.monotonic()-start>max_minutes*60: raise TimeoutError('Bounded verification exceeded')
    source_check(root);state=read(root/'PIPELINE.json');base=read(root/'config.json');results={}
    for prefix,smoke in [('smoke_',True),('',False)]:
        inp=root/(prefix+'inputs')
        if state['stages'].get(prefix+'inputs')!='VERIFIED': continue
        rs=verify_inputs(inp,full,budget);c=load(root/'config.json',smoke)
        for arm in ARMS:
            label=prefix+'train_'+arm
            if state['stages'].get(label)=='VERIFIED':results[label]=verify_train(root/label,rs['train'],c,smoke,full,budget)
        for arm in ('frozen',)+ARMS:
            label=prefix+'eval_'+arm
            if state['stages'].get(label)=='VERIFIED':results[label]=verify_eval(root/label,rs['qual_dev'],c,smoke,budget)
        if all(prefix+'train_'+a in results for a in ARMS):
            a,b=[read(root/(prefix+'train_'+arm)/'identity.json') for arm in ARMS]
            if a['initial_hashes']!=b['initial_hashes']:raise ValueError('Paired training initial weights differ')
    required=['train_'+a for a in ARMS]+['eval_'+a for a in ('frozen',)+ARMS]
    if state['status']=='COMPLETE' and not all(k in results and 'smoke_'+k in results for k in required): raise ValueError('Partial labelled complete')
    if state['status']=='SMOKE_COMPLETE' and not all('smoke_'+k in results for k in required): raise ValueError('Partial labelled smoke complete')
    result=dict(status=state['status'],verified_stages=results,large_arrays_and_weights_rechecked=full,quality_success=False)
    if all('eval_'+a in results for a in ('frozen',)+ARMS):result['comparison']=aggregate(root,base)
    return result

def stage(root,label,full=False):
    source_check(root)
    smoke=label.startswith('smoke_');prefix='smoke_' if smoke else ''
    c=load(root/'config.json',smoke);inp=root/(prefix+'inputs')
    rs=verify_inputs(inp,full if label==prefix+'inputs' else False,lambda:None)
    if label==prefix+'inputs': return dict(kind='inputs',smoke=smoke,arrays_checked_here=full)
    if label.startswith(prefix+'train_'): return verify_train(root/label,rs['train'],c,smoke,full,lambda:None)
    if label.startswith(prefix+'eval_'): return verify_eval(root/label,rs['qual_dev'],c,smoke,lambda:None)
    raise ValueError('Unknown stage label')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--stage');p.add_argument('--full',action='store_true');p.add_argument('--max-minutes',type=float,default=60)
    a=p.parse_args()
    if not math.isfinite(a.max_minutes) or not 0<a.max_minutes<=180:p.error('Finite0..180minutes')
    root=Path(a.root).resolve(strict=True)
    r=stage(root,a.stage,a.full) if a.stage else pipeline(root,a.full,a.max_minutes)
    atomic(root/a.stage/'VERIFIED.json' if a.stage else root/'INDEPENDENT_REPORT.json',r)
    print('Independent CPU report completed; quality not inferred from engineering success',flush=True)
