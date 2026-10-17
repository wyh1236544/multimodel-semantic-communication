"""CPU matrix/identity/scoring verification, no Torch or pickle loading."""
import argparse
import hashlib
import math
from pathlib import Path
import sys
import numpy as np
EXT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_digital_compare_v19.common import read,lines,sha,score,unseal,load,ci,trials,CHECKPOINTS,FOLDER
from docci_digital_compare_v19.inputs import validate_sources

def close(a,b):return math.isfinite(a) and math.isclose(a,b,rel_tol=3e-6,abs_tol=3e-7)
def metric_check(row,source):
    wanted=score(source,row['received_caption'])
    for k in ('normalized_word_errors','normalized_source_words','text_exact'):
        if row[k]!=wanted[k]:raise ValueError('Scoring identity differs: '+k)
    if not close(row['normalized_wer'],wanted['normalized_wer']):raise ValueError('WER differs')
def matrix(rows,expected):
    keys=[(r['arm'],r['example_id'],r['snr_db'],r['repeat']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('Missing/duplicate/extra evaluation cells')
def epsilon_check(row,c):
    from docci_text_channel_v18.core import noise,trial_seed
    s=row['snr_db'];value=trial_seed(c,'validation',0,row['example_id'],s,row['repeat'])
    ep=noise(18432,s,value);h=hashlib.sha256(ep.tobytes()).hexdigest()
    stats=row.get('own_stats',row)
    if row.get('epsilon_sha256',stats['epsilon_sha256'])!=h or stats['epsilon_sha256']!=h:raise ValueError('Noise identity differs')
    pn=float(np.mean(ep.astype(float)**2))
    if not close(stats['noise_power'],pn) or abs(stats['signal_power']-1)>.0002 or stats['post_noise_normalization']:raise ValueError('Power/noise convention differs')
    if s is None:
        if stats['realized_snr_db'] is not None:raise ValueError('Clean is not a numeric SNR')
    elif not close(stats['realized_snr_db'],10*math.log10(stats['signal_power']/pn)):raise ValueError('Measured Es/N0 differs')
    return ep
def stage(root,prepared,c,budget,replay=False):
    root=Path(root);unseal(root);identity=read(root/'identity.json');done=read(root/'COMPLETE.json')
    sources=read(Path(prepared)/'sources.json');sources=sources[:2] if identity['smoke'] else sources
    origin={r['example_id']:r for r in sources};ids=list(origin)
    if identity['config']!=c or identity['ids']!=ids or identity['source_sha256']!=sha(Path(prepared)/'sources.json'):raise ValueError('Stage config/cohort differs')
    digital=identity['kind']=='digital'
    if digital:schedule=list(trials(c,identity['smoke']))
    else:schedule=[(None,0),(0,0),(-5,0)] if identity['smoke'] else list(trials(c,negative=True))
    arms=c['profiles'] if digital else list(CHECKPOINTS)
    expected={(a,e,s,r) for a in arms for e in ids for s,r in schedule}
    rows=lines(root/'rows.jsonl');matrix(rows,expected)
    if identity['trials']!=[list(x) for x in schedule] or done['rows']!=len(rows) or identity['total_rows']!=len(rows) or not done['no_training']:raise ValueError('Stage completion differs')
    anchors={(r['arm'],r['example_id'],r['snr_db'],r['repeat']):r for r in lines(Path(prepared)/'old_rows.jsonl')}
    for row_index,r in enumerate(rows):
        budget();source=origin[r['example_id']]['text'];metric_check(r,source);ep=epsilon_check(r,c)
        if digital:
            from docci_digital_compare_v19.digital import encode,evaluate_packet,allocation
            signal,a,_=encode(source,r['arm'],c)
            if r['allocation']!=a or a['total_symbols']!=9216 or a['pilot_symbols']+a['header_symbols']+a['body_symbols']!=9216:raise ValueError('Resource allocation differs')
            if not 0<=r['pre_fec_errors']<=r['pre_fec_bits'] or r['pre_fec_bits']!=18432:raise ValueError('Raw BER denominator differs')
            if r['post_fec_errors'] is None:
                if r['post_fec_bits']!=0 or r['header_crc_pass']:raise ValueError('Conditional post-FEC BER differs')
            elif not r['header_crc_pass'] or r['post_fec_bits']!=(a['compressed_bytes']+4)*8 or not 0<=r['post_fec_errors']<=r['post_fec_bits']:raise ValueError('Post-FEC BER differs')
            if r['packet_crc_pass']:
                if r['reason']!='ACCEPT' or r['undetected_wrong_packet']!=(r['received_caption']!=source):raise ValueError('Accepted packet status differs')
            elif r['received_caption']!='' or r['reason'] not in ('HEADER_REJECT','BODY_REJECT'):raise ValueError('Failed packet was discarded/repaired')
            if r['snr_db'] is None and (not r['packet_crc_pass'] or not r['text_exact'] or r['pre_fec_errors'] or r['post_fec_errors']):raise ValueError('Clean real digital roundtrip failed')
            if replay:
                actual,_=evaluate_packet(source,r['arm'],c,r['snr_db'],ep)
                for key,val in actual.items():
                    if r[key]!=val:raise ValueError('Deterministic packet replay differs: '+key)
        else:
            if r['arm'] not in CHECKPOINTS or identity['checkpoints']!=CHECKPOINTS:raise ValueError('Wrong continuous checkpoint')
            if len(r['chunk_generated_token_ids'])!=9:raise ValueError('Public frame count changed')
            if identity['smoke'] and r['snr_db'] in (None,0):
                prev=anchors[(r['arm'],r['example_id'],r['snr_db'],r['repeat'])]
                if r['chunk_generated_token_ids']!=prev['chunk_generated_token_ids'] or r['received_caption']!=prev['received_caption']:raise ValueError('V18 real generation anchor differs')
        if replay and (row_index==0 or (row_index+1)%200==0 or row_index+1==len(rows)):
            print('VERIFY/REPLAY {} {}/{} packets'.format(root.name,row_index+1,len(rows)),flush=True)
    if digital and not identity['smoke']:
        from docci_digital_compare_v19.experiment import capacity
        if read(root/'capacity.json')!=capacity(sources,c):raise ValueError('Capacity/compression arithmetic differs')
    if not digital:
        if set(done['audits'])!=set(CHECKPOINTS) or not done['neural_generation_executed']:raise ValueError('Incomplete GPU/model audit')
        for a,v in done['audits'].items():
            if v['checkpoint_sha256']!=CHECKPOINTS[a] or not all(v[k] for k in ('codec_unchanged','BART_unchanged','all_parameters_frozen')) or v['anchor_rows_checked']!=(4 if identity['smoke'] else 0):raise ValueError('Frozen-weight audit differs')
    return dict(kind=identity['kind'],rows=len(rows),smoke=identity['smoke'],
                packet_receiver_replayed_here=bool(digital and replay),neural_inference_here=False)

def aggregate(rows,c):
    groups={}
    for r in rows:groups.setdefault((r['arm'],r['snr_db']),{}).setdefault(r['example_id'],[]).append(r)
    result={};vectors={}
    for (arm,snr),items in groups.items():
        v={eid:float(np.mean([r['normalized_wer'] for r in rs])) for eid,rs in items.items()}
        vectors[(arm,snr)]=v;values=[v[e] for e in sorted(v)];allrows=[r for rs in items.values() for r in rs]
        out=dict(WER=ci(values,c['bootstrap_draws']),descriptions=len(v),rows=len(allrows),
            exact_fraction=float(np.mean([np.mean([r['text_exact'] for r in rs]) for rs in items.values()])),
            noise_draws_per_description=1 if snr is None else c['repeats'])
        if arm in c['profiles']:
            err=sum(r['pre_fec_errors'] for r in allrows);nb=sum(r['pre_fec_bits'] for r in allrows)
            ne=sum(r['post_fec_errors'] or 0 for r in allrows);nd=sum(r['post_fec_bits'] for r in allrows)
            out.update(packet_fail_fraction=float(np.mean([not r['packet_crc_pass'] for r in allrows])),
                header_fail_fraction=float(np.mean([not r['header_crc_pass'] for r in allrows])),
                pre_FEC_BER=err/nb,pre_FEC_error_bits=err,pre_FEC_total_bits=nb,
                post_FEC_BER_conditional_on_header=ne/nd if nd else None,
                post_FEC_error_bits=ne,post_FEC_total_bits=nd,
                header_success_rows=sum(r['header_crc_pass'] for r in allrows),
                undetected_wrong_packets=sum(r['undetected_wrong_packet'] for r in allrows))
        result.setdefault('clean' if snr is None else str(snr),{})[arm]=out
    effects={}
    for profile in c['profiles']:
        for snr in [None]+c['snr_db']:
            if ('noise_replay',snr) not in vectors or (profile,snr) not in vectors:continue
            a,b=vectors[('noise_replay',snr)],vectors[(profile,snr)]
            if set(a)!=set(b):raise ValueError('Unpaired descriptions')
            # Positive means continuous noise_replay has LOWER WER than this practical digital profile.
            effects.setdefault('clean' if snr is None else str(snr),{})[profile]=ci([(b[e]-a[e])*100 for e in sorted(a)],c['bootstrap_draws'])
    return dict(scores=result,continuous_gain_pp_vs_digital=effects,
        uncertainty='3draws averaged within100descriptions; paired bootstrap10000 seed42; pointwise unadjusted; one training seed; reused qual_dev',
        semantic_facts='PENDING',BERTScore='NOT_RUN',full_semantic_success=False,
        universal_continuous_superiority=False,FSO='NOT_RUN',image='NOT_RUN',
        limits=['Best of four stdlib compressors is not an LM compressor or universal best digital baseline.',
                'Whole-paragraph CRC erasure may be worse than packetized/progressive semantic digital reception.',
                'Fixed-code profiles with length-only repetition, not adaptive optimal MCS.',
                'Negative SNR is out of V18 training range; no new training.',
                'Measured CPU simulation time is not link transmission latency.',
                'Zero observed packet errors does not prove zero error probability.'])

def verify(root,budget=lambda:None,replay=False):
    root=Path(root);c=load();m=read(FOLDER/'MANIFEST.json')
    if sha(root/'source_snapshot/MANIFEST.json')!=sha(FOLDER/'MANIFEST.json'):raise ValueError('Different V19 source release')
    for base,spec in (('source_snapshot',m['files']),('dependency_snapshot',m['dependencies'])):
        for n,h in spec.items():
            budget()
            if sha(root/base/n)!=h:raise ValueError('Source snapshot differs: '+n)
    state=read(root/'PIPELINE.json');p=root/'prepared';stages={}
    if not p.exists():return dict(status=state['status'],stages=stages,comparison=None)
    unseal(p);sources=read(p/'sources.json');validate_sources(sources);origin={r['example_id']:r for r in sources}
    old=lines(p/'old_rows.jsonl')
    wanted={(a,e,s,k) for a in CHECKPOINTS for e in origin for s,k in [(None,0)]+[(s,k) for s in (20,15,10,5,0) for k in range(3)]}
    matrix(old,wanted)
    for r in old:budget();metric_check(r,origin[r['example_id']]['text'])
    complete=[]
    for label in ('smoke_digital','smoke_negative','digital','negative'):
        folder=root/label
        if (folder/'COMPLETE.json').exists():
            print('VERIFY '+label,flush=True)
            stages[label]=stage(folder,p,c,budget,replay if label=='digital' else False)
            if label in ('digital','negative'):complete.extend(lines(folder/'rows.jsonl'))
    for label in state['stages']:
        if label=='prepare':continue
        if state['stages'][label] in ('COMPLETE','REUSED') and label not in stages:raise ValueError('Completion without valid stage')
    required={'smoke_digital','digital'} if state['action']=='digital' else {'smoke_digital','digital','smoke_negative','negative'}
    if state['status'] in ('COMPLETE','DIGITAL_COMPLETE') and not required<=set(stages):raise ValueError('Partial labelled complete')
    return dict(status=state['status'],stages=stages,comparison=aggregate(old+complete,c),
                no_new_training=True,large_neural_arrays_rechecked_here=False)

if __name__=='__main__':
    import json
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--replay-digital',action='store_true');a=p.parse_args()
    print(json.dumps(verify(a.root,replay=a.replay_digital),ensure_ascii=False,indent=2,allow_nan=False))
