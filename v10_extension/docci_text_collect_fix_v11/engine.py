"""CPU-only recovery verifier; only FP32 prefix-mean aggregation is changed."""
import math
from pathlib import Path
import sys
import numpy as np
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT)); sys.path.insert(0,str(EXT.parent))
from docci_text_continue_v11 import report as legacy
from docci_text_continue_v11.core import read,jsonlines,digest
FOLDER=Path(__file__).resolve().parent
PROJECT=FOLDER.parents[1]
POLICY='FP32 nonnegative reduction bound v1; NOT a recovery quality threshold'


def sources():
    m=read(FOLDER/'MANIFEST.json')
    for n,sha in m['files'].items():
        if digest(FOLDER/n)!=sha: raise ValueError('New verifier source differs: '+n)
    for n,sha in m['required_sources'].items():
        if digest(PROJECT/n)!=sha: raise ValueError('Pinned old source differs: '+n)
    return m


def scalar32(value):
    if type(value) not in (int,float) or not math.isfinite(value) or value<0:
        raise ValueError('Diagnostic mean must be finite and nonnegative')
    with np.errstate(over='ignore'):
        rounded=float(np.float32(value))
    if rounded!=value: raise ValueError('Expected original FP32 diagnostic scalar')
    return float(value)


def compare(actual,average,count,max_slot,slots):
    actual=scalar32(actual)
    if not math.isfinite(average) or average<0 or type(count) is not int or not 1<=count<=510 or type(max_slot) is not int or not 1<=max_slot<=min(62,count) or type(slots) is not int or not 1<=slots<=9:
        raise ValueError('Invalid public count/nonnegative reduction')
    def gamma(k,u): return k*u/(1-k*u)
    # Conservative summation + optional prescaling + mean rounding, not a fit to observed errors.
    # Global and per-slot reductions operate on the same nonnegative FP32 token losses/accuracies.
    local=gamma(2*max_slot+4,2.**-24)
    pooled=gamma(2*count+4,2.**-24)
    double=gamma(4*slots+4,2.**-53)
    bound=max(1e-8,(local+pooled+double)*average/(1-local-double))
    error=abs(actual-average)
    if error>bound:
        raise ValueError('FP32 aggregation differs: actual={!r} weighted={!r} error={!r} bound={!r} positions={}'.format(actual,average,error,bound,count))
    return dict(recorded=actual,recomputed_weighted=average,absolute_difference=error,absolute_bound=bound,
        positions=count,max_slot_positions=max_slot,nonempty_slots=slots,
        old_strict_check_passed=legacy.close(actual,average),policy=POLICY)


def aggregation(row,key,weight):
    values=[s for s in row['slots'] if s[weight]]
    if any(type(s[weight]) is not int or not 0<=s[weight]<=62 for s in row['slots']): raise ValueError('Noninteger slot count')
    count=sum(s[weight] for s in values)
    if not count:
        if row[key] is not None: raise ValueError('Nonempty mean for empty content')
        return None
    for s in values: scalar32(s[key])
    average=math.fsum(s[key]*s[weight] for s in values)/count
    result=compare(row[key],average,count,max(s[weight] for s in values),len(values))
    if key=='TF_content_accuracy' and not 0<=row[key]<=1: raise ValueError('Overall accuracy outside probability range')
    return dict(metric=key,**result)

def diagnostic(root,declared,receipts,matched,audit=None):
    rows=jsonlines(root/'rows.jsonl'); done=read(root/'DIAGNOSTIC_COMPLETE.json')
    if [r['example_id'] for r in rows]!=declared['example_ids'] or len(rows)!=done['rows'] or digest(root/'rows.jsonl')!=done['rows_sha256']: raise ValueError('Prefix diagnosis matrix differs')
    for row in rows:
        eid=row['example_id']; source=receipts[eid]
        if row['source_text_sha256']!=source['text_sha256'] or row['source_tokens']!=source['source_tokens'] or row['actual_wire_sha256']!=matched[eid]['tx_wire_sha256'] or not row['diagnostic_only'] or not row['gold_prefix_is_privileged_NOT_actual_RX'] or not row['native_source_memory_NOT_transmitted'] or row['slot_count']!=9 or len(row['slots'])!=9: raise ValueError('Diagnosis treated as real recovery/source mismatch')
        for j,s in enumerate(row['slots']):
            n=min(62,max(0,row['source_tokens']-2-62*j))
            seq=matched[eid]['chunk_generated_token_ids'][j]
            end=next((k for k,t in enumerate(seq[1:],1) if t==2),len(seq)-1)
            common=min(n,max(0,min(64,end)-1))
            if s['slot']!=j or s['gold_content_BPE']!=n or s['common_content_positions']!=common or not 0<=s['TF_argmax_early_EOS']<=n: raise ValueError('Diagnosis content/EOS counts differ')
            for key in ('TF_pEOS_before_gold_end','TF_pEOS_at_gold_end','TF_content_accuracy'):
                v=s[key]
                if v is not None and (not math.isfinite(v) or not 0<=v<=1): raise ValueError('Invalid diagnostic probability')
            for key,expected_count in (('TF_content_CE',n),('native_TF_content_CE',n),('TF_CE_common',common),('free_prefix_CE_common',common)):
                v=s[key]
                if (not expected_count and v is not None) or (expected_count and (v is None or not math.isfinite(v) or v<0)): raise ValueError('Invalid/empty diagnostic CE')
        for key,weight in (('TF_content_CE','gold_content_BPE'),('native_TF_content_CE','gold_content_BPE'),('TF_content_accuracy','gold_content_BPE'),('TF_CE_common','common_content_positions'),('free_prefix_CE_common','common_content_positions')):
            checked=aggregation(row,key,weight)
            if audit is not None and checked is not None:
                audit.append(dict(stage=Path(root).name,example_id=eid,**checked))
        if row['common_content_positions']!=sum(s['common_content_positions'] for s in row['slots']) or not math.isfinite(row['TF_total_CE']) or row['TF_total_CE']<0: raise ValueError('Diagnostic total differs')
    return rows


def verify(root):
    sources()
    if 'torch' in sys.modules: raise RuntimeError('Recovery entry must remain CPU-only and not load Torch/checkpoints')
    original=legacy.diagnostic; strict=legacy.close; audit=[]
    def fixed(folder,declared,receipts,matched):
        return diagnostic(folder,declared,receipts,matched,audit)
    legacy.diagnostic=fixed
    try: result=legacy.verify(root)
    finally:
        legacy.diagnostic=original
        if legacy.close is not strict: raise RuntimeError('Other comparison policy was changed')
    result['aggregation_audit']=audit
    return result
