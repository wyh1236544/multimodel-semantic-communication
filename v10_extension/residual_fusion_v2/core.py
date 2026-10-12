"""Fixed-budget decoder-only experiment and explicit diagnostic-stop semantics."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from fso_jscc_v1.core import seed_for, digest
from residual_fusion_v2 import ARMS

TEMPLATE = Path(__file__).with_name('config.json')


def load(path=TEMPLATE):
    c = json.loads(Path(path).read_text())
    template = json.loads(TEMPLATE.read_text())
    if set(c) != set(template) or c['schema'] != template['schema']:
        raise ValueError('Unknown/missing configuration fields')
    for k,lo,hi in [('seed',0,2**31-1),('train_samples',8,10000),('clean_epochs',1,4),
            ('noise_epochs',1,4),('grad_accum',1,8),('val_samples',2,128),('draws',1,3),
            ('save_images',0,8),('diagnostic_samples',2,64),('diagnostic_updates',1,5000),
            ('residual_hidden',16,128),('bootstrap_resamples',100,10000)]:
        if type(c[k]) is not int or not lo <= c[k] <= hi: raise ValueError('Invalid integer '+k)
    for k in set(c)-{'schema'}-{k for k,v in template.items() if type(v) is int}:
        if isinstance(c[k],bool) or not isinstance(c[k],(int,float)) or not math.isfinite(c[k]) or c[k]<=0:
            raise ValueError('Invalid positive finite '+k)
    if c['residual_hidden']%4 or c['diagnostic_samples']>c['train_samples'] or c['save_images']>c['val_samples']:
        raise ValueError('Invalid dimensions/sample coverage')
    if c['drop_fraction']+c['shuffle_fraction']>=.5 or not 0<c['diagnostic_wer_gate']<=.2:
        raise ValueError('Invalid augmentation/diagnostic threshold')
    if max(c['diagnostic_lr'],c['residual_lr'])>.001 or c['arm_max_hours']>4 or c['reserve_free_gib']>=c['min_free_gib']:
        raise ValueError('Invalid learning/resource limits')
    return c


def augmentation(c,phase,epoch,pos):
    u=np.random.default_rng(seed_for(c['seed'],'residual-augmentation',phase,epoch,pos)).random()
    return 'zero' if u<c['drop_fraction'] else 'shuffle' if u<c['drop_fraction']+c['shuffle_fraction'] else 'correct'


def donor_index(index,count):
    if count<2: raise ValueError('A distinct donor is required')
    return (index+1)%count


def diagnostic_decision(metrics,threshold,smoke=False):
    if set(metrics)!= {'D0_precompression','D1_T32_clean'} or any(not math.isfinite(v) or v<0 for v in metrics.values()):
        raise ValueError('Missing/nonfinite actual generation diagnosis')
    observed=all(v<=threshold for v in metrics.values())
    return dict(observed_pass=observed,required=not smoke,proceed=observed or smoke,
                threshold=threshold,actual_covered_wer=metrics,
                smoke_bypasses_quality_gate_only=True if smoke else False,
                diagnostic_not_validation_or_production_text_decoder=True)


def signature(value):
    data=value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str(data.dtype).encode()+json.dumps(list(data.shape)).encode()+data.tobytes()).hexdigest()


def paired_bootstrap(rows,first,second,conditions,metric,resamples,available=False):
    index={(r['arm'],r['image_id'],r['condition'],r['draw']):r for r in rows}
    values={}
    for key,a in index.items():
        arm,i,c,d=key
        if arm!=first or c not in conditions: continue
        b=index[(second,i,c,d)]
        if available and not a['receiver_available']: continue
        values.setdefault(i,[]).append(b[metric]-a[metric])
    x=np.asarray([np.mean(values[i]) for i in sorted(values)],float)
    if not len(x): return dict(mean=None,ci95=None,images=0,trials=0)
    rng=np.random.default_rng(seed_for('residual-image-bootstrap',first,second,conditions,metric,available))
    means=x[rng.integers(len(x),size=(resamples,len(x)))].mean(1)
    return dict(mean=float(x.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),images=len(x),
        trials=sum(map(len,values.values())),confidence_scope='fixed weather/one trained replicate; conditional image clusters',
        exploratory=True,multiplicity_adjusted=False)


def check_receipt_group(rows):
    if {r['arm'] for r in rows}!=set(ARMS) or len(rows)!=len(ARMS): raise ValueError('Incomplete decoder variants')
    fixed=('frame_key','received_sha256','payload_sha256','source_sha256','receiver_available',
           'payload_complex_symbols','analog_repeats','raw_utf8_wire_bytes','received_caption')
    for k in fixed:
        if any(r[k]!=rows[0][k] for r in rows): raise ValueError('Different physical receipt/text '+k)
    if any(r['payload_complex_symbols']!=2304 or r['analog_repeats']!=1 or r['raw_utf8_wire_bytes']!=0 for r in rows):
        raise ValueError('Changed transmission budget/hidden text')
    baseline=next(r for r in rows if r['arm']=='B4_frozen')
    off=next(r for r in rows if r['arm']=='R1_off')
    if off['latent_sha256']!=baseline['latent_sha256']: raise ValueError('Disabled fusion is not exact baseline')


def parameter_counts(input_hidden,hidden):
    return {'query.weight':input_hidden*hidden,'memory.weight':input_hidden*hidden,
        'attention.in_proj_weight':3*hidden*hidden,'attention.out_proj.weight':hidden*hidden,
        'gate.0.weight':64,'gate.0.bias':16,'gate.2.weight':16,'gate.2.bias':1,'output.weight':4*hidden}
