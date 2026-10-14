"""Pure contracts, historic initialization identity and independent gates."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from docci_longclip_joint_v8.core import safe_write,atomic_json,digest,read,records,image_index,image_path,jsonlines
from docci_longclip_joint_v8.metrics import metrics,summarize

FOLDER=Path(__file__).resolve().parent
PROJECT=FOLDER.parents[1]
DEFAULT_TEXT=PROJECT/'experiment_results/v10_clip_bart_joint_v6_03/v10_screen/text_prefit/clean/last.pth'
DEFAULT_IMAGE=PROJECT/'experiment_results/v10_docci_longclip_joint_v8_root_01/v10_diagnostic/prefit/image/last.pth'
TEXT_SHA='fa739ff95ae3c1e54190524f2ef169235d9cc16eefefceabf670a398d36e3ff7'
IMAGE_SHA='af53e27c71c0fafcb66454a4ed889e5fd487d6ac6314c36c1b960f5fda8d7196'


def load(path=None,image_size=None,max_text_tokens=None):
    template=read(FOLDER/'config.json'); c=read(path or FOLDER/'config.json')
    if set(c)!=set(template) or c['schema']!=template['schema']: raise ValueError('Unknown/missing configuration keys')
    if image_size is not None: c['image_size']=image_size
    if max_text_tokens is not None: c['max_text_tokens']=max_text_tokens
    exact=dict(image_size=(256,512),max_text_tokens=(512,1024),text_slots=(32,),text_width=(64,),depth=(3,),head_dim=(64,),image_width=(8,),hidden=(64,),window=(8,))
    for k,v in exact.items():
        if type(c[k]) is not int or c[k] not in v: raise ValueError('Unsupported '+k)
    bounds=dict(seed=(0,2**31-1),diagnostic_samples=(2,8),train_samples=(8,1024),val_samples=(2,100),text_batch_size=(1,4),image_grad_accum=(1,4),
        text_diagnostic_updates=(8,4000),image_diagnostic_updates=(8,2000),text_clean_updates=(8,10000),image_clean_updates=(8,5000),
        text_continue_updates=(8,3000),image_continue_updates=(8,2000),lr_warmup_updates=(1,500),snapshot_every=(20,1000),save_images=(0,8),proxy_pilots=(16,256))
    for k,(lo,hi) in bounds.items():
        if type(c[k]) is not int or not lo<=c[k]<=hi: raise ValueError('Invalid '+k)
    for k in set(c)-set(exact)-set(bounds)-{'schema','proxy_snr_db','proxy_profiles'}:
        if type(c[k]) not in (float,int) or not math.isfinite(c[k]) or c[k]<=0: raise ValueError('Finite positive '+k)
    if max(c[k] for k in ('text_learning_rate','image_learning_rate','continue_lr'))>.001: raise ValueError('Learning rate cap')
    if not 0<c['reserve_free_gib']<c['min_free_gib']<=48: raise ValueError('GPU resources')
    if c['diagnostic_samples']>c['train_samples'] or c['save_images']>c['val_samples']: raise ValueError('Samples/resources')
    for k in ('text_overfit_wer','text_validation_wer','text_memory_margin','clean_anchor_weight','pixel_weight','memory_consistency_weight','weight_decay'):
        if not 0<c[k]<1: raise ValueError('Invalid fraction '+k)
    if not 5<=c['image_overfit_psnr']<=40 or not 5<=c['image_validation_psnr']<=40 or not .1<=c['image_memory_margin_db']<=10: raise ValueError('Invalid image gate')
    if set(c['proxy_profiles'])!={'fso_moderate','fso_strong'}: raise ValueError('Fixed proxy profiles')
    for p in c['proxy_profiles'].values():
        if set(p)!={'alpha','beta','xi','a0','loss_db'} or any(type(x) not in (float,int) or not math.isfinite(x) for x in p.values()): raise ValueError('Invalid proxy schema')
        if min(p[k] for k in ('alpha','beta','xi'))<=0 or not 0<p['a0']<=1 or not 0<=p['loss_db']<=30: raise ValueError('Invalid proxy values')
    if not c['proxy_snr_db'] or any(type(x) not in (int,float) or not -10<=x<=30 for x in c['proxy_snr_db']) or len(set(c['proxy_snr_db']))!=len(c['proxy_snr_db']): raise ValueError('Invalid nominal proxy SNR')
    return c


def geometry(c,branch):
    if branch not in ('text','image'): raise ValueError('Independent branch required')
    reals=c['text_slots']*c['text_width'] if branch=='text' else (c['image_size']//16)**2*c['image_width']
    return dict(branch=branch,common_reals=0,image_reals=reals if branch=='image' else 0,text_reals=reals if branch=='text' else 0,
        total_reals=reals,payload_complex_symbols=reals//2,analog_repeats=1,raw_utf8_wire_bytes=0,
        original_length_or_aspect_header_symbols=0,fs_hz=64e9,sps=4,symbol_rate_baud=16e9,rolloff=.1,
        bit_rate_and_BER_for_continuous_features_undefined=True,scope='independent modality frame; NOT equal-budget joint/separate comparison')


def initialization(branch,value,custom_sha=None):
    if value=='fresh': return dict(mode='explicit_fresh',checkpoint=None,sha256=None,old_checkpoint_unchanged=True)
    default=DEFAULT_TEXT if branch=='text' else DEFAULT_IMAGE
    path=default if value=='history' else Path(value).resolve(strict=True)
    expected=(TEXT_SHA if branch=='text' else IMAGE_SHA) if value=='history' else custom_sha
    if not isinstance(expected,str) or len(expected)!=64 or any(x not in '0123456789abcdef' for x in expected): raise ValueError('Custom own checkpoint needs explicit --'+branch+'-init-sha256; no arbitrary pickle')
    if not path.is_file(): raise FileNotFoundError('Historical '+branch+' checkpoint missing: '+str(path)+'; no silent fresh fallback')
    done=path.with_name('TRAIN_COMPLETE.json')
    if not done.is_file() or read(done).get('checkpoint_sha256')!=expected or digest(path)!=expected: raise ValueError('Completed own '+branch+' checkpoint identity differs')
    return dict(mode='historical_weights_only',checkpoint=str(path),sha256=expected,preset=value=='history',optimizer_resumed=False,old_checkpoint_unchanged=True)


def gate(rows,branch,c,diagnostic):
    groups={k:[r for r in rows if r['control']==k] for k in ('matched','zero','shuffled')}
    if not all(groups.values()) or len({len(v) for v in groups.values()})!=1: raise ValueError('Complete three-control matrix required')
    if branch=='text':
        summaries={k:summarize(v) for k,v in groups.items()}; matched=summaries['matched']['full_wer']
        margin={k:summaries[k]['full_wer']-matched for k in ('zero','shuffled')}; limit=c['text_overfit_wer'] if diagnostic else c['text_validation_wer']
        eos=summaries['matched']['capacity_stop_fraction']==0
        passed=matched<=limit and min(margin.values())>=c['text_memory_margin'] and eos
        return dict(passed=bool(passed),groups=summaries,matched_wer=matched,margins=margin,natural_eos_all=eos,quality_limit=limit,memory_margin_limit=c['text_memory_margin'])
    summaries={k:dict(psnr=float(np.mean([r['psnr'] for r in v])),ssim=float(np.mean([r['ssim'] for r in v]))) for k,v in groups.items()}
    matched=summaries['matched']['psnr']; margin={k:matched-summaries[k]['psnr'] for k in ('zero','shuffled')}
    limit=c['image_overfit_psnr'] if diagnostic else c['image_validation_psnr']
    return dict(passed=bool(matched>=limit and min(margin.values())>=c['image_memory_margin_db']),groups=summaries,matched_psnr=matched,margins=margin,
        quality_limit=limit,memory_margin_limit=c['image_memory_margin_db'],image_threshold_is_engineering_not_universal_recognizability=True)


def schedule(c,mode,branch):
    if mode not in ('smoke','diagnostic','screen','pipeline'): raise ValueError('Unknown mode')
    if mode=='smoke': return dict(diagnostic_samples=2,train_samples=8,val_samples=2,diagnostic_updates=2,clean_updates=0,continue_updates=0,smoke=True)
    return dict(diagnostic_samples=c['diagnostic_samples'],train_samples=c['train_samples'],val_samples=c['val_samples'],
        diagnostic_updates=c[branch+'_diagnostic_updates'],clean_updates=c[branch+'_clean_updates'] if mode in ('screen','pipeline') else 0,
        continue_updates=c[branch+'_continue_updates'] if mode=='pipeline' else 0,smoke=False)


def state_hash(state):
    return {name:hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for name,value in state.items()}
