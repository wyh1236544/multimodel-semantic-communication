"""Full-text framing and explicit rate/loss/initialization contracts (no Torch)."""
import math
from pathlib import Path
import numpy as np
from docci_modal_v9.core import (atomic_json,safe_write,read,digest,jsonlines,
    initialization,state_hash,gate,metrics,summarize)

FOLDER=Path(__file__).resolve().parent
PROJECT=FOLDER.parents[1]
TEXT_ARMS=('T0_whole','T1_chunks')
IMAGE_ARMS=('I0_latent','I1_pixel','I2_shifted')


def load(path=None,image_size=None,image_width=None,text_width=None):
    template=read(FOLDER/'config.json'); c=read(path or FOLDER/'config.json')
    if set(c)!=set(template) or c['schema']!=template['schema']: raise ValueError('Unknown configuration keys/schema')
    for k,v in (('image_size',image_size),('image_width',image_width),('text_width',text_width)):
        if v is not None: c[k]=v
    choices=dict(image_size=(256,512),max_text_tokens=(512,),chunk_tokens=(64,),text_slots=(32,),text_width=(16,32,64),
        depth=(3,),head_dim=(64,),image_width=(8,16,32),hidden=(64,),window=(8,))
    for k,values in choices.items():
        if type(c[k]) is not int or c[k] not in values: raise ValueError('Unsupported '+k)
    bounds=dict(seed=(0,2**31-1),diagnostic_samples=(2,32),train_samples=(32,2048),val_samples=(2,100),text_batch_size=(1,2),
        image_grad_accum=(1,4),text_diagnostic_updates=(8,4000),image_diagnostic_updates=(8,2000),text_clean_updates=(8,10000),
        image_clean_updates=(8,5000),text_continue_updates=(8,3000),image_continue_updates=(8,2000),lr_warmup_updates=(1,500),
        snapshot_every=(20,2000),save_images=(0,8),proxy_pilots=(16,256))
    for k,(lo,hi) in bounds.items():
        if type(c[k]) is not int or not lo<=c[k]<=hi: raise ValueError('Invalid '+k)
    for k in set(c)-set(choices)-set(bounds)-{'schema','proxy_profiles','proxy_snr_db'}:
        if type(c[k]) not in (int,float) or not math.isfinite(c[k]) or c[k]<=0: raise ValueError('Finite positive '+k)
    if max(c[k] for k in ('text_learning_rate','image_learning_rate','continue_lr'))>.001: raise ValueError('LR cap')
    if not 0<c['reserve_free_gib']<c['min_free_gib']<=48 or c['diagnostic_samples']>c['train_samples'] or c['save_images']>c['val_samples']: raise ValueError('Resource bounds')
    for k in ('text_overfit_wer','text_validation_wer','text_memory_margin','pixel_weight','latent_aux_weight','clean_anchor_weight','memory_consistency_weight','weight_decay'):
        if not 0<c[k]<1: raise ValueError('Invalid fraction '+k)
    for k in ('image_overfit_psnr','image_validation_psnr'):
        if not 5<=c[k]<=40: raise ValueError('Image quality gate')
    if not .1<=c['image_memory_margin_db']<=10: raise ValueError('Image memory gate')
    if set(c['proxy_profiles'])!={'fso_moderate','fso_strong'}: raise ValueError('Proxy profiles')
    for p in c['proxy_profiles'].values():
        if set(p)!={'alpha','beta','xi','a0','loss_db'} or any(type(v) not in (int,float) or not math.isfinite(v) for v in p.values()): raise ValueError('Proxy schema')
        if min(p[k] for k in ('alpha','beta','xi'))<=0 or not 0<p['a0']<=1 or not 0<=p['loss_db']<=30: raise ValueError('Proxy values')
    if not c['proxy_snr_db'] or len(set(c['proxy_snr_db']))!=len(c['proxy_snr_db']) or any(type(v) not in (int,float) or not math.isfinite(v) or not -10<=v<=30 for v in c['proxy_snr_db']): raise ValueError('Nominal proxy SNR')
    return c


def chunks(c): return int(math.ceil((c['max_text_tokens']-2)/(c['chunk_tokens']-2)))


def layout(c,arm):
    if arm not in TEXT_ARMS+IMAGE_ARMS: raise ValueError('Unknown arm')
    result=dict(c)
    if arm=='T1_chunks': result['text_slots']=c['text_slots']*chunks(c)
    return result


def geometry(c,arm):
    from docci_modal_v9.core import geometry as old
    branch='text' if arm in TEXT_ARMS else 'image'
    g=old(layout(c,arm),branch)
    g.update(arm=arm,fixed_text_chunks=chunks(c) if arm=='T1_chunks' else 1,
        padded_empty_chunks_also_transmitted=arm=='T1_chunks',source_count_or_length_given_to_RX=False,
        framing='public fixed maximum layout; no variable-length/count header needed',
        scope='independent modality; chunking/rate/architecture costs explicitly NOT equal')
    if branch=='image': g['canvas_complex_symbols_per_RGB_value']=g['payload_complex_symbols']/(3*c['image_size']**2)
    return g


def chunk_plan(body,c,bos=0,eos=2,pad=1):
    if len(body)+2>c['max_text_tokens']: raise ValueError('Full source exceeds capacity; no truncation/filter')
    if any(type(v) is not int or v in (bos,eos,pad) or v<0 for v in body): raise ValueError('Invalid content token')
    width=c['chunk_tokens']; capacity=width-2; count=chunks(c); result=[]
    for i in range(count):
        piece=body[i*capacity:(i+1)*capacity]
        ids=[bos]+piece+[eos]; result.append(ids+[pad]*(width-len(ids)))
    rebuilt=[v for row in result for v in row[1:row.index(eos)]]
    if rebuilt!=body: raise RuntimeError('Chunking dropped or reordered original BPE')
    return np.asarray(result,np.int64)


def joined_ids(generated,bos=0,eos=2,pad=1):
    """RX only: preserve BPE adjacency, do not invent spaces or use source count."""
    joined=[]; capacity=False
    for row in generated:
        body=[]; ended=False
        for v in row[1:]:
            if v==eos: ended=True; break
            if v not in (bos,pad): body.append(int(v))
        joined+=body; capacity=capacity or not ended
    return joined,capacity


def objective(c,arm,latent,pixel):
    if arm=='I0_latent': return latent+c['pixel_weight']*pixel
    if arm not in ('I1_pixel','I2_shifted'): raise ValueError('Image objective arm required')
    return pixel+c['latent_aux_weight']*latent


def choose_indices(count,size,update,micro,seed,accumulation=1):
    from fso_jscc_v1.core import seed_for
    result=[]
    for cursor in range((update*accumulation+micro)*size,(update*accumulation+micro+1)*size):
        order=np.random.default_rng(seed_for(seed,'v9-order',cursor//count)).permutation(count)
        result.append(int(order[cursor%count]))
    return result


def plan(c,mode,branch):
    if mode=='smoke': return dict(train_samples=8,val_samples=2,updates=2,gate_train_samples=2,smoke=True)
    if mode not in ('diagnostic','screen','pipeline'): raise ValueError('Unknown run mode')
    # Unlike V9: diagnostic actually trains ALL64 descriptions/images, not TRAIN4 repeated.
    return dict(train_samples=c['train_samples'],val_samples=c['val_samples'],gate_train_samples=c['diagnostic_samples'],
        updates=c[branch+('_diagnostic_updates' if mode=='diagnostic' else '_clean_updates')],smoke=False)
