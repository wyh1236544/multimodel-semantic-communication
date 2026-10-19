"""Frozen experiment contracts. No optimizer, implicit downloads, or stage promotion."""
import json
import math
from pathlib import Path
import time
import numpy as np
from docci_pretrained_checks_v22.core import (sha, digest, read, rows, safe, local,
    seed, metric, donor_map, case_key, case_name, seal_tree, verify_tree)

FOLDER = Path(__file__).resolve().parent
PROJECT = FOLDER.parents[1]
SCHEMA = 'docci-evidence-v23'
GRID = [None, -8, -5, -2, 0, 2, 5, 10, 20]
CAPTION_SNR = [None, 0, -5]
PROFILES = {'smoke':dict(images=2,repeats=1,snr_db=CAPTION_SNR),
            'quick':dict(images=8,repeats=1,snr_db=GRID),
            'pilot':dict(images=32,repeats=3,snr_db=GRID)}
CONTROLS = ('received_text','caption_only','joint_correct','joint_wrong_image','joint_blank_image','joint_wrong_text')
PROTOCOL = dict(schema=SCHEMA,profiles=PROFILES,caption_snr_db=CAPTION_SNR,
    image_frame_symbols=4128,text_frame_symbols=9248,assisted_frame_symbols=13376,
    transmissions_per_trial=1,caption_max_new_tokens=512,seed=42,
    channel='AWGN',negative_snr='OOD inference, no additional noise training',
    caption_primary='-5 dB fact F1 versus both single modalities and wrong/blank-image controls',
    facts='PENDING',semantic_success=False,training_updates=0)

def atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.writing')
    if path.exists():
        raise FileExistsError('Preserve existing JSON '+str(path))
    with tmp.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
    tmp.rename(path)

def line(f, value):
    f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n'); f.flush()

def source_check():
    m = read(FOLDER/'MANIFEST.json')
    for base, mapping in ((FOLDER,m['files']),(PROJECT,m['dependencies'])):
        for name,h in mapping.items():
            if sha(local(base,name)) != h:
                raise ValueError('Pinned source differs '+name)
    return m

class Budget:
    def __init__(self,hours,gpu=False):
        if not math.isfinite(hours) or not 0<hours<=12:
            raise ValueError('Explicit stage cap in (0,12] hours required')
        self.start=time.monotonic(); self.limit=hours*3600; self.gpu=gpu
    def __call__(self):
        if time.monotonic()-self.start>=self.limit:
            raise TimeoutError('Stage budget exhausted; partial output preserved')
        if self.gpu:
            import torch
            if torch.cuda.mem_get_info()[0]<4*2**30:
                raise RuntimeError('GPU free reserve below 4GiB; no automatic changes')

def noise(family,eid,snr,repeat,kind,n):
    if kind not in ('image','text','image_pilot','text_pilot') or snr not in GRID:
        raise ValueError('Unregistered noise request')
    if family=='v22':
        from docci_pretrained_checks_v22.core import noise as old
        return old(eid,snr,repeat,kind,n)
    if family!='v23':
        raise ValueError('Unknown noise lineage')
    if snr is None:
        return np.zeros(n,np.float32)
    rng=np.random.Generator(np.random.PCG64(seed(42,SCHEMA,eid,snr,repeat,kind)))
    return rng.standard_normal(n).astype(np.float32)*np.float32(10.**(-snr/20.))

def digital_observation(signal,index,row):
    e=noise(index['noise_family'],row['example_id'],row['snr_db'],row['repeat'],'image',8192).reshape(-1,2)
    p=noise(index['noise_family'],row['example_id'],row['snr_db'],row['repeat'],'image_pilot',64).reshape(-1,2)
    return signal+np.concatenate((p[:,0]+1j*p[:,1],e[:,0]+1j*e[:,1]))/math.sqrt(2)

def complete(root,stage,metadata):
    files={}
    for p in sorted(Path(root).rglob('*')):
        if p.is_symlink(): raise ValueError('No symlink artifacts')
        if p.is_file() and p.name not in ('COMPLETE.json','LAUNCH.json'):
            files[p.relative_to(root).as_posix()]=sha(p)
    atomic(Path(root)/'COMPLETE.json',dict(schema=SCHEMA,stage=stage,files=files,
        metadata=metadata,source_manifest_sha256=sha(FOLDER/'MANIFEST.json'),
        training_updates=0,semantic_success=False,facts='PENDING'))

def verify_files(root,stage=None,arrays=True):
    root=Path(root).resolve(strict=True); c=read(root/'COMPLETE.json')
    if (c['schema']!=SCHEMA or (stage and c['stage']!=stage) or c['training_updates']!=0
            or c['source_manifest_sha256']!=sha(FOLDER/'MANIFEST.json')):
        raise ValueError('Wrong completion identity')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name not in ('COMPLETE.json','LAUNCH.json')}
    expected=set(c['files'])
    omitted={n for n in expected if Path(n).suffix=='.npz'} if not arrays else set()
    if actual != expected-omitted:
        raise ValueError('Missing/unexpected artifact matrix')
    for n in actual:
        if sha(local(root,n))!=c['files'][n]: raise ValueError('Changed artifact '+n)
    return c

def prepared(root):
    verify_files(root,'prepare'); index=read(Path(root)/'index.json'); data=rows(Path(root)/'rows.jsonl')
    spec=PROFILES[index['profile']]
    ids=[x['example_id'] for x in index['examples']]
    if (len(ids)!=spec['images'] or len(ids)!=len(set(ids)) or index['spec']!=spec
            or index['noise_family'] not in ('v22','v23') or (index['noise_family']=='v22' and index['profile']!='smoke')):
        raise ValueError('Wrong profile/cohort identity')
    wanted={(eid,s,r) for eid in ids for s in spec['snr_db'] for r in range(spec['repeats'])}
    if len(data)!=len(wanted) or {case_key(r) for r in data}!=wanted:
        raise ValueError('Prepared case matrix differs')
    examples={x['example_id']:x for x in index['examples']}
    for ex in examples.values():
        if ex['cohort']!='dev' or ex['original_split']!='train': raise ValueError('Not the pinned exploratory dev')
        if sha(local(root,ex['source_image']))!=ex['source_image_sha256'] or digest(ex['source_text'].encode())!=ex['source_text_sha256']:
            raise ValueError('Truth bytes differ')
    for r in data:
        if (r['source_text_at_RX'] or r['post_noise_renormalization'] or r['total_symbols']!=13376
                or r['image_payload_symbols']!=4096 or r['text_payload_symbols']!=9216
                or r['donor_id']!=donor_map(ids)[r['example_id']]):
            raise ValueError('Public receiver/resource contract differs')
        if sha(local(root,r['received_image']))!=r['received_image_sha256']:
            raise ValueError('Received PNG bytes differ')
        if r['snr_db'] in CAPTION_SNR:
            if not isinstance(r['received_caption'],str) or digest(r['received_caption'].encode())!=r['received_caption_sha256']:
                raise ValueError('Received caption identity differs')
        elif r['received_caption'] is not None or r['received_caption_sha256'] is not None:
            raise ValueError('Unrequested text inference must be explicit null')
    return index,data,examples

def prompts(text):
    return ('Question: Describe the visible objects, their attributes, counts and spatial relationships. Answer:',
        'Question: Correct this received description using the image. Preserve supported facts and do not invent details. '
        'Received description: '+json.dumps(text,ensure_ascii=False)+' Corrected description: Answer:')

def residual_stats(error):
    a=np.asarray(error,dtype=np.float64)
    if a.shape!=(4,64,64) or not np.isfinite(a).all(): raise ValueError('Expected finite VAE latent error')
    mean=float(a.mean()); sd=float(a.std()); centered=a-mean
    def corr(x,y):
        x=x.ravel()-x.mean();y=y.ravel()-y.mean();den=float(np.linalg.norm(x)*np.linalg.norm(y))
        return None if den==0 else float(np.dot(x,y)/den)
    return dict(mean=mean,MSE=float(np.mean(a*a)),std=sd,
        skew=None if sd==0 else float(np.mean(centered**3)/sd**3),
        excess_kurtosis=None if sd==0 else float(np.mean(centered**4)/sd**4-3),
        adjacent_horizontal_correlation=corr(a[:,:,:-1],a[:,:,1:]),
        adjacent_vertical_correlation=corr(a[:,:-1,:],a[:,1:,:]),
        channel_MSE=[float(np.mean(v*v)) for v in a],
        note='descriptive moments, not a Gaussianity test or a diffusion-timestep mapping')
