"""Stdlib contracts: independent DOCCI view, budgets, safe writes and identity."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re

FOLDER = Path(__file__).resolve().parent
PROJECT = FOLDER.parents[1]
LONGCLIP_SHA256 = 'ca2face3716d60473a4e284e5be2cf0a1f3a4ad9954a6a651d878e814cc0c894'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1048576), b''): h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def safe_write(path, root=PROJECT):
    root = Path(root).resolve(strict=True)
    p = Path(path)
    if not p.is_absolute(): p = root/p
    try: relative = p.absolute().relative_to(root)
    except ValueError: raise ValueError('Writes must stay inside the existing project')
    if not relative.parts or any(x in ('.', '..') for x in relative.parts): raise ValueError('Unsafe output path')
    at = root
    for part in relative.parts:
        at = at/part
        if at.is_symlink(): raise ValueError('Symlink write destination rejected: '+str(at))
    return p


def atomic_json(path, value):
    path = safe_write(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with tmp.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


def load(path=None, image_size=None, max_text_tokens=None):
    c = read(path or FOLDER/'config.json'); template = read(FOLDER/'config.json')
    if set(c) != set(template) or c['schema'] != template['schema']: raise ValueError('Configuration keys/schema differ')
    if image_size is not None: c['image_size'] = image_size
    if max_text_tokens is not None: c['max_text_tokens'] = max_text_tokens
    choices = dict(image_size=(256, 512), max_text_tokens=(256, 512, 1024), common_slots=(32,), common_width=(16,),
        text_slots=(32,64), text_width=(64,), image_width=(8,16), hidden=(64,), window=(8,))
    for k, values in choices.items():
        if type(c[k]) is not int or c[k] not in values: raise ValueError('Unsupported '+k)
    bounds = dict(seed=(0,2**31-1),train_samples=(8,2048),val_samples=(2,128),text_updates=(8,10000),image_updates=(8,10000),
        clean_epochs=(1,12),fso_epochs=(1,8),grad_accum=(1,8),save_images=(0,8),proxy_pilots=(16,256))
    for k,(lo,hi) in bounds.items():
        if type(c[k]) is not int or not lo<=c[k]<=hi: raise ValueError('Invalid '+k)
    for k in ('learning_rate','text_learning_rate','joint_text_learning_rate'):
        if type(c[k]) not in (int,float) or not 0<c[k]<=.001: raise ValueError('Unsafe '+k)
    for k in ('pixel_weight','common_weight','clean_anchor_weight','text_wer_gate','memory_margin_gate'):
        if type(c[k]) not in (int,float) or not 0<c[k]<1: raise ValueError('Invalid fraction '+k)
    for k in ('reserve_free_gib','min_free_gib'):
        if type(c[k]) not in (int,float) or not math.isfinite(c[k]): raise ValueError('Finite GPU resources required')
    if not 0<c['reserve_free_gib']<c['min_free_gib']<=48 or c['save_images']>c['val_samples']: raise ValueError('Unsafe resources')
    if set(c['proxy_profiles']) != {'fso_moderate','fso_strong'}: raise ValueError('Fixed proxy profiles')
    for profile in c['proxy_profiles'].values():
        if set(profile)!= {'alpha','beta','xi','a0','loss_db'}: raise ValueError('Invalid proxy fields')
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in profile.values()): raise ValueError('Invalid proxy number')
        if min(profile[k] for k in ('alpha','beta','xi'))<=0 or not 0<profile['a0']<=1 or not 0<=profile['loss_db']<=30:
            raise ValueError('Invalid proxy range')
    if not c['proxy_snr_db'] or len(set(c['proxy_snr_db']))!=len(c['proxy_snr_db']) or any(type(s) not in (int,float) or not -10<=s<=30 for s in c['proxy_snr_db']):
        raise ValueError('Invalid nominal proxy SNR')
    return c


def geometry(c):
    grid = c['image_size']//16  # VAE stride8 then learned spatial stride2
    n = dict(common_reals=c['common_slots']*c['common_width'], image_reals=grid*grid*c['image_width'], text_reals=c['text_slots']*c['text_width'])
    total = sum(n.values())
    return dict(n,image_grid=grid,image_size=c['image_size'],total_reals=total,payload_complex_symbols=total//2,
        analog_repeats=1,raw_utf8_wire_bytes=0,original_length_header_symbols=0,
        receiver_image_shape='fixed public square canvas; original aspect rectangle is evaluation-only',
        fs_hz=64e9,sps=4,symbol_rate_baud=16e9,rolloff=.1,
        continuous_payload_has_no_defined_bit_mapping=True)


def box(width,height,size):
    if min(width,height)<=0 or size not in (224,256,512): raise ValueError('Invalid image dimensions')
    scale = min(size/width,size/height)
    w,h = max(1,round(width*scale)),max(1,round(height*scale))
    left,top = (size-w)//2,(size-h)//2
    return [left,top,left+w,top+h]


def records(root, seed):
    root = Path(root).resolve(strict=True); annotation = root/'docci_descriptions.jsonlines'
    if not annotation.is_file(): raise FileNotFoundError('DOCCI original docci_descriptions.jsonlines missing; no COCO fallback')
    train=[]; validation=[]; seen=set(); declared_counts={}
    with annotation.open(encoding='utf-8') as f:
        for number,line in enumerate(f,1):
            if not line.strip(): continue
            row=json.loads(line); eid=row.get('example_id'); split=row.get('split'); name=row.get('image_file'); text=row.get('description')
            if not isinstance(eid,str) or not re.fullmatch(r'[A-Za-z0-9_\-]+',eid) or eid in seen: raise ValueError('Duplicate/unsafe DOCCI example at line '+str(number))
            seen.add(eid)
            if split not in ('train','test','qual_dev','qual_test'): raise ValueError('Unknown official DOCCI split')
            declared_counts[split]=declared_counts.get(split,0)+1
            # Never open/consume official test/qual_test descriptions or image pixels in this version.
            if split in ('test','qual_test'): continue
            p=Path(name) if isinstance(name,str) else Path('/')
            if p.is_absolute() or any(x=='..' for x in p.parts) or p.suffix.lower() not in ('.jpg','.jpeg','.png'):
                raise ValueError('Unsafe image_file')
            if not isinstance(text,str) or not text.strip(): raise ValueError('Empty original description')
            entry=dict(example_id=eid,official_split=split,image_file=name,text=text,text_sha256=hashlib.sha256(text.encode()).hexdigest())
            (train if split=='train' else validation).append(entry)
    train.sort(key=lambda r:r['example_id']); random.Random(seed).shuffle(train)
    validation.sort(key=lambda r:r['example_id'])
    if not train or len(validation)<2: raise ValueError('Need official train and qual_dev; test is not validation')
    return train,validation,dict(annotation_sha256=digest(annotation),official_counts=declared_counts,
        train_source='official train',validation_source='official qual_dev',test_access='forbidden for training/validation/model choice',seed=seed)


def image_index(root):
    root=Path(root).resolve(strict=True); bases=[root/'images_unpacked',root/'docci_images',root/'images']
    bases=[p for p in bases if p.is_dir()]
    if not bases: raise FileNotFoundError('Extract docci_images.tar.gz into DOCCI/images_unpacked first')
    index={}
    for base in bases:
        for p in base.rglob('*'):
            if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png'):
                resolved=p.resolve(strict=True)
                try: resolved.relative_to(root)
                except ValueError: raise ValueError('DOCCI image link leaves dataset root')
                if p.name in index and index[p.name]!=resolved: raise ValueError('Ambiguous DOCCI basename: '+p.name)
                index[p.name]=resolved
    return index


def image_path(root,index,record):
    p=index.get(Path(record['image_file']).name)
    if p is None: raise FileNotFoundError('DOCCI image missing: '+record['image_file'])
    return p


def jsonlines(path):
    with Path(path).open(encoding='utf-8') as f: return [json.loads(line) for line in f if line.strip()]


def schedule(c,mode):
    if mode not in ('smoke','diagnostic','screen','pipeline'): raise ValueError('Unknown mode')
    if mode=='smoke': return dict(train_samples=8,val_samples=2,text_steps=8,image_steps=8,clean_steps=8,fso_steps=8,quality_gate_required=False)
    if mode=='diagnostic': return dict(train_samples=32,val_samples=8,text_steps=min(c['text_updates'],400),image_steps=min(c['image_updates'],128),
        clean_steps=128,fso_steps=64,quality_gate_required=True)
    return dict(train_samples=c['train_samples'],val_samples=c['val_samples'],text_steps=c['text_updates'],image_steps=c['image_updates'],
        clean_steps=c['train_samples']*c['clean_epochs'],fso_steps=c['train_samples']*c['fso_epochs'],quality_gate_required=True)
