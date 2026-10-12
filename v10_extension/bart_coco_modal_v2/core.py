"""Immutable image splits, independently hashed official-caption view and slots."""
import json
import hashlib
import math
import random
from pathlib import Path
import numpy as np
from bart_coco_modal_v2 import MODES
from bart_clip_joint_v1.core import digest, seed_for, metrics, summarize, quality_gate, tensor_signature, file_under
from v10.data import validate_manifest

TEMPLATE = Path(__file__).with_name('config.json')
INDICES = dict(image_only=tuple(range(256)), text_only=tuple(range(32))+tuple(range(256,288)), joint=tuple(range(288)))


def load(path=TEMPLATE):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    template = json.loads(TEMPLATE.read_text(encoding='utf-8'))
    if set(c) != set(template) or c['schema'] != template['schema']: raise ValueError('Unknown/missing config')
    if c['caption_policy'] != 'minimum_annotation_id' or not c['default_modes'] or len(set(c['default_modes'])) != len(c['default_modes']) or not set(c['default_modes']) <= set(MODES):
        raise ValueError('Invalid predeclared caption/mode policy')
    fixed = dict(max_tokens=128,common_slots=32,image_slots=224,text_slots=32,width=16,hidden=256,layers=2,heads=4)
    for k,v in fixed.items():
        if type(c[k]) is not int or c[k] != v: raise ValueError('Current architecture fixed: '+k)
    bounds = dict(seed=(0,2**31-1),diagnostic_samples=(2,32),diagnostic_updates=(8,5000),
        feature_warmup_updates=(0,1000),train_samples=(8,10000),val_samples=(2,128),
        clean_epochs=(1,20),grad_accum=(1,8),save_images=(0,8))
    for k,(lo,hi) in bounds.items():
        if type(c[k]) is not int or not lo <= c[k] <= hi: raise ValueError('Invalid integer '+k)
    for k in set(c)-set(fixed)-set(bounds)-{'schema','caption_policy','default_modes'}:
        v=c[k]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0: raise ValueError('Invalid finite positive '+k)
    if c['learning_rate']>.001 or c['diagnostic_samples']>c['train_samples'] or c['feature_warmup_updates']>=c['diagnostic_updates'] or c['save_images']>c['val_samples']:
        raise ValueError('Inconsistent budget')
    if not c['reserve_free_gib']<c['min_free_gib'] or c['forward_tolerance']>.001: raise ValueError('Unsafe resource/consistency gate')
    for k in ('overfit_wer_gate','memory_margin_gate','clean_validation_wer_gate'):
        if not 0<c[k]<1: raise ValueError('Invalid quality gate')
    return c


def indices(mode):
    if mode not in INDICES: raise ValueError('Unknown public mode')
    return INDICES[mode]


def geometry(mode):
    n=len(indices(mode))
    return dict(mode=mode,common_slots=32,image_slots=224 if mode!='text_only' else 0,
        text_slots=32 if mode!='image_only' else 0,slots=n,width=16,reals=n*16,complex_symbols=n*8,
        analog_repeats=1,raw_utf8_wire_bytes=0,mode_public_predeclared=True)


def pack_numpy(wire,mode):
    if np.shape(wire)!=(288,16) or not np.isfinite(wire).all(): raise ValueError('Wrong source wire')
    return np.asarray(wire)[list(indices(mode))].copy()


def expand_numpy(payload,mode):
    if np.shape(payload)!=(len(indices(mode)),16) or not np.isfinite(payload).all(): raise ValueError('Wrong compact payload')
    full=np.zeros((288,16),dtype=np.asarray(payload).dtype); full[list(indices(mode))]=payload
    return full


def caption_view(data_root,manifest_file,split,count,seed):
    if split not in ('train','validation'): raise ValueError('No test/calibration business access')
    manifest=json.loads(Path(manifest_file).read_text()); validate_manifest(manifest)
    ids=list(manifest[split])
    if split=='train': random.Random(seed).shuffle(ids)
    if type(count) is not int or not 0<count<=len(ids): raise ValueError('Insufficient image split')
    ids=ids[:count]; wanted=set(ids); part='train' if split=='train' else 'val'
    path=Path(data_root)/'annotations'/('captions_'+part+'2017.json')
    if not path.is_file(): raise FileNotFoundError('Official COCO captions missing: '+str(path)+'; do not overwrite Localized Narratives')
    raw=path.read_bytes(); data=json.loads(raw.decode('utf-8'))
    if not isinstance(data,dict) or not isinstance(data.get('images'),list) or not isinstance(data.get('annotations'),list): raise ValueError('Official COCO JSON required, not rewritten JSONL')
    files={}; selected={}; annotations=set()
    for row in data['images']:
        iid=int(row['id'])
        if iid not in wanted: continue
        if iid in files: raise ValueError('Duplicate COCO image record')
        name=str(row['file_name'])
        if Path(name).name!=name or name!=str(iid).zfill(12)+'.jpg': raise ValueError('Unexpected COCO2017 filename')
        files[iid]=name
    for row in data['annotations']:
        iid=int(row['image_id'])
        if iid not in wanted: continue
        aid=int(row['id'])
        if aid in annotations: raise ValueError('Duplicate chosen-caption annotation ID')
        annotations.add(aid); text=' '.join(str(row.get('caption','')).strip().split())
        if not text: continue
        if iid not in selected or aid<selected[iid][0]: selected[iid]=(aid,text)
    if set(files)!=wanted or set(selected)!=wanted: raise ValueError('Missing COCO labels for original image split; no silent resampling')
    annotation_sha=hashlib.sha256(raw).hexdigest()
    records=[dict(image_id=i,file_name=files[i],annotation_id=selected[i][0],text=selected[i][1],split=split,
                  source_utf8_bytes=len(selected[i][1].encode()),source_file_sha256=annotation_sha) for i in ids]
    # Compute large source-file hash once, not once per caption.
    sha=records[0]['source_file_sha256']
    receipt=dict(schema='original-coco-caption-view-v2',base_image_split_sha256=digest(manifest_file),
        original_split_not_rewritten=True,annotation_path=str(path),annotation_sha256=sha,split=split,
        policy='minimum_annotation_id, fixed before model evaluation; not best recovery selection',
        image_ids=ids,annotation_ids={str(r['image_id']):r['annotation_id'] for r in records},
        source_utf8_bytes=[r['source_utf8_bytes'] for r in records],
        source_word_counts=[len(r['text'].split()) for r in records])
    return records,receipt
