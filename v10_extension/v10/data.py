from __future__ import annotations
import hashlib
import json
import random
from pathlib import Path
from .paths import write_path

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def annotation_ids(path):
    ids = set()
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if str(item.get('caption', '')).strip():
                    ids.add(int(item['image_id']))
    if not ids:
        raise ValueError('Empty annotations: '+str(path))
    return sorted(ids)

def create_manifest(root, destination, seed=20260920, val_size=500, calibration_size=300,
                    test_size=1000, extra_excluded=()):
    root, destination = Path(root), write_path(destination)
    if destination.exists():
        raise FileExistsError('Split manifest is immutable; use a new filename')
    annotations = root/'annotations/localized_narratives'
    tr, va = annotations/'coco_train_captions.jsonl', annotations/'coco_val_captions.jsonl'
    train, validation = annotation_ids(tr), annotation_ids(va)
    if set(train) & set(validation):
        raise ValueError('Train/val image IDs overlap')
    historic = list(validation)
    random.Random(42).shuffle(historic)
    excluded = set(historic[:500]) | set(map(int, extra_excluded))
    available = [i for i in validation if i not in excluded]
    random.Random(seed).shuffle(available)
    if min(val_size, calibration_size, test_size) <= 0 or val_size+calibration_size+test_size > len(available):
        raise ValueError('Not enough unused validation IDs')
    for split, ids in (('train2017', train), ('val2017', validation)):
        missing = [i for i in ids if not (root/split/('{:012d}.jpg'.format(i))).is_file()]
        if missing:
            raise FileNotFoundError('Missing {} images: {} (total {})'.format(split, missing[:5], len(missing)))
    data = {'schema': 'v10-image-split-v1', 'seed': seed,
            'annotation_sha256': {'train': sha256(tr), 'val': sha256(va)},
            'excluded_history': sorted(excluded),
            'history_scope': 'known seed42 first500 checkpoint-selection IDs plus explicit additions; audit other historical usage separately',
            'train': train, 'validation': available[:val_size],
            'calibration': available[val_size:val_size+calibration_size],
            'test': available[val_size+calibration_size:val_size+calibration_size+test_size]}
    validate_manifest(data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data

def validate_manifest(data):
    groups = ['train', 'validation', 'calibration', 'test', 'excluded_history']
    seen = set()
    for key in groups:
        ids = data[key]
        if len(ids) != len(set(ids)) or (key != 'excluded_history' and not ids):
            raise ValueError('Empty or duplicate split: '+key)
        if seen & set(ids):
            raise ValueError('Split overlap: '+key)
        seen.update(ids)

def build_loader(config, manifest_path, split, limit, seed, workers=None):
    import torch
    from torch.utils.data import DataLoader
    from data.coco_captions import COCOCaptionDataset, _collate
    data = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    validate_manifest(data)
    if split not in ('train', 'validation', 'calibration', 'test'):
        raise ValueError('Invalid split')
    is_train = split == 'train'
    part = 'train' if is_train else 'val'
    path = config['data'][part+'_annotation_file']
    if sha256(path) != data['annotation_sha256'][part]:
        raise ValueError('Annotation hash differs from split manifest')
    ids = list(data[split])
    if is_train:
        random.Random(seed).shuffle(ids)
    if limit is not None:
        if limit <= 0 or limit > len(ids):
            raise ValueError('Sample limit exceeds split or is nonpositive')
        ids = ids[:limit]
    dataset = COCOCaptionDataset(config['data'][part+'_image_root'], path,
                                image_size=int(config['model']['image_size']),
                                train=False, caption_mode='random' if is_train else 'first')
    # Disable the old left/right-unsafe horizontal flip. Keep crop unchanged so
    # channel experiments have one consistent image-transform contract.
    lookup = {r['image_id']: r for r in dataset.records}
    missing = [i for i in ids if i not in lookup]
    if missing:
        raise ValueError('Missing manifest images: '+str(missing[:10]))
    dataset.records = [lookup[i] for i in ids]
    return DataLoader(dataset, batch_size=int(config['train']['batch_size']), shuffle=is_train,
                      num_workers=int(config['train'].get('num_workers', 0) if workers is None else workers),
                      collate_fn=_collate, drop_last=False, pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(seed))
