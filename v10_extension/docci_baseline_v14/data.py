"""Disk-backed immutable FP32 features, bounded RAM; official train/qual_dev only."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from docci_baseline_v14.core import atomic, sha, array_hash
from docci_longclip_joint_v8.core import records, image_index, image_path
from docci_longclip_joint_v8.data import letterbox
from docci_reconstruction_v10.core import chunk_plan

LAYOUT = dict(max_text_tokens=512, chunk_tokens=64)


def view(args, c, task):
    train, val, source = records(args.data_root, c['seed'])
    if source['official_counts'] != dict(train=9647, test=5000, qual_dev=100, qual_test=100):
        raise ValueError('Complete official DOCCI annotations required')
    train, val = train[:c[task+'_train_samples']], val[:c['val_samples']]
    if len(train) != c[task+'_train_samples'] or len(val) != c['val_samples']:
        raise ValueError('Incomplete source matrix')
    return train, val, source


class Dataset:
    def __init__(self, entries, limit):
        self.entries, self.limit, self.cached = entries, limit, OrderedDict()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[x] for x in range(*i.indices(len(self)))]
        if i not in self.cached:
            entry = self.entries[i]
            with np.load(entry['cache_file'], allow_pickle=False) as z:
                meta = json.loads(str(z['meta']))
                values = {k: z[k].copy() for k in z.files if k != 'meta'}
            if sha(entry['cache_file']) != entry['cache_sha256']:
                raise ValueError('Feature cache changed')
            if meta['identity'] != entry['identity'] or set(values) != set(meta['array_hashes']):
                raise ValueError('Feature cache identity/schema differs')
            for k, v in values.items():
                if not np.isfinite(v).all() or array_hash(v) != meta['array_hashes'][k]:
                    raise ValueError('Bad feature array: ' + k)
            self.cached[i] = dict(entry, **values)
            while len(self.cached) > self.limit:
                self.cached.popitem(last=False)
        self.cached.move_to_end(i)
        return self.cached[i]


def cache(args, c, selected, task, backbone, tok, backbone_record, folder, device, budget):
    import torch
    folder.mkdir(parents=True, exist_ok=False)
    entries = []
    index = image_index(args.data_root) if task == 'image' else None
    for i, record in enumerate(selected):
        budget()
        identity = dict(schema='v14-feature', example_id=record['example_id'], task=task,
                        backbone_files=backbone_record['files'], source_split=record['official_split'])
        entry = dict(example_id=record['example_id'], official_split=record['official_split'])
        if task == 'text':
            body = tok(record['text'], add_special_tokens=False, truncation=False,
                       max_length=1000000)['input_ids']
            planned = chunk_plan(body, LAYOUT, tok.bos_token_id, tok.eos_token_id, tok.pad_token_id)
            ids = torch.as_tensor(planned, device=device)
            with torch.no_grad():
                hidden = backbone.model.encoder(input_ids=ids,
                    attention_mask=ids.ne(tok.pad_token_id).long(), return_dict=True).last_hidden_state
            arrays = dict(chunk_ids=planned, chunk_hidden=hidden.float().cpu().numpy())
            identity.update(text_sha256=record['text_sha256'], layout='9x64-full-original-BPE')
            entry.update(text=record['text'], text_sha256=record['text_sha256'], source_tokens=len(body)+2)
        else:
            path = image_path(args.data_root, index, record)
            with Image.open(path) as image:
                native_size = list(image.size)
                canvas, rectangle = letterbox(image, 512)
                rgb = np.asarray(canvas).copy()
            x = torch.as_tensor(rgb, device=device).permute(2, 0, 1)[None].float()/255.
            with torch.no_grad():
                latent = backbone.encode(x*2-1).latent_dist.mean*float(backbone.config.scaling_factor)
            arrays = dict(rgb=rgb, rectangle=np.asarray(rectangle, np.int64), latent=latent[0].float().cpu().numpy())
            identity.update(image_sha256=sha(path), image_size=512,
                            preprocess='native full view letterbox bicubic black v1')
            entry.update(native_size=native_size, image_file=str(path), image_sha256=identity['image_sha256'])
        file = folder/(record['example_id']+'.npz')
        meta = dict(identity=identity, array_hashes={k: array_hash(v) for k, v in arrays.items()})
        with file.open('xb') as f:
            np.savez_compressed(f, meta=np.asarray(json.dumps(meta)), **arrays)
        entry.update(cache_file=str(file), cache_sha256=sha(file), identity=identity)
        entries.append(entry)
        if i == 0 or (i+1) % 50 == 0 or i+1 == len(selected):
            print('CACHE {} {} {}/{}'.format(task, record['official_split'], i+1, len(selected)), flush=True)
    atomic(folder.parent/(folder.name+'_identity.json'), dict(records=entries, RAM_lru_limit=c['cache_lru'],
           opposite_modality_encoded=False, input_truncated=False))
    return Dataset(entries, c['cache_lru'])


def tensors(items, ids, task, device):
    import torch
    keys = ('chunk_ids', 'chunk_hidden') if task == 'text' else ('latent',)
    return {k: torch.as_tensor(np.stack([items[i][k] for i in ids]), device=device) for k in keys}
