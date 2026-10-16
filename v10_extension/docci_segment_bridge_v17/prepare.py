"""Reuse the sealed V16 cache; add TRAIN-only teachers and small VAL observations."""
from pathlib import Path
import shutil
import numpy as np
import torch
from docci_segment_bridge_v17.core import (read, sha, atomic, lines, snapshot, seal, load,
    CACHE_INDEX, eval_conditions, noise_seed)
from docci_segment_bridge_v17.data import Items, Receiver, save_npz
from docci_segment_bridge_v17.model import native_teacher, generate
from docci_longclip_joint_v8.backbones import signature
from docci_text_continue_v11.metrics import metrics


def prepare(args, model, tokenizer, device, budget):
    root, cache = Path(args.output), Path(args.cache).resolve(strict=True)
    smoke, c = args.smoke, load(args.smoke)
    receipt = read(Path(args.cache_receipt))
    if sha(cache/'index.json') != CACHE_INDEX or not receipt['arrays_verified'] or receipt['smoke']:
        raise ValueError('Full approved cache must have independent CPU verification first')
    if receipt['light_manifest_sha256'] != sha(cache/'LIGHT_MANIFEST.json'):
        raise ValueError('Cache receipt does not bind this input')
    snapshot(root, dict(stage='inputs', smoke=smoke, cache=str(cache), cache_index_sha256=CACHE_INDEX,
                        dimensions=model.bart.config.d_model, FSO=False, private=True))
    atomic(root/'config.json', c)
    atomic(root/'cache_receipt.json', receipt)
    from docci_rx_bridge_v16.collect import copy_light
    copy_light(cache, root/'cache_reference', budget)
    before = signature(model)
    train = Items(cache, 'train', c['train_samples'], c['memory_lru'])
    val = Items(cache, 'qual_dev', c['val_samples'], c['memory_lru'])
    dim = model.bart.config.d_model
    teacher = np.zeros((len(train), 10, dim), dtype=np.float32)
    counts = np.zeros((len(train), 9), dtype=np.int64)
    teacher_index = []
    for i in range(len(train)):
        budget()
        row = train[i]
        ids = torch.as_tensor(row['ids'][None], device=device)
        target, n = native_teacher(model.bart, ids)
        teacher[i], counts[i] = target[0].cpu().numpy(), n[0].cpu().numpy()
        teacher_index.append(dict(example_id=row['example_id'], text_sha256=row['text_sha256'],
                                  original_ids_sha256=__import__('hashlib').sha256(row['ids'].tobytes()).hexdigest()))
        if i == 0 or (i+1) % 200 == 0 or i+1 == len(train):
            print('TRAIN native-BART teacher {}/{}; source labels never receiver inputs'.format(i+1, len(train)), flush=True)
    digest = save_npz(root/'teacher.npz', teacher=teacher, counts=counts)
    atomic(root/'TEACHER.json', dict(file='teacher.npz', sha256=digest, index=teacher_index,
           shape=list(teacher.shape), teacher='frozen native BART contextual content-token means; no random target projection',
           paragraph_token_weighted=True, supervision_only=True, validation_teachers=False))
    receiver, observations, baseline = Receiver(model), [], []
    (root/'observations').mkdir()
    (root/'images').mkdir()
    for image in (cache/'images').glob('*.png'):
        shutil.copyfile(image, root/'images'/image.name)
    for channel in eval_conditions(smoke):
        for i in range(len(val)):
            budget()
            row = val[i]
            memory, gate, stats = receiver(torch.as_tensor(row['text_wire'][None], device=device),
                          channel, noise_seed('validation', 0, row['example_id'], channel))
            name = 'observations/{}__{}.npz'.format(row['example_id'], channel)
            digest = save_npz(root/name, memory=memory[0].cpu().numpy(), gate=gate[0].cpu().numpy())
            entry = dict(example_id=row['example_id'], official_split='qual_dev', condition=channel,
                         source_text=row['text'], source_text_sha256=row['text_sha256'], source_tokens=row['source_tokens'],
                         text_cache_sha256=row['text_file_sha256'], image_cache_sha256=row['image_file_sha256'],
                         file=name, sha256=digest, stats=stats)
            observations.append(entry)
            result = generate(model.bart, None, tokenizer, memory, None, gate)
            baseline.append(dict(entry, **result, **metrics(row['text'], result['received_caption'])))
            if channel == 'clean':
                old = next(x for x in lines(cache/'baseline.jsonl') if x['example_id'] == row['example_id'])
                if result['chunk_generated_token_ids'] != old['chunk_generated_token_ids']:
                    raise ValueError('Recomputed clean receiver baseline differs from original V14')
            if i == 0 or (i+1) % 10 == 0 or i+1 == len(val):
                print('VAL observation/native baseline {} {}/{}'.format(channel, i+1, len(val)), flush=True)
    atomic(root/'observations.json', observations)
    with (root/'baseline.jsonl').open('x', encoding='utf-8') as f:
        import json
        for row in baseline:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    after = signature(model)
    if after != before or any(p.grad is not None or p.requires_grad for p in model.parameters()):
        raise RuntimeError('Frozen parent changed during preparation')
    atomic(root/'frozen_audit.json', dict(before=before, after=after, unchanged=True))
    seal(root)


class Prepared:
    def __init__(self, root, device):
        self.root, self.device = Path(root), device
        self.identity = read(self.root/'identity.json')
        self.cache = Path(self.identity['cache'])
        self.c = read(self.root/'config.json')
        self.train = Items(self.cache, 'train', self.c['train_samples'], self.c['memory_lru'])
        self.val = Items(self.cache, 'qual_dev', self.c['val_samples'], self.c['memory_lru'])
        spec = read(self.root/'TEACHER.json')
        if sha(self.root/spec['file']) != spec['sha256']:
            raise ValueError('TRAIN teacher array changed')
        with np.load(self.root/spec['file'], allow_pickle=False) as z:
            self.teacher, self.counts = z['teacher'].copy(), z['counts'].copy()
        self.observations = {(r['example_id'], r['condition']): r for r in read(self.root/'observations.json')}
        self.baseline = {(r['example_id'], r['condition']): r for r in lines(self.root/'baseline.jsonl')}
        self.observed = {}

    def observation(self, i, channel):
        eid = self.val.entries[i]['example_id']
        key = eid, channel
        if key not in self.observed:
            entry = self.observations[key]
            if sha(self.root/entry['file']) != entry['sha256']:
                raise ValueError('Prepared observed memory changed')
            with np.load(self.root/entry['file'], allow_pickle=False) as z:
                self.observed[key] = (torch.as_tensor(z['memory'][None], device=self.device),
                                      torch.as_tensor(z['gate'][None], device=self.device))
        return self.observed[key]
