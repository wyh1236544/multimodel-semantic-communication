"""Sequential frozen-model caching; no trainable bridge outputs are cached."""
from collections import OrderedDict
import gc
import json
from pathlib import Path
import numpy as np
from docci_rx_bridge_v16.core import atomic, sha, read, lines, PARENTS, DOCCI_SHA256


def checkpoint(kind, override=None):
    spec = PARENTS[kind]
    from docci_rx_bridge_v16.core import PROJECT
    p = Path(override or PROJECT/spec['path']).resolve(strict=True)
    if sha(p) != spec['sha256']:
        raise ValueError('Actual approved V14 '+kind+' checkpoint SHA differs; no fallback')
    return p


def text_model(args, device):
    import torch
    from docci_modal_v9.backbones import load, historical_backbone
    from docci_baseline_v14.model import TextModel
    bart, tok, identity = load('text', args, device)
    historical_backbone('text', identity['files'])
    p = checkpoint('text', args.text_checkpoint)
    record = torch.load(str(p), map_location='cpu')
    if (record.get('schema'), record.get('task'), record.get('arm'), record.get('updates')) != ('v14-checkpoint', 'text', 'released64', 5120):
        raise ValueError('Expected completed released64 V14 text checkpoint')
    model = TextModel(bart, 'released64').to(device)
    model.restore(record['state'])
    model.requires_grad_(False).eval()
    return model, tok, identity


def patches(visual, rgb):
    """Frozen Long-CLIP final patch states, no source crop/aspect metadata."""
    import torch
    import torch.nn.functional as F
    x = F.interpolate(rgb, (224, 224), mode='bicubic', align_corners=False).clamp(0, 1)
    mean = x.new_tensor([.48145466, .4578275, .40821073])[None, :, None, None]
    std = x.new_tensor([.26862954, .26130258, .27577711])[None, :, None, None]
    x = visual.conv1((x-mean)/std).flatten(2).transpose(1, 2)
    cls = visual.class_embedding.to(x)[None, None].expand(x.shape[0], 1, -1)
    x = visual.ln_pre(torch.cat((cls, x), 1)+visual.positional_embedding.to(x))
    x = visual.transformer(x.transpose(0, 1)).transpose(0, 1)
    result = visual.ln_post(x)[:, 1:]
    if result.shape[1:] != (196, 768):
        raise ValueError('Expected Long-CLIP-B/16 spatial patch states')
    return result


def write_npz(path, arrays):
    if any(not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Nonfinite cache array')
    with path.open('xb') as f:
        np.savez_compressed(f, **arrays)
    return sha(path)


def create(args, c, root, device, budget):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from docci_longclip_joint_v8.core import records, image_index, image_path, LONGCLIP_SHA256
    from docci_longclip_joint_v8.data import letterbox
    from docci_longclip_joint_v8.vendor_longclip import build_model
    from docci_longclip_joint_v8.backbones import signature
    from docci_modal_v9.backbones import load, historical_backbone
    from docci_reconstruction_v10.model import ImageBranch
    from docci_reconstruction_v10.core import chunk_plan
    from docci_baseline_v14.model import decode
    from docci_text_continue_v11.metrics import metrics
    from docci_rx_bridge_v16.model import content_gate
    selected_train, selected_val, source = records(args.data_root, c['seed'])
    if source['official_counts'] != dict(train=9647, test=5000, qual_dev=100, qual_test=100):
        raise ValueError('Expected complete official DOCCI source')
    if source['annotation_sha256'] != DOCCI_SHA256:
        raise ValueError('Original audited DOCCI annotation identity changed')
    selected = selected_train[:c['train_samples']]+selected_val[:c['val_samples']]
    atomic(root/'source_identity.json', source)
    entries = {r['example_id']: dict(example_id=r['example_id'], official_split=r['official_split'],
               text=r['text'], text_sha256=r['text_sha256']) for r in selected}
    paths = {k: checkpoint(k, getattr(args, k+'_checkpoint')) for k in PARENTS}
    atomic(root/'parents.json', {k: dict(path=str(p), sha256=sha(p)) for k, p in paths.items()})
    (root/'text').mkdir()
    (root/'image').mkdir()
    (root/'images').mkdir()
    model, tok, identity = text_model(args, device)
    text_before = signature(model)
    atomic(root/'text_backbone.json', identity)
    baseline = []
    for i, row in enumerate(selected):
        budget()
        body = tok(row['text'], add_special_tokens=False, truncation=False, max_length=1000000)['input_ids']
        chunk_ids = chunk_plan(body, dict(max_text_tokens=512, chunk_tokens=64))
        ids = torch.as_tensor(chunk_ids, device=device)
        with torch.no_grad():
            hidden = model.bart.model.encoder(input_ids=ids, attention_mask=ids.ne(1).long(), return_dict=True).last_hidden_state
            wire = model.encode(ids[None], hidden[None])
            memory = model.receive(wire)['memory']
            gate = content_gate(model.bart, memory)
        if wire.shape != (1, 18432) or abs(float(wire.square().mean())-1) > .0002:
            raise ValueError('Text frame/power contract differs')
        name = 'text/'+row['example_id']+'.npz'
        digest = write_npz(root/name, dict(ids=chunk_ids, memory=memory[0].cpu().numpy(),
                           gate=gate[0].cpu().numpy(), wire=wire[0].cpu().numpy()))
        entry = entries[row['example_id']]
        entry.update(text_file=name, text_file_sha256=digest, source_tokens=len(body)+2)
        if row['official_split'] == 'qual_dev' or i < min(8, c['train_samples']):
            budget()
            result = decode(model.bart, dict(memory=memory, receiver_mask=torch.ones(1, 9, 32, device=device, dtype=torch.bool)), tok, 'greedy')
            baseline.append(dict(example_id=row['example_id'], official_split=row['official_split'],
                                 source_text=row['text'], source_text_sha256=row['text_sha256'],
                                 **result, **metrics(row['text'], result['received_caption'])))
        if i == 0 or (i+1) % 50 == 0 or i+1 == len(selected):
            print('CACHE text {}/{} (frozen V14)'.format(i+1, len(selected)), flush=True)
    if signature(model) != text_before or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('Frozen text pipeline changed')
    atomic(root/'text_frozen_audit.json', dict(before=text_before, after=signature(model), unchanged=True))
    with (root/'baseline.jsonl').open('x', encoding='utf-8') as f:
        for row in baseline:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    del model, hidden, memory, wire, ids, gate
    gc.collect()
    torch.cuda.empty_cache()
    budget()
    vae, _, identity = load('image', args, device)
    historical_backbone('image', identity['files'])
    image = ImageBranch(dict(image_size=512, image_width=8), 'I2_shifted').to(device)
    record = torch.load(str(paths['image']), map_location='cpu')
    if (record.get('schema'), record.get('task'), record.get('arm'), record.get('updates')) != ('v14-checkpoint', 'image', 'I2_shifted', 5120):
        raise ValueError('Expected expanded V14 image checkpoint')
    image.restore(record['state'])
    del record
    image.requires_grad_(False).eval()
    clip_path = Path(args.longclip_path).resolve(strict=True)
    if sha(clip_path) != LONGCLIP_SHA256:
        raise ValueError('Long-CLIP official weight SHA differs; refuse pickle')
    state = torch.load(str(clip_path), map_location='cpu')
    clip = build_model(state, load_from_clip=False).float().to(device).requires_grad_(False).eval()
    del state
    if clip.visual.input_resolution != 224 or clip.visual.conv1.kernel_size != (16, 16):
        raise ValueError('Long-CLIP-B visual geometry differs')
    before = [signature(m) for m in (vae, image, clip)]
    atomic(root/'image_backbone.json', dict(vae=identity, longclip_sha256=LONGCLIP_SHA256,
          preprocessing='full512letterbox; RX RGB resize224 bicubic align_corners=False clamp; CLIP normalization',
          clip_level='final patch states after ln_post; no CLS projection; 196x768'))
    index = image_index(args.data_root)
    captured = []
    def capture(module, inputs, output):
        captured.append(output.detach())
    handle = image.image_decoder.b.register_forward_hook(capture)
    try:
        for i, row in enumerate(selected):
            budget()
            p = image_path(args.data_root, index, row)
            with Image.open(p) as im:
                canvas, rectangle = letterbox(im, 512)
                rgb = torch.as_tensor(np.asarray(canvas).copy(), device=device).permute(2, 0, 1)[None].float()/255.
            with torch.no_grad():
                latent = vae.encode(rgb*2-1).latent_dist.mean*float(vae.config.scaling_factor)
                wire = image.encode(dict(latent=latent))
                captured.clear()
                received = image.receive(wire, torch.ones(1, device=device, dtype=torch.bool), wire.new_full((1,), 40.), wire.new_ones(1))['latent']
                if len(captured) != 1 or captured[0].shape != (1, 96, 32, 32):
                    raise ValueError('I0 actual decoder hook differs')
                i0 = F.adaptive_avg_pool2d(captured[0], (14, 14)).flatten(2).transpose(1, 2)
                restored = ((vae.decode(received/float(vae.config.scaling_factor)).sample+1)/2).clamp(0, 1)
                actual = patches(clip.visual, restored)
                oracle = patches(clip.visual, rgb)
            if wire.shape != (1, 8192) or abs(float(wire.square().mean())-1) > .0002:
                raise ValueError('Image frame/power contract differs')
            name = 'image/'+row['example_id']+'.npz'
            digest = write_npz(root/name, dict(wire=wire[0].cpu().numpy(), rx_i0=i0[0].cpu().numpy(),
                               rx_clip=actual[0].cpu().numpy(), oracle_clip=oracle[0].cpu().numpy()))
            entry = entries[row['example_id']]
            entry.update(image_file=name, image_file_sha256=digest, source_image_sha256=sha(p))
            # Original rectangle is evaluation-only and never passed to an auxiliary.
            if row['official_split'] == 'qual_dev' and len(list((root/'images').glob('*__source.png'))) < c['save_examples']:
                canvas.save(root/'images'/(row['example_id']+'__source.png'))
                out = (restored[0].permute(1, 2, 0).cpu().numpy()*255).round().clip(0, 255).astype(np.uint8)
                Image.fromarray(out).save(root/'images'/(row['example_id']+'__received.png'))
            if i == 0 or (i+1) % 50 == 0 or i+1 == len(selected):
                print('CACHE image {}/{} (actual RX and labelled oracle)'.format(i+1, len(selected)), flush=True)
    finally:
        handle.remove()
    after = [signature(m) for m in (vae, image, clip)]
    if before != after or any(p.grad is not None for m in (vae, image, clip) for p in m.parameters()):
        raise RuntimeError('Frozen image pipeline changed')
    atomic(root/'image_frozen_audit.json', dict(before=before, after=after, unchanged=True))
    for kind, p in paths.items():
        if sha(p) != PARENTS[kind]['sha256']:
            raise ValueError('Historical weight changed during caching')
    if sha(Path(args.data_root)/'docci_descriptions.jsonlines') != source['annotation_sha256']:
        raise ValueError('Source annotations changed during run')
    atomic(root/'index.json', dict(records=list(entries.values()),
           array_storage='float32, no trainable bridge states cached', memory_lru=c['memory_lru'],
           oracle_separate_from_actual=True, source_mask_at_RX=False))


class Dataset:
    def __init__(self, root, split, limit=8):
        self.root, self.limit = Path(root), limit
        self.entries = [r for r in read(self.root/'index.json')['records'] if r['official_split'] == split]
        self.cached = OrderedDict()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        if i not in self.cached:
            entry = self.entries[i]
            value = dict(entry)
            for kind in ('text', 'image'):
                p = self.root/entry[kind+'_file']
                if sha(p) != entry[kind+'_file_sha256']:
                    raise ValueError('Immutable frozen feature cache changed')
                with np.load(p, allow_pickle=False) as z:
                    for key in z.files:
                        value[kind+'_wire' if key == 'wire' else key] = z[key].copy()
            self.cached[i] = value
            while len(self.cached) > self.limit:
                self.cached.popitem(last=False)
        self.cached.move_to_end(i)
        return self.cached[i]


def batch(items, ix, kind, device):
    import torch
    from docci_rx_bridge_v16.model import canonical
    # Materialize each compressed entry ONCE per real batch. Iterating an LRU8
    # independently for memory/IDs/gate/aux would inflate batch64 I/O fourfold.
    values = [items[i] for i in ix]
    memory = torch.as_tensor(np.stack([v['memory'] for v in values]), device=device)
    ids = torch.as_tensor(np.stack([v['ids'] for v in values]), device=device)
    gate = torch.as_tensor(np.stack([v['gate'] for v in values]), device=device)
    raw = memory if kind == 'self' else torch.as_tensor(np.stack([v[kind] for v in values]), device=device)
    return memory, canonical(raw, kind), gate, ids
