"""Frozen Long-CLIP text ONLY for a privileged, uncharged side-information probe."""
import hashlib
import json
from pathlib import Path
import numpy as np
from docci_semantic_probe_v12.core import PROJECT, safe, digest, atomic_json


def load_longclip(args, device):
    import torch
    from transformers import CLIPTokenizer
    from docci_longclip_joint_v8.vendor_longclip import build_model
    from docci_longclip_joint_v8.core import LONGCLIP_SHA256
    from docci_modal_v9.backbones import file_identity
    weight = Path(args.longclip_path).resolve(strict=True)
    if digest(weight) != LONGCLIP_SHA256: raise ValueError('Official Long-CLIP-B hash differs before loading')
    tok = CLIPTokenizer.from_pretrained(args.clip_tokenizer_path, local_files_only=True)
    if (tok.bos_token_id, tok.eos_token_id, len(tok)) != (49406, 49407, 49408): raise ValueError('Wrong CLIP BPE tokenizer')
    state = torch.load(str(weight), map_location='cpu')
    model = build_model(state, load_from_clip=False).float().to(device).eval().requires_grad_(False)
    del state
    if model.context_length != 248 or tuple(model.text_projection.shape) != (512, 512): raise ValueError('Expected Long-CLIP-B')
    return model, tok, dict(weight_sha256=LONGCLIP_SHA256,tokenizer_files=file_identity(args.clip_tokenizer_path),
        policy='all contiguous246-content-token windows; length-weighted normalized pooled vectors; no truncation',
        scope='ORACLE SOURCE TEXT; NOT transmitted or charged; NOT an end-to-end communication input')


def semantic_cache(records, model, tok, identity, device, budget, label):
    import torch
    import torch.nn.functional as F
    root = safe('.cache/v10/docci_semantic_probe_v12'); root.mkdir(parents=True, exist_ok=True)
    values = {}; receipts = []
    for i, r in enumerate(records):
        budget()
        raw = tok(r['text'], add_special_tokens=False, truncation=False, max_length=1000000)['input_ids']
        if not raw or len(raw) > 4096: raise ValueError('Empty/oversized source description; no filtering or truncation')
        key = dict(schema='v12-oracle-semantic',text_sha256=r['text_sha256'],backbone=identity)
        file = root/(hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()+'.npz')
        if file.is_symlink(): raise ValueError('Symlink semantic cache rejected')
        if not file.exists():
            vectors = []; weights = []
            with torch.no_grad():
                for start in range(0, len(raw), 246):
                    body = raw[start:start+246]
                    ids = [tok.bos_token_id]+body+[tok.eos_token_id]+[0]*(246-len(body))
                    vector = model.encode_text(torch.tensor([ids],device=device)).float()
                    vectors.append(F.normalize(vector,dim=-1)); weights.append(len(body))
                result = F.normalize(sum(v*w for v,w in zip(vectors,weights))/sum(weights),dim=-1)[0].cpu().numpy()
            with file.open('xb') as f: np.savez_compressed(f,vector=result,identity=np.asarray(json.dumps(key,sort_keys=True)))
        with np.load(file,allow_pickle=False) as z:
            if set(z.files) != {'vector','identity'} or json.loads(str(z['identity'])) != key: raise ValueError('Semantic cache identity differs')
            result = z['vector'].copy()
        if result.shape != (512,) or not np.isfinite(result).all() or abs(float(np.linalg.norm(result))-1) > .0001:
            raise ValueError('Invalid complete-description semantic vector')
        values[r['example_id']] = result
        receipts.append(dict(example_id=r['example_id'],official_split=r['official_split'],text_sha256=r['text_sha256'],
            semantic_sha256=hashlib.sha256(result.tobytes()).hexdigest(),cache_sha256=digest(file),content_tokens=len(raw),
            chunks=(len(raw)+245)//246,input_truncated=False,ORACLE_untransmitted=True))
        if i == 0 or (i+1)%64 == 0: print('ORACLE LONGCLIP {} {}/{}'.format(label,i+1,len(records)),flush=True)
    return values, receipts


def image_wires(items, base, device, budget, label):
    import torch
    from docci_reconstruction_v10.data import batch
    for i,item in enumerate(items):
        budget()
        with torch.no_grad(): wire = base.encode(batch(items,[i],'image',device))[0].float().cpu().numpy()
        if wire.shape != (8192,) or not np.isfinite(wire).all() or abs(float(np.mean(wire**2))-1) > .0001:
            raise ValueError('Frozen I2 4096-complex unit-power wire differs')
        item['wire'] = wire
        if i == 0 or (i+1)%64 == 0: print('FROZEN IMAGE TX {} {}/{}'.format(label,i+1,len(items)),flush=True)
    return [dict(example_id=x['example_id'],wire_sha256=hashlib.sha256(x['wire'].tobytes()).hexdigest()) for x in items]
