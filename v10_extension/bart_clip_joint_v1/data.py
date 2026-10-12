"""TRAIN/validation source cache; no cache object is passed to the receiver."""
import hashlib
import torch
import torch.nn.functional as F
from v10.data import build_loader
from v10.runtime import load_config
from joint_jscc_arch_v1.model import encode_text
from bart_clip_joint_v1.core import tensor_signature


@torch.no_grad()
def cache(args, cfg, old, core, bart, tok, clip, processor, split, count, device, budget):
    if split not in ('train', 'validation'): raise ValueError('No test/calibration business access')
    config = load_config(args); config['train'].update(batch_size=1, num_workers=0)
    loader = build_loader(config, args.split_manifest, split, count, cfg['seed'])
    loader.dataset.caption_mode = 'first'
    records = []
    for batch in loader:
        budget(); image = batch['images']; text = batch['texts'][0]; iid = int(batch['image_ids'][0])
        raw = tok(text, add_special_tokens=False, truncation=False)['input_ids']
        ids = [tok.bos_token_id]+raw[:126]+[tok.eos_token_id]
        length = len(ids); ids += [tok.pad_token_id]*(128-length)
        input_ids = torch.tensor([ids], device=device)
        mask = (torch.arange(128, device=device)[None] < length)
        hidden = bart.model.encoder(input_ids=input_ids, attention_mask=mask.long(), return_dict=True).last_hidden_state
        posterior = old.base.vae_wrapper.vae.encode(image.to(device)*2.-1.).latent_dist
        latent = posterior.mean*old.base.vae_wrapper.scaling_factor
        legacy_text = encode_text(old.base, text, 128, device)
        wire = core.source_encode(latent, legacy_text['features'].to(device), legacy_text['mask'].to(device))
        # CLIP image/text share pretrained projected 512-space. Long text is chunked, not silently cut to77.
        from torchvision.transforms.functional import to_pil_image
        pixels = processor(images=to_pil_image(image[0]), return_tensors='pt')['pixel_values'].to(device)
        image_clip = F.normalize(clip.get_image_features(pixel_values=pixels), dim=-1)
        ctok = processor.tokenizer; words = ctok(text, add_special_tokens=False, truncation=False)['input_ids']
        features = []
        for start in range(0, max(1, len(words)), 75):
            part = [ctok.bos_token_id]+words[start:start+75]+[ctok.eos_token_id]
            ti = torch.tensor([part], device=device)
            features.append(F.normalize(clip.get_text_features(input_ids=ti, attention_mask=torch.ones_like(ti)), dim=-1))
        text_clip = F.normalize(torch.stack(features).mean(0), dim=-1)
        source_sha = hashlib.sha256(image.numpy().tobytes()+text.encode()).hexdigest()
        records.append(dict(image_id=iid, text=text, covered_text=tok.decode(ids[:length], skip_special_tokens=True),
            split=split, input_ids=input_ids.cpu(), mask=mask.cpu(), bart_hidden=hidden.cpu(),
            image_wire=wire[:, 32:256].cpu(), image=image, latent=latent.cpu(), clip_image=image_clip.cpu(), clip_text=text_clip.cpu(),
            source_utf8_bytes=len(text.encode()), source_sha256=source_sha, truncated=len(raw)>126,
            source_tokens=len(raw)+2, covered_tokens=length, bart_hidden_sha256=tensor_signature(hidden)))
        if len(records) == 1 or len(records) % 32 == 0: print('CACHE {} {}/{}'.format(split, len(records), count), flush=True)
    if len(records) != count or len({r['image_id'] for r in records}) != count: raise ValueError('Incomplete source cache')
    return records


def source_wire(model, items, device):
    keys = ('bart_hidden', 'mask', 'image_wire', 'clip_image', 'clip_text')
    inputs = [torch.cat([r[k] for r in items], 0).to(device) for k in keys]
    return model.source_encode(*inputs)
