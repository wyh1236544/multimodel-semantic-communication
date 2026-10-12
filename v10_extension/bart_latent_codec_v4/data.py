"""Original fixed COCO text, frozen BART cache; no CLIP/VAE/image loading."""
import hashlib
import torch
from bart_latent_codec_v4.core import caption_view


@torch.no_grad()
def cache(args, c, bart, tokenizer, split, count, device, budget):
    records, receipt = caption_view(args.data_root, args.split_manifest, split, count, c['seed'])
    items = []
    for source in records:
        budget()
        raw = tokenizer(source['text'], add_special_tokens=False, truncation=False)['input_ids']
        content = raw[:c['max_tokens']-2]
        ids = [tokenizer.bos_token_id]+content+[tokenizer.eos_token_id]; length = len(ids)
        ids += [tokenizer.pad_token_id]*(c['max_tokens']-length)
        ids = torch.tensor([ids], device=device); mask = ids.ne(tokenizer.pad_token_id)
        hidden = bart.model.encoder(input_ids=ids, attention_mask=mask.long(), return_dict=True).last_hidden_state
        items.append(dict(source, input_ids=ids.cpu(), mask=mask.cpu(), bart_hidden=hidden.cpu(),
            covered_text=tokenizer.decode(ids[0, :length].tolist(), skip_special_tokens=True),
            covered_tokens=length, source_tokens=len(raw)+2, truncated=len(raw)>c['max_tokens']-2,
            source_sha256=hashlib.sha256(source['text'].encode()).hexdigest()))
        if len(items) == 1 or len(items)%64 == 0: print('BART CACHE {} {}/{}'.format(split, len(items), count), flush=True)
    receipt.update(image_pixels_loaded=False, CLIP_VAE_loaded=False,
        truncation_count=sum(v['truncated'] for v in items), token_counts=[v['source_tokens'] for v in items],
        source_sha256={str(v['image_id']): v['source_sha256'] for v in items})
    return items, receipt


def batch(items, indices, device):
    return {k: torch.cat([items[i][k] for i in indices], 0).to(device) for k in ('input_ids', 'mask', 'bart_hidden')}
