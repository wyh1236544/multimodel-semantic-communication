"""Official fixed COCO short captions; no image pixels/old text targets loaded."""
import hashlib
import torch
import torch.nn.functional as F
from text_recovery_v3.core import caption_view


@torch.no_grad()
def cache(args, c, bart, tokenizer, clip, processor, split, count, device, budget):
    records, receipt = caption_view(args.data_root, args.split_manifest, split, count, c['seed'])
    items = []
    for source in records:
        budget()
        raw = tokenizer(source['text'], add_special_tokens=False, truncation=False)['input_ids']
        content = raw[:126]; ids = [tokenizer.bos_token_id]+content+[tokenizer.eos_token_id]
        length = len(ids); ids += [tokenizer.pad_token_id]*(128-length)
        ids = torch.tensor([ids], device=device)
        mask = torch.arange(128, device=device)[None] < length
        hidden = bart.model.encoder(input_ids=ids, attention_mask=mask.long(), return_dict=True).last_hidden_state
        ct = processor.tokenizer; tokens = ct(source['text'], add_special_tokens=False, truncation=False)['input_ids']
        chunks = []
        for start in range(0, max(1, len(tokens)), 75):
            ti = torch.tensor([[ct.bos_token_id]+tokens[start:start+75]+[ct.eos_token_id]], device=device)
            chunks.append(F.normalize(clip.get_text_features(input_ids=ti, attention_mask=torch.ones_like(ti)), dim=-1))
        items.append(dict(source, input_ids=ids.cpu(), mask=mask.cpu(), bart_hidden=hidden.cpu(),
            clip_text=F.normalize(torch.stack(chunks).mean(0), dim=-1).cpu(),
            covered_text=tokenizer.decode(ids[0, :length].tolist(), skip_special_tokens=True),
            truncated=len(raw)>126, source_tokens=len(raw)+2, covered_tokens=length,
            source_sha256=hashlib.sha256(source['text'].encode()).hexdigest()))
        if len(items) == 1 or len(items)%32 == 0: print('TEXT CACHE {} {}/{}'.format(split, len(items), count), flush=True)
    receipt.update(image_pixels_loaded=False, source_truncation_count=sum(v['truncated'] for v in items),
        token_counts=[v['source_tokens'] for v in items], source_sha256={str(v['image_id']):v['source_sha256'] for v in items})
    return items, receipt


def transmit(model, item, device):
    return model.transmit(item['bart_hidden'].to(device), item['mask'].to(device), item['clip_text'].to(device))
