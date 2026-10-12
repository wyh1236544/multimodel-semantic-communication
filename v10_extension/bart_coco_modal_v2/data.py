"""Official caption view: image-only never encodes text; text-only never opens images."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import torch
import torch.nn.functional as F
from bart_coco_modal_v2.core import caption_view,tensor_signature


@torch.no_grad()
def cache(args,cfg,mode,old,reference,bart,tokenizer,clip,processor,split,count,device,budget):
    records,receipt=caption_view(args.data_root,args.split_manifest,split,count,cfg['seed'])
    out=[]
    if reference.options['separate'] is not True: raise ValueError('B4 image source must be separate')
    for source in records:
        budget(); item=dict(source,mode=mode)
        if mode!='image_only':
            raw=tokenizer(source['text'],add_special_tokens=False,truncation=False)['input_ids']
            ids=[tokenizer.bos_token_id]+raw[:126]+[tokenizer.eos_token_id]; length=len(ids)
            ids += [tokenizer.pad_token_id]*(128-length)
            ti=torch.tensor([ids],device=device); mask=(torch.arange(128,device=device)[None]<length)
            hidden=bart.model.encoder(input_ids=ti,attention_mask=mask.long(),return_dict=True).last_hidden_state
            ctok=processor.tokenizer; rawclip=ctok(source['text'],add_special_tokens=False,truncation=False)['input_ids']
            chunks=[]
            for pos in range(0,max(1,len(rawclip)),75):
                ci=torch.tensor([[ctok.bos_token_id]+rawclip[pos:pos+75]+[ctok.eos_token_id]],device=device)
                chunks.append(F.normalize(clip.get_text_features(input_ids=ci,attention_mask=torch.ones_like(ci)),dim=-1))
            item.update(input_ids=ti.cpu(),mask=mask.cpu(),bart_hidden=hidden.cpu(),
                clip_text=F.normalize(torch.stack(chunks).mean(0),dim=-1).cpu(),
                covered_text=tokenizer.decode(ids[:length],skip_special_tokens=True),truncated=len(raw)>126,
                source_tokens=len(raw)+2,covered_tokens=length)
        if mode!='text_only':
            part='train2017' if split=='train' else 'val2017'; image_path=Path(args.data_root)/part/source['file_name']
            with Image.open(image_path) as im:
                pil=ImageOps.fit(im.convert('RGB'),(128,128),method=Image.Resampling.BICUBIC)
                image=torch.from_numpy(np.asarray(pil,dtype=np.float32).copy()/255.).permute(2,0,1)[None]
                pixels=processor(images=pil,return_tensors='pt')['pixel_values'].to(device)
            latent=old.base.vae_wrapper.vae.encode(image.to(device)*2.-1.).latent_dist.mean*old.base.vae_wrapper.scaling_factor
            # Exact separate B4 I source ignores dummy TEXT; no caption encoder/embedding here.
            dummy=latent.new_zeros(1,128,old.base.hidden_dim)
            dummy_mask=torch.zeros(1,128,device=device,dtype=torch.bool); dummy_mask[:,0]=True
            wire=reference.source_encode(latent,dummy,dummy_mask)
            item.update(image=image,latent=latent.cpu(),image_wire=wire[:,32:256].cpu(),
                        clip_image=F.normalize(clip.get_image_features(pixel_values=pixels),dim=-1).cpu(),
                        source_rgb8_bytes=128*128*3)
        digest_input=source['text'].encode() if mode=='text_only' else item['image'].numpy().tobytes() if mode=='image_only' else item['image'].numpy().tobytes()+source['text'].encode()
        item['source_sha256']=hashlib.sha256(digest_input).hexdigest()
        item['text_used_for_source']=mode!='image_only'; item['image_used_for_source']=mode!='text_only'
        out.append(item)
        if len(out)==1 or len(out)%32==0: print('COCO CACHE {} {} {}/{}'.format(mode,split,len(out),count),flush=True)
    receipt.update(mode=mode,text_encoder_executed=mode!='image_only',image_file_opened=mode!='text_only',
        image_caption_used_only_for_display=mode=='image_only',source_sha256={str(r['image_id']):r['source_sha256'] for r in out},
        bart_token_counts=[r['source_tokens'] for r in out] if mode!='image_only' else None,
        source_truncation_count=sum(r.get('truncated',False) for r in out))
    return out,receipt


def source_wire(model,item,device):
    def get(key): return item[key].to(device) if key in item else None
    text=model.modality!='image_only'; image=model.modality!='text_only'
    full=model.source_encode(get('bart_hidden') if text else None,get('mask') if text else None,
        get('image_wire') if image else None,get('clip_image') if image else None,get('clip_text') if text else None)
    return model.payload(full)


def source_receipt(item):
    return dict(image_id=item['image_id'],annotation_id=item['annotation_id'],source_sha256=item['source_sha256'],
        split=item['split'],text_used_for_source=item['text_used_for_source'],image_used_for_source=item['image_used_for_source'])
