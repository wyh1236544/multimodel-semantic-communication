"""Full DOCCI original captions and native-image letterboxes; no COCO fallback."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
from docci_longclip_joint_v8.core import box,digest,atomic_json,safe_write,records,image_index,image_path


def letterbox(image,size):
    image=image.convert('RGB'); rectangle=box(*image.size,size)
    left,top,right,bottom=rectangle
    result=Image.new('RGB',(size,size),(0,0,0))
    method=Image.Resampling.BICUBIC if hasattr(Image,'Resampling') else Image.BICUBIC
    result.paste(image.resize((right-left,bottom-top),method),(left,top))
    return result,rectangle


def token_plan(text, tokenizer, clip_tokenizer, maximum):
    raw=tokenizer(text,add_special_tokens=False,truncation=False,max_length=1000000)['input_ids']
    if len(raw)+2>maximum:
        raise ValueError('BART full description requires {} tokens, configured {}; no silent truncation'.format(len(raw)+2,maximum))
    ids=[tokenizer.bos_token_id]+raw+[tokenizer.eos_token_id]
    length=len(ids); ids += [tokenizer.pad_token_id]*(maximum-length)
    clip_raw=clip_tokenizer(text,add_special_tokens=False,truncation=False,max_length=1000000)['input_ids']
    if not clip_raw or len(clip_raw)>4096: raise ValueError('Empty or >4096 CLIP-token description; no silent filtering')
    blocks=[]
    for pos in range(0,len(clip_raw),246):
        body=clip_raw[pos:pos+246]
        blocks.append(dict(ids=[clip_tokenizer.bos_token_id]+body+[clip_tokenizer.eos_token_id]+[0]*(246-len(body)),
            source_offset=pos,content_tokens=len(body)))
    return dict(ids=ids,length=length,source_tokens=length,clip_tokens=len(clip_raw),clip_blocks=blocks,
        input_truncated=False,longclip_split_policy='contiguous246-content chunks; ordered token features, NOT mean-only source')


def audit(args,c):
    from transformers import BartTokenizer,CLIPTokenizer
    tok=BartTokenizer.from_pretrained(args.bart_path,local_files_only=True)
    ctok=CLIPTokenizer.from_pretrained(args.clip_tokenizer_path,local_files_only=True)
    train,val,receipt=records(args.data_root,c['seed']); index=image_index(args.data_root)
    out=dict(receipt,configured_max_text_tokens=c['max_text_tokens'],image_size=c['image_size'],splits={})
    for split,items in (('train',train),('validation',val)):
        counts=[]; clipcounts=[]; oversized=[]; missing=[]
        for r in items:
            counts.append(len(tok(r['text'],add_special_tokens=False,truncation=False,max_length=1000000)['input_ids'])+2)
            clipcounts.append(len(ctok(r['text'],add_special_tokens=False,truncation=False,max_length=1000000)['input_ids'])+2)
            if counts[-1]>c['max_text_tokens']: oversized.append(dict(example_id=r['example_id'],tokens=counts[-1]))
            try: image_path(args.data_root,index,r)
            except FileNotFoundError: missing.append(r['example_id'])
        out['splits'][split]=dict(samples=len(items),bart_max=max(counts),bart_quantiles=np.quantile(counts,[.5,.9,.95,.99]).tolist(),
            bart_coverage={str(cap):float(np.mean(np.asarray(counts)<=cap)) for cap in (128,256,512,1024)},
            longclip_native248_fraction=float(np.mean(np.asarray(clipcounts)<=248)),longclip_max=max(clipcounts),
            oversized_count=len(oversized),oversized_examples=oversized[:16],missing_images=missing[:16],missing_count=len(missing))
    return out


def cache_view(args,c,items,models,tok,ctok,model_record,device,budget,label):
    import torch
    import torch.nn.functional as F
    bart,vae,clip=models; index=image_index(args.data_root); cached=[]; started=time.monotonic()
    cache_root=safe_write(Path('.cache/v10/docci_longclip_joint_v8')); cache_root.mkdir(parents=True,exist_ok=True)
    receipts=[]
    for position,r in enumerate(items):
        budget(); path=image_path(args.data_root,index,r); plan=token_plan(r['text'],tok,ctok,c['max_text_tokens'])
        with Image.open(path) as im: width,height=im.size
        identity=dict(example_id=r['example_id'],image_sha256=digest(path),text_sha256=r['text_sha256'],image_size=c['image_size'],
            max_text_tokens=c['max_text_tokens'],model_record=model_record,preprocess='native-full-frame-letterbox-bicubic-black-v1',version='docci-cache-v8')
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(); target=cache_root/(key+'.npz')
        if target.is_symlink(): raise ValueError('Symlink cache rejected')
        if not target.exists():
            with Image.open(path) as im:
                canvas,rectangle=letterbox(im,c['image_size']); semantic,_=letterbox(im,224)
                rgb=np.asarray(canvas,dtype=np.uint8).copy()
                pixels=torch.as_tensor(rgb.copy(),device=device).permute(2,0,1)[None].float()/255.
                cp=torch.as_tensor(np.asarray(semantic).copy(),device=device).permute(2,0,1)[None].float()/255.
            ids=torch.tensor([plan['ids']],device=device); mask=ids.ne(tok.pad_token_id)
            with torch.no_grad():
                hidden=bart.model.encoder(input_ids=ids,attention_mask=mask.long(),return_dict=True).last_hidden_state
                latent=vae.encode(pixels*2.-1.).latent_dist.mean*float(vae.config.scaling_factor)
                mean=cp.new_tensor([.48145466,.4578275,.40821073])[None,:,None,None]
                std=cp.new_tensor([.26862954,.26130258,.27577711])[None,:,None,None]
                image_features=F.normalize(clip.encode_image((cp-mean)/std),dim=-1)
                block_features=[]; summaries=[]; weights=[]
                for block in plan['clip_blocks']:
                    token=torch.tensor([block['ids']],device=device)
                    # Keep ALL valid ordered content representations. Padding/BOS/EOS never masquerade as content.
                    block_features.append(clip.encode_text_full(token)[:,1:1+block['content_tokens']].float())
                    summaries.append(F.normalize(clip.encode_text(token).float(),dim=-1)); weights.append(block['content_tokens'])
                clip_tokens=torch.cat(block_features,1)
                summary=F.normalize(sum(x*w for x,w in zip(summaries,weights))/sum(weights),dim=-1)
            arrays=dict(rgb=rgb,rectangle=np.asarray(rectangle,np.int64),ids=np.asarray(plan['ids'],np.int64),
                bart_hidden=hidden[0].cpu().numpy(),latent=latent[0].cpu().numpy(),
                clip_tokens=clip_tokens[0].cpu().numpy(),clip_image=image_features[0].cpu().numpy(),clip_summary=summary[0].cpu().numpy())
            metadata=dict(identity=identity,plan={k:v for k,v in plan.items() if k!='clip_blocks'},native_size=[width,height],
                source_split=r['official_split'],image_path=str(path),array_hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()})
            tmp=target.with_name(target.name+'.part-'+str(os.getpid()))
            with tmp.open('xb') as f: np.savez_compressed(f,meta=np.asarray(json.dumps(metadata)),**arrays)
            os.replace(str(tmp),str(target))
        with np.load(target,allow_pickle=False) as z:
            metadata=json.loads(str(z['meta'])); arrays={k:z[k].copy() for k in metadata['array_hashes']}
        if metadata['identity']!=identity or metadata['source_split']!=r['official_split']: raise ValueError('Cache identity differs')
        for k,v in arrays.items():
            if hashlib.sha256(v.tobytes()).hexdigest()!=metadata['array_hashes'][k] or not np.isfinite(v).all(): raise ValueError('Cache corrupted: '+k)
        if arrays['rgb'].shape!=(c['image_size'],c['image_size'],3) or arrays['bart_hidden'].shape!=(c['max_text_tokens'],bart.config.d_model):
            raise ValueError('Cached image/text size differs')
        if arrays['latent'].shape!=(4,c['image_size']//8,c['image_size']//8) or arrays['clip_tokens'].shape!=(plan['clip_tokens'],512):
            raise ValueError('Cached latent/complete CLIP token coverage differs')
        item=dict(r,**arrays,plan=metadata['plan'],cache_file=str(target),native_size=metadata['native_size'],image_sha256=identity['image_sha256'])
        cached.append(item); receipts.append(dict(example_id=r['example_id'],official_split=r['official_split'],text_sha256=r['text_sha256'],
            image_sha256=identity['image_sha256'],source_tokens=plan['source_tokens'],clip_tokens=plan['clip_tokens'],clip_chunks=len(plan['clip_blocks']),
            cache_sha256=digest(target),native_size=metadata['native_size'],rectangle=arrays['rectangle'].tolist()))
        if position==0 or (position+1)%16==0: print('DOCCI CACHE {} {}/{} elapsed={:.1f}s'.format(label,position+1,len(items),time.monotonic()-started),flush=True)
    return cached,dict(records=receipts,cache_seconds=time.monotonic()-started,source='DOCCI original full captions',
        input_truncation_count=0,spatial_preprocess='full-frame letterbox; no crop/flip',persistent_cache=True,
        loss_eval_rectangles_never_passed_to_receiver=True)


def tensors(item,device):
    import torch
    return {k:torch.as_tensor(item[k],device=device)[None] for k in ('ids','bart_hidden','latent','clip_tokens','clip_image','clip_summary')}
