"""Only the requested modality enters cache/encoder; full original DOCCI view."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
from docci_modal_v9.core import safe_write,digest,records,image_index,image_path
from docci_longclip_joint_v8.data import letterbox


def token_plan(text,tok,maximum):
    body=tok(text,add_special_tokens=False,truncation=False,max_length=1000000)['input_ids']
    if len(body)+2>maximum: raise ValueError('Full DOCCI description requires {} BART tokens, configured{}; no truncation/filter'.format(len(body)+2,maximum))
    ids=[tok.bos_token_id]+body+[tok.eos_token_id]; length=len(ids)
    return dict(ids=ids+[tok.pad_token_id]*(maximum-length),source_tokens=length,input_truncated=False)


def source_view(args,c,mode):
    train,val,receipt=records(args.data_root,c['seed'])
    if receipt['official_counts']!={'train':9647,'test':5000,'qual_dev':100,'qual_test':100}: raise ValueError('Original complete official DOCCI annotations required')
    tr,vr=(8,2) if mode=='smoke' else (c['train_samples'],c['val_samples'])
    if len(train)<tr or len(val)<vr: raise ValueError('Insufficient DOCCI source view')
    train,val=train[:tr],val[:vr]
    if {r['example_id'] for r in train}&{r['example_id'] for r in val}: raise ValueError('DOCCI view overlap')
    return train,val,receipt


def audit(args,c,branch):
    train,val,receipt=records(args.data_root,c['seed']); result=dict(receipt,branch=branch,splits={})
    if branch=='text':
        from transformers import BartTokenizer
        tok=BartTokenizer.from_pretrained(args.bart_path,local_files_only=True)
        for label,items in (('train',train),('validation',val)):
            counts=[len(tok(r['text'],add_special_tokens=False,truncation=False,max_length=1000000)['input_ids'])+2 for r in items]
            result['splits'][label]=dict(samples=len(items),max_bart_tokens=max(counts),configured_max=c['max_text_tokens'],
                oversized=sum(n>c['max_text_tokens'] for n in counts),image_pixels_or_VAE_or_CLIP_accessed=False)
        return result
    index=image_index(args.data_root)
    for label,items in (('train',train),('validation',val)):
        missing=[r['example_id'] for r in items if Path(r['image_file']).name not in index]
        result['splits'][label]=dict(samples=len(items),missing_count=len(missing),missing_examples=missing[:8],image_size=c['image_size'],
            text_tokenizer_or_BART_or_CLIP_accessed=False)
    return result


def cache(args,c,records,branch,backbone,tok,model_record,device,budget,label):
    import torch
    root=safe_write('.cache/v10/docci_modal_v9'); root.mkdir(parents=True,exist_ok=True)
    index=image_index(args.data_root) if branch=='image' else None
    items=[]; receipts=[]; started=time.monotonic()
    for i,r in enumerate(records):
        budget(); identity=dict(schema='docci-modal-v9-cache',branch=branch,example_id=r['example_id'],backbone=model_record['files'])
        if branch=='text':
            plan=token_plan(r['text'],tok,c['max_text_tokens'])
            identity.update(text_sha256=r['text_sha256'],maximum=c['max_text_tokens'])
            path=None
        else:
            path=image_path(args.data_root,index,r)
            identity.update(image_sha256=digest(path),image_size=c['image_size'],preprocess='native full view letterbox bicubic black v1')
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(); file=root/(key+'.npz')
        if file.is_symlink(): raise ValueError('Symlink modality cache rejected')
        if not file.exists():
            if branch=='text':
                ids=torch.tensor([plan['ids']],device=device); mask=ids.ne(tok.pad_token_id)
                with torch.no_grad(): hidden=backbone.model.encoder(input_ids=ids,attention_mask=mask.long(),return_dict=True).last_hidden_state
                arrays=dict(ids=np.asarray(plan['ids'],np.int64),bart_hidden=hidden[0].cpu().numpy())
                metadata=dict(identity=identity,plan={k:v for k,v in plan.items() if k!='ids'})
            else:
                with Image.open(path) as image:
                    native_size=list(image.size); canvas,rectangle=letterbox(image,c['image_size']); rgb=np.asarray(canvas).copy()
                x=torch.as_tensor(rgb.copy(),device=device).permute(2,0,1)[None].float()/255.
                with torch.no_grad(): latent=backbone.encode(x*2-1).latent_dist.mean*float(backbone.config.scaling_factor)
                arrays=dict(rgb=rgb,rectangle=np.asarray(rectangle,np.int64),latent=latent[0].cpu().numpy())
                metadata=dict(identity=identity,native_size=native_size)
            metadata['array_hashes']={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}
            temporary=file.with_name(file.name+'.part-'+str(os.getpid()))
            with temporary.open('xb') as f: np.savez_compressed(f,meta=np.asarray(json.dumps(metadata)),**arrays)
            os.replace(str(temporary),str(file))
        with np.load(file,allow_pickle=False) as z:
            metadata=json.loads(str(z['meta'])); arrays={k:z[k].copy() for k in metadata['array_hashes']}
        if metadata['identity']!=identity: raise ValueError('Cached modality identity differs')
        expected={'ids','bart_hidden'} if branch=='text' else {'rgb','rectangle','latent'}
        if set(arrays)!=expected: raise ValueError('Unexpected opposite modality in source cache')
        for k,v in arrays.items():
            if not np.isfinite(v).all() or hashlib.sha256(v.tobytes()).hexdigest()!=metadata['array_hashes'][k]: raise ValueError('Cache corruption')
        common=dict(example_id=r['example_id'],official_split=r['official_split'],cache_file=str(file),**arrays)
        record=dict(example_id=r['example_id'],official_split=r['official_split'],cache_sha256=digest(file))
        if branch=='text':
            if arrays['ids'].shape!=(c['max_text_tokens'],) or arrays['bart_hidden'].shape!=(c['max_text_tokens'],backbone.config.d_model): raise ValueError('Full text cache shape differs')
            common.update(text=r['text'],text_sha256=r['text_sha256'],**metadata['plan'])
            record.update(text_sha256=r['text_sha256'],source_tokens=metadata['plan']['source_tokens'],input_truncated=False)
        else:
            if arrays['latent'].shape!=(4,c['image_size']//8,c['image_size']//8) or arrays['rgb'].shape!=(c['image_size'],c['image_size'],3): raise ValueError('Native image cache size differs')
            common.update(native_size=metadata['native_size'],image_sha256=identity['image_sha256'],image_file=str(path))
            record.update(image_sha256=identity['image_sha256'],native_size=metadata['native_size'],rectangle=arrays['rectangle'].tolist())
        items.append(common); receipts.append(record)
        if i==0 or (i+1)%16==0: print('DOCCI {} CACHE {} {}/{}'.format(branch,label,i+1,len(records)),flush=True)
    return items,dict(branch=branch,records=receipts,cache_seconds=time.monotonic()-started,opposite_modality_encoded=False)


def batch(items,indices,branch,device):
    import torch
    keys=('ids','bart_hidden') if branch=='text' else ('latent',)
    return {k:torch.as_tensor(np.stack([items[i][k] for i in indices]),device=device) for k in keys}
