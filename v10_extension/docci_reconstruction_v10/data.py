"""Authentic frozen BART chunk features, full descriptions, no RX clean mask."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from docci_modal_v9.data import cache as original_cache,source_view,audit,batch as old_batch
from docci_reconstruction_v10.core import chunk_plan,chunks,safe_write,digest


def cache(args,c,records,branch,backbone,tok,record,device,budget,label):
    items,receipt=original_cache(args,c,records,branch,backbone,tok,record,device,budget,label)
    if branch!='text': return items,receipt
    import torch
    root=safe_write('.cache/v10/docci_reconstruction_v10'); root.mkdir(parents=True,exist_ok=True)
    for i,(item,r) in enumerate(zip(items,receipt['records'])):
        budget(); ids=item['ids']; body=ids[1:int(item['source_tokens'])-1].tolist()
        planned=chunk_plan(body,c,tok.bos_token_id,tok.eos_token_id,tok.pad_token_id)
        identity=dict(schema='v10-full-BPE-chunks',source_sha256=item['text_sha256'],source_cache_sha256=r['cache_sha256'],
            backbone=record['files'],chunk_tokens=c['chunk_tokens'],fixed_chunks=chunks(c))
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(); file=root/(key+'.npz')
        if file.is_symlink(): raise ValueError('Symlink chunk cache rejected')
        if not file.exists():
            t=torch.as_tensor(planned,device=device)
            with torch.no_grad(): hidden=backbone.model.encoder(input_ids=t,attention_mask=t.ne(tok.pad_token_id).long(),return_dict=True).last_hidden_state.cpu().numpy()
            arrays=dict(chunk_ids=planned,chunk_hidden=hidden)
            meta=dict(identity=identity,array_hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()})
            temporary=file.with_name(file.name+'.part-'+str(os.getpid()))
            with temporary.open('xb') as f: np.savez_compressed(f,meta=np.asarray(json.dumps(meta)),**arrays)
            os.replace(str(temporary),str(file))
        with np.load(file,allow_pickle=False) as z:
            if set(z.files)!={'meta','chunk_ids','chunk_hidden'}: raise ValueError('Wrong chunk cache schema')
            meta=json.loads(str(z['meta'])); arrays={k:z[k].copy() for k in ('chunk_ids','chunk_hidden')}
        if meta['identity']!=identity or not np.array_equal(arrays['chunk_ids'],planned): raise ValueError('Original BPE chunk sequence differs')
        if arrays['chunk_hidden'].shape!=(chunks(c),c['chunk_tokens'],backbone.config.d_model): raise ValueError('Chunk hidden shape differs')
        for k,v in arrays.items():
            if not np.isfinite(v).all() or hashlib.sha256(v.tobytes()).hexdigest()!=meta['array_hashes'][k]: raise ValueError('Chunk cache corrupt')
        item.update(arrays); r.update(chunk_cache_sha256=digest(file),chunk_ids_sha256=meta['array_hashes']['chunk_ids'],
            chunk_hidden_sha256=meta['array_hashes']['chunk_hidden'],fixed_chunks=chunks(c),no_truncation=True)
        if i==0 or (i+1)%16==0: print('BART CHUNK CACHE {} {}/{}; ALL{} public segments retained'.format(label,i+1,len(items),chunks(c)),flush=True)
    return items,receipt


def batch(items,indices,branch,device):
    import torch
    result=old_batch(items,indices,branch,device)
    if branch=='text':
        for k in ('chunk_ids','chunk_hidden'): result[k]=torch.as_tensor(np.stack([items[i][k] for i in indices]),device=device)
    return result
