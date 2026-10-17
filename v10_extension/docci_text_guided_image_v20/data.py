"""Fresh joint cache: frozen source features/V18 wire only, never image TX wire."""
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
from PIL import Image
from docci_text_guided_image_v20.core import (read,sha,atomic,array_sha,split_records,
    IMAGE_SHA,TEXT_SHA,DOCCI_SHA,seed)

PARENT_METADATA = {
    'text':'ae088842a4fa02c5b5a1df7a19e54cdca3e961c59ecc97675406cfb4a28ec142',
    'image':'76eb1cf1a774c2c0c93288cab6f783934b80a213be5b6ce1038e8e15dc643b3a'
}

def parents(args):
    result = {}
    for kind,expected in (('image',IMAGE_SHA),('text',TEXT_SHA)):
        p = Path(getattr(args,kind+'_checkpoint')).resolve(strict=True)
        if sha(p) != expected:
            raise ValueError('Approved '+kind+' parent differs before torch.load')
        result[kind] = dict(path=str(p),sha256=expected)
    return result

def text_model(args, device, bart=None):
    import torch
    from docci_baseline_v14.model import TextModel
    if bart is None:
        from docci_modal_v9.backbones import load,historical_backbone
        bart,_,identity = load('text',args,device)
        historical_backbone('text',identity['files'])
    spec = parents(args)['text']
    ck = torch.load(spec['path'],map_location='cpu')
    if (ck.get('schema'),ck.get('arm'),ck.get('updates')) != ('v18-text-checkpoint','noise_replay',2560):
        raise ValueError('Expected completed V18 noise_replay codec')
    model = TextModel(bart,'released64').to(device)
    model.restore(ck['state'])
    return model.requires_grad_(False).eval()

def selection(args,c,budget,smoke):
    from docci_longclip_joint_v8.core import records,image_index,image_path
    train,val,source = records(args.data_root,c['seed'])
    if source['annotation_sha256'] != DOCCI_SHA:
        raise ValueError('Audited DOCCI annotation differs')
    used = set()
    used_images = set()
    used_text = set()
    receipts = {}
    for kind in ('image','text'):
        p = Path(getattr(args,kind+'_parent_metadata'))
        if sha(p) != PARENT_METADATA[kind]:
            raise ValueError('Approved parent cohort metadata differs: '+kind)
        old = read(p)['records']
        if len(old) != 2048 or any(r['official_split']!='train' for r in old):
            raise ValueError('Wrong parent TRAIN cohort')
        used.update(r['example_id'] for r in old)
        used_images.update(r['image_sha256'] for r in old if 'image_sha256' in r)
        used_text.update(r['text_sha256'] for r in old if 'text_sha256' in r)
        receipts[kind] = dict(path=str(p.resolve()),sha256=sha(p),count=len(old))
    index = image_index(args.data_root)
    for i,r in enumerate(train+val):
        budget()
        r['source_image_path'] = str(image_path(args.data_root,index,r))
        r['image_sha256'] = sha(r['source_image_path'])
        if r['image_sha256'] in used_images or r['text_sha256'] in used_text:
            used.add(r['example_id'])
        if i==0 or (i+1)%500==0:
            print('SOURCE image-byte audit {}/9747'.format(i+1),flush=True)
    selected = split_records(train,val,used,c,smoke)
    # Do not filter difficult or duplicate evaluation samples after looking at
    # results. An exact-byte split leak stops preparation and must be reviewed.
    groups = {k:{r['image_sha256'] for r in selected[k]} for k in ('train','dev','qual_dev')}
    texts = {k:{r['text_sha256'] for r in selected[k]} for k in groups}
    for a,b in (('train','dev'),('train','qual_dev'),('dev','qual_dev')):
        if groups[a]&groups[b] or texts[a]&texts[b]:
            raise ValueError('Exact image/text duplicate crosses '+a+'/'+b)
    source.update(parent_cohort_receipts=receipts,
        development='500 fixed official TRAIN candidates unseen by known V14/V18 parents; first100 evaluated',
        prior_V16_V17_bridges_not_reused=True,
        historical_pretraining_exposure_unknown=True,qual_dev_reused=True)
    return selected,source

def create(args,c,root,device,budget,smoke):
    import torch
    import torch.nn.functional as F
    from docci_longclip_joint_v8.backbones import load_models,signature,verify_frozen
    from docci_longclip_joint_v8.data import letterbox,token_plan
    from docci_reconstruction_v10.core import chunk_plan
    selected,source = selection(args,c,budget,smoke)
    root.mkdir(exist_ok=False)
    atomic(root/'source_identity.json',source)
    atomic(root/'parents.json',parents(args))
    atomic(root/'protected_split.json',{k:[r['example_id'] for r in v] for k,v in selected.items()})
    models,tok,ctok,identity = load_models(args,device)
    bart,vae,clip = models
    from docci_modal_v9.backbones import historical_backbone
    historical_backbone('text',identity['bart_files'])
    historical_backbone('image',identity['vae_files'])
    text = text_model(args,device,bart)
    before = [signature(m) for m in (text,vae,clip)]
    atomic(root/'backbones.json',identity)
    entries = []
    (root/'arrays').mkdir()
    for group in ('train','dev','qual_dev'):
        for i,r in enumerate(selected[group]):
            budget()
            plan = token_plan(r['text'],tok,ctok,512)
            body = tok(r['text'],add_special_tokens=False,truncation=False,max_length=1000000)['input_ids']
            ids = chunk_plan(body,dict(max_text_tokens=512,chunk_tokens=64))
            ti = torch.as_tensor(ids,device=device)
            with Image.open(r['source_image_path']) as im:
                canvas,rect = letterbox(im,512)
                rgb = np.asarray(canvas).copy()
            if min(rect[2]-rect[0],rect[3]-rect[1]) < 32:
                raise ValueError('Native valid view too narrow for LPIPS; no silent resize/filter')
            pixels = torch.as_tensor(rgb,device=device).permute(2,0,1)[None].float()/255.
            with torch.no_grad():
                latent = vae.encode(pixels*2-1).latent_dist.mean*float(vae.config.scaling_factor)
                hidden = bart.model.encoder(input_ids=ti,attention_mask=ti.ne(1).long(),
                                            return_dict=True).last_hidden_state
                wire = text.encode(ti[None],hidden[None])
                all_tokens = []
                for block in plan['clip_blocks']:
                    tokens = torch.as_tensor([block['ids']],device=device)
                    all_tokens.append(clip.encode_text_full(tokens)[:,1:1+block['content_tokens']])
                semantic = F.adaptive_avg_pool1d(torch.cat(all_tokens,1).transpose(1,2),32).transpose(1,2)
                # Explicit privileged source-memory probe only. Never used during
                # actual-RX training or matched evaluation.
                oracle = F.adaptive_avg_pool1d((hidden*ti.ne(1)[...,None]).transpose(1,2),32).transpose(1,2)
            arrays = dict(rgb=rgb,rectangle=np.asarray(rect,np.int64),
                latent=latent[0].float().cpu().numpy(),source_semantics=semantic[0].float().cpu().numpy(),
                text_wire=wire[0].float().cpu().numpy(),ids=ids,
                oracle_memory=oracle.float().cpu().numpy().reshape(288,768))
            file = root/'arrays'/(r['example_id']+'.npz')
            with file.open('xb') as f:
                np.savez(f,**arrays)
            entry = dict(r,cohort=group,cache_file=file.relative_to(root).as_posix(),cache_sha256=sha(file),
                array_hashes={k:array_sha(v) for k,v in arrays.items()},source_tokens=plan['source_tokens'],
                longclip_tokens=plan['clip_tokens'],input_truncated=False)
            entries.append(entry)
            if i==0 or (i+1)%50==0 or i+1==len(selected[group]):
                print('V20 CACHE {} {}/{} frozen source/V18 text only'.format(group,i+1,len(selected[group])),flush=True)
    verify_frozen((text,vae,clip),before)
    atomic(root/'frozen_audit.json',dict(signatures=before,unchanged=True,no_parameter_gradients=True))
    atomic(root/'index.json',dict(records=entries,smoke=smoke,
        image_codec_outputs_cached=False,V18_text_wire_frozen=True,
        original_rectangle_evaluation_only=True,geometry=dict(image=[4,64,64],source_semantics=[32,512],
        text_wire=[18432],oracle_memory=[288,768])))
    atomic(root/'CACHE_COMPLETE.json',dict(index_sha256=sha(root/'index.json'),
        counts={k:len(selected[k]) for k in ('train','dev','qual_dev')},
        source_sha256=DOCCI_SHA,parents=parents(args)))

class Items:
    def __init__(self,root,cohort):
        self.root = Path(root)
        self.entries = [r for r in read(self.root/'index.json')['records'] if r['cohort']==cohort]
        self.lru = OrderedDict()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self,i):
        if i not in self.lru:
            r = self.entries[i]
            p = self.root/r['cache_file']
            if sha(p) != r['cache_sha256']:
                raise ValueError('Frozen joint cache modified')
            with np.load(p,allow_pickle=False) as z:
                values = {k:z[k].copy() for k in z.files}
            if set(values) != set(r['array_hashes']):
                raise ValueError('Unexpected source cache array')
            for k,v in values.items():
                if not np.isfinite(v).all() or array_sha(v) != r['array_hashes'][k]:
                    raise ValueError('Invalid cache array '+k)
            self.lru[i] = dict(r,**values)
            while len(self.lru)>4:
                self.lru.popitem(last=False)
        self.lru.move_to_end(i)
        return self.lru[i]

def tensors(item,device):
    import torch
    return {k:torch.as_tensor(item[k],device=device)[None].float()
            for k in ('latent','source_semantics','text_wire','oracle_memory')}
