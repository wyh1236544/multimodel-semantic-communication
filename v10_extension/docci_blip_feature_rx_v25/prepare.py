"""One frozen feature cache; actual RX RGB only enters pretrained BLIP-2."""
from pathlib import Path
import time
import numpy as np
from PIL import Image
from docci_blip_feature_rx_v25.core import (read,sha,atomic,line,array_sha,choose,CHANNELS,
    REGIMES,regimes,save_npz,seal)

def create(a,c,root,device,budget):
    import torch
    from docci_text_aid_v21.core import CACHE_SHA,PARENTS
    from docci_text_aid_v21.runtime import anchors
    from docci_text_guided_image_v20.data import Items,tensors
    from docci_text_guided_image_v20.engine import make,rgb_decode,state_hash,signatures,check_frozen
    from docci_text_guided_image_v20_fix1.compat import pooling_patch,validate_gate
    from docci_text_guided_image_v20 import model as old_model
    from docci_pretrained_checks_v22.neural import observed
    from docci_caption_calibration_v24 import runtime
    from docci_caption_calibration_v24.core import verify_assets
    from docci_caption_calibration_v24_fix1.fix import model_view
    from docci_caption_calibration_v24_fix2.fix import verify as verify_caption,metadata_copy
    from docci_longclip_joint_v8.backbones import signature
    from docci_blip_feature_rx_v25.model import content_gate
    from docci_blip_feature_rx_v25.visual import extract
    start = time.monotonic(); parent = Path(a.v20_root)
    if read(parent/'PIPELINE.json')['status']!='COMPLETE' or sha(parent/'cache/index.json')!=CACHE_SHA:
        raise ValueError('Completed pinned V20 full source cache required')
    gate_receipt = validate_gate(read(parent/'FIX1_GPU_GATE.json'))
    checkpoint = parent/'train_A_self/last.pth'
    if sha(checkpoint)!=PARENTS['A_self']: raise ValueError('Pinned V20 A checkpoint differs')
    proof = verify_caption(a.caption_root)
    if proof['stage']!='caption' or proof['empty_basic_outputs'] or proof['capacity_stops']:
        raise ValueError('Completed V24 basic caption engineering proof required')
    if sha(Path(a.caption_root)/'ASSET_LOCK.json')!=sha(a.asset_lock):
        raise ValueError('Use same sealed BLIP-2 assets as V24 proof')
    lock = read(a.asset_lock); asset_path = verify_assets(lock,budget)
    asset_stats = {n:[(asset_path/n).stat().st_size,(asset_path/n).stat().st_mtime_ns] for n in lock['tree']['files']}
    selected = choose(read(parent/'cache/index.json'),c); a.cache = str(parent/'cache')
    atomic(root/'config.json',c)
    atomic(root/'ASSET_LOCK.json',lock)
    atomic(root/'PARENTS.json',dict(cache_index_sha256=CACHE_SHA,image_checkpoint_sha256=PARENTS['A_self'],
        caption_complete_sha256=sha(Path(a.caption_root)/'COMPLETE.json'),caption_engineering=proof,
        caption_human_QA_authorization=False,old_GPU_gate=gate_receipt))
    by = {}
    for group in ('train','dev'):
        items = Items(a.cache,group); lookup = {r['example_id']:i for i,r in enumerate(items.entries)}
        by[group] = items,lookup
    print('V25 LOAD frozen V20 A/VAE/V18 and BLIP-2; no source captions in visual encoder',flush=True)
    image,vae,text,lpips = make(a,device,checkpoint); image.requires_grad_(False).eval()
    ibefore,frozen = state_hash(image),signatures(vae,text,lpips)
    blip,processor,loading = runtime.load_model(asset_path,torch.float16,device)
    blip = model_view(blip)
    print('V25 AUDIT frozen BLIP-2 tensors before feature cache',flush=True)
    bbefore = runtime.state_digest(blip,budget)
    metadata,conversions = metadata_copy(dict(loading_info=loading,config=blip.config.to_dict(),
        generation_config=blip.generation_config.to_dict(),processor=processor.image_processor.to_dict()))
    atomic(root/'BLIP_MODEL.json',dict(metadata=metadata,dtype_conversions=conversions,
        pretrained_alignment='BLIP2 QFormer -> OPT, NOT pretrained QFormer -> BART',frozen=True,
        visual_shape=[32,768],native_visual_input=224,full512_detail_preservation=False))
    with torch.no_grad():
        gray,gray_trace = extract(blip,processor,Image.new('RGB',(512,512),(128,128,128)),device,torch.float16,budget)
    if tuple(gray.shape)!=(1,32,768): raise ValueError('Pinned BLIP-2 feature shape differs')
    save_npz(root/'gray.npz',visual=gray[0].cpu().numpy())
    atomic(root/'gray.json',dict(file='gray.npz',sha256=sha(root/'gray.npz'),trace=gray_trace,
        visual_slice_sha256=array_sha(gray.cpu().numpy())))
    (root/'arrays').mkdir(); (root/'images').mkdir()
    entries = []; captions = []
    with pooling_patch(old_model),torch.no_grad():
        anchors(a,dict(selected=selected,mask_blocks=0),root,'A_self',image,vae,text,lpips,device,budget)
        for group in ('train','dev'):
            items,lookup = by[group]
            for i,r in enumerate(selected[group]):
                budget(); item = items[lookup[r['example_id']]]; src = tensors(item,device)
                wire = image.encode(src['latent'],None)
                ms,gs,vs,rgbs,traces,imstats,tstats = [],[],[],[],[],[],[]
                image_files = {}
                for snr in CHANNELS:
                    y,est,rel,ims = observed(wire,r['example_id'],snr,0,'image')
                    latent = image.receive(y,None,est,rel)
                    rgb = rgb_decode(vae,latent).clamp(0,1)
                    # Quantized received RGB, same representation as the V24 PNG path.
                    pixels = (rgb[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
                    received = Image.fromarray(pixels)
                    visual,trace = extract(blip,processor,received,device,torch.float16,budget)
                    if visual.shape!=(1,32,768): raise ValueError('Q-Former geometry differs')
                    ty,_,_,ts = observed(src['text_wire'],r['example_id'],snr,0,'text')
                    memory = text.receive(ty)['memory']; gate = content_gate(text.bart,memory)
                    ms.append(memory[0].cpu().numpy()); gs.append(gate[0].cpu().numpy())
                    vs.append(visual[0].cpu().numpy()); rgbs.append(pixels)
                    traces.append(trace); imstats.append(ims); tstats.append(ts)
                    if group=='dev':
                        name = 'images/{}__{}.png'.format(r['example_id'],'clean' if snr is None else 'snr'+str(snr))
                        received.save(root/name); image_files[str(snr)] = dict(file=name,sha256=sha(root/name))
                        value = runtime.generate_one(blip,processor,received,None,'caption',device,torch.float16,budget)
                        captions.append(dict(example_id=r['example_id'],image_snr_db=snr,
                            image_sha256=sha(root/name),**value))
                arrays = dict(ids=item['ids'].astype(np.int64),memory=np.stack(ms).astype(np.float32),
                    gate=np.stack(gs).astype(np.float32),visual=np.stack(vs).astype(np.float32),received_rgb=np.stack(rgbs))
                name = 'arrays/'+r['example_id']+'.npz'; checksum = save_npz(root/name,**arrays)
                entry = dict(example_id=r['example_id'],cohort=group,official_split=r['official_split'],
                    text=r['text'],text_sha256=r['text_sha256'],source_tokens=r['source_tokens'],
                    source_image_sha256=r['image_sha256'],source_cache_sha256=r['cache_sha256'],
                    file=name,sha256=checksum,array_hashes={k:array_sha(v) for k,v in arrays.items()},
                    visual_slice_sha256=[array_sha(arrays['visual'][j:j+1]) for j in range(3)],
                    memory_slice_sha256=[array_sha(arrays['memory'][j:j+1]) for j in range(3)],
                    image_files=image_files,visual_traces=traces,image_stats=imstats,text_stats=tstats,
                    input_truncated=False,source_mask_at_RX=False)
                if group=='dev':
                    name = 'images/'+r['example_id']+'__source.png'; Image.fromarray(item['rgb']).save(root/name)
                    entry['source_image_file'] = name; entry['source_image_file_sha256'] = sha(root/name)
                entries.append(entry)
                if i==0 or (i+1)%20==0 or i+1==len(selected[group]):
                    print('V25 CACHE {} {}/{} three RX image/text conditions; elapsed={:.1f}s'.format(
                        group,i+1,len(selected[group]),time.monotonic()-start),flush=True)
    if state_hash(image)!=ibefore: raise RuntimeError('Frozen image parent changed')
    check_frozen((vae,text,lpips),frozen)
    print('V25 AUDIT frozen BLIP-2 tensors after feature cache',flush=True)
    bafter = runtime.state_digest(blip,budget)
    if bbefore!=bafter or any(p.requires_grad or p.grad is not None for p in blip.parameters()):
        raise RuntimeError('Frozen BLIP-2 changed')
    final_stats = {n:[(asset_path/n).stat().st_size,(asset_path/n).stat().st_mtime_ns] for n in lock['tree']['files']}
    if final_stats!=asset_stats: raise RuntimeError('BLIP-2 asset stat changed')
    if sha(parent/'cache/index.json')!=CACHE_SHA or sha(checkpoint)!=PARENTS['A_self']:
        raise RuntimeError('Historical cache/weights changed')
    atomic(root/'frozen_audit.json',dict(BLIP_before=bbefore,BLIP_after=bafter,
        image_unchanged=True,VAE_BART_V18_LPIPS_unchanged=True,pretrained_parameters_trained=0,
        asset_stats_before=asset_stats,asset_stats_after=final_stats,full_asset_hash_before_load=True))
    with (root/'caption_only.jsonl').open('x',encoding='utf-8') as f:
        for r in captions: line(f,r)
    atomic(root/'index.json',dict(records=entries,channels=list(CHANNELS),
        profile=c['profile'],received_only=True,pretraining_overlap='unknown',reused_dev=True,independent_test=False))
    atomic(root/'CACHE_RESULT.json',dict(examples=len(entries),wall_seconds=time.monotonic()-start,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,training_updates=0,
        shape_memory=[3,9,32,768],shape_visual=[3,32,768],image_symbols=4128,text_symbols=9248,total_symbols=13376))
    seal(root,'cache')

class Dataset:
    def __init__(self,root,cohort):
        from collections import OrderedDict
        self.root = Path(root); self.entries = [r for r in read(self.root/'index.json')['records'] if r['cohort']==cohort]
        self.cached = OrderedDict()
    def __len__(self): return len(self.entries)
    def __getitem__(self,i):
        if i not in self.cached:
            r = self.entries[i]; p = self.root/r['file']
            if sha(p)!=r['sha256']: raise ValueError('Fixed receiver cache changed')
            with np.load(p,allow_pickle=False) as z:
                value = {k:z[k].copy() for k in z.files if k!='received_rgb'}
            self.cached[i] = dict(r,**value)
            while len(self.cached)>4: self.cached.popitem(last=False)
        self.cached.move_to_end(i); return self.cached[i]

def observation(item,regime,device):
    import torch
    ims,ts = REGIMES[regime]
    it,tt = CHANNELS.index(ims),CHANNELS.index(ts)
    memory = torch.as_tensor(item['memory'][tt:tt+1],device=device)
    gate = torch.as_tensor(item['gate'][tt:tt+1],device=device)
    visual = torch.as_tensor(item['visual'][it:it+1],device=device)
    return memory,gate,visual,it,tt
