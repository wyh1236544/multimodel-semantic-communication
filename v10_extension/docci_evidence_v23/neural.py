"""Explicit frozen CUDA stages, compatible with existing Torch1.12/HF4.30."""
from contextlib import contextmanager
from pathlib import Path
import math
import numpy as np
from PIL import Image
from docci_evidence_v23.core import (PROFILES,CAPTION_SNR,CONTROLS,PROTOCOL,
    atomic,line,complete,prepared,read,sha,digest,local,seed,noise,donor_map,
    case_key,case_name,metric,prompts,residual_stats,verify_tree)

def observed(wire,eid,snr,repeat,kind):
    import torch
    from docci_text_guided_image_v20.core import array_sha
    n=8192 if kind=='image' else 18432
    if tuple(wire.shape)!=(1,n) or abs(float(wire.square().mean())-1)>.0002:
        raise ValueError('Wrong frozen payload/power')
    e=noise('v23',eid,snr,repeat,kind,n);p=noise('v23',eid,snr,repeat,kind+'_pilot',64)
    n0=float(np.mean(p.astype(float)**2));est=40. if snr is None else max(-40.,min(40.,-10*math.log10(max(n0,1e-12))))
    return (wire+torch.as_tensor(e[None],device=wire.device),wire.new_tensor([est]),
        wire.new_tensor([1/(1+n0)]),dict(nominal_snr_db=snr,
        pilot_noise_variance=n0,estimated_snr_db=None if snr is None else est,
        realized_snr_db=None if snr is None else -10*math.log10(float(np.mean(e.astype(float)**2))),
        noise_sha256=array_sha(e),pilot_sha256=array_sha(p),signal_power=float(wire.square().mean()),
        post_noise_renormalization=False))

@contextmanager
def parents(a,root,budget):
    import torch
    from docci_pretrained_checks_v22.neural import cuda
    from docci_text_aid_v21.core import CACHE_SHA,PARENTS
    from docci_text_aid_v21.runtime import anchors
    from docci_text_guided_image_v20_fix1.compat import validate_gate,pooling_patch
    from docci_text_guided_image_v20 import model as old_model
    from docci_text_guided_image_v20.data import Items
    from docci_text_guided_image_v20.engine import make,state_hash,signatures,check_frozen
    old=Path(a.v20_root)
    if read(old/'PIPELINE.json')['status']!='COMPLETE' or sha(old/'cache/index.json')!=CACHE_SHA:
        raise ValueError('Pinned completed V20/cache required')
    validate_gate(read(old/'FIX1_GPU_GATE.json'))
    ck=old/'train_A_self/last.pth'
    if sha(ck)!=PARENTS['A_self']: raise ValueError('Actual V20 A weight differs before load')
    a.cache=str(old/'cache');dev=Items(a.cache,'dev')
    if len(dev)!=100: raise ValueError('Expected original 100 dev')
    selected=sorted(dev.entries,key=lambda r:seed(42,'V22-fixed-dev',r['example_id']))
    device=cuda();budget.gpu=True
    with pooling_patch(old_model),torch.no_grad():
        image,vae,text,lpips=make(a,device,ck)
        image.requires_grad_(False).eval()
        before=state_hash(image);frozen=signatures(vae,text,lpips)
        anchors(a,dict(selected={'dev':selected[:2],'train':[]},mask_blocks=0),root,
            'A_self',image,vae,text,lpips,device,budget)
        yield image,vae,text,lpips,device,dev,selected
        if before!=state_hash(image): raise RuntimeError('Image codec changed')
        check_frozen((vae,text,lpips),frozen)

def prepare(a,root,budget):
    from transformers import BartTokenizer
    from docci_text_guided_image_v20.data import tensors
    from docci_text_guided_image_v20.engine import rgb_decode
    from docci_baseline_v14.model import decode
    spec=PROFILES[a.profile]
    (root/'truth').mkdir();(root/'receiver').mkdir()
    with parents(a,root,budget) as (image,vae,text,lpips,device,items,selected):
        tok=BartTokenizer.from_pretrained(a.bart_path,local_files_only=True)
        selected=selected[:spec['images']];ids=[r['example_id'] for r in selected];donors=donor_map(ids)
        positions={r['example_id']:i for i,r in enumerate(items.entries)};examples=[]
        with (root/'rows.jsonl').open('x',encoding='utf-8') as f:
            for i,entry in enumerate(selected):
                budget();item=items[positions[entry['example_id']]];src=tensors(item,device);eid=item['example_id']
                if item['cohort']!='dev' or item['official_split']!='train' or item['input_truncated']:
                    raise ValueError('Invalid selected dev input')
                source=root/'truth'/(eid+'.png');Image.fromarray(item['rgb']).save(source)
                examples.append(dict(example_id=eid,cohort='dev',original_split='train',source_image=source.relative_to(root).as_posix(),
                    source_image_sha256=sha(source),source_text=item['text'],source_text_sha256=item['text_sha256'],rectangle=list(map(int,item['rectangle']))))
                wire=image.encode(src['latent'],None)
                for snr in spec['snr_db']:
                    for rep in range(spec['repeats']):
                        budget();y,est,rel,stats=observed(wire,eid,snr,rep,'image')
                        latent=image.receive(y,None,est,rel);rgb=rgb_decode(vae,latent).clamp(0,1)
                        path=root/'receiver'/(case_name(eid,snr,rep)+'.png')
                        Image.fromarray((rgb[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)).save(path)
                        caption=None;ts=None
                        if snr in CAPTION_SNR:
                            ty,_,_,ts=observed(src['text_wire'],eid,snr,rep,'text')
                            caption=decode(text.bart,text.receive(ty),tok,'greedy')['received_caption']
                        line(f,dict(example_id=eid,donor_id=donors[eid],snr_db=snr,repeat=rep,
                            received_image=path.relative_to(root).as_posix(),received_image_sha256=sha(path),
                            received_caption=caption,received_caption_sha256=None if caption is None else digest(caption.encode()),
                            image_stats=stats,text_stats=ts,image_payload_symbols=4096,text_payload_symbols=9216,
                            image_pilot_symbols=32,text_pilot_symbols=32,total_symbols=13376,
                            source_text_at_RX=False,post_noise_renormalization=False,negative_SNR_is_OOD=snr is not None and snr<0))
                print('V23 PREPARE {}/{} frozen branches; caption at clean/0/-5 only'.format(i+1,len(selected)),flush=True)
    atomic(root/'index.json',dict(schema='v23-prepared',profile=a.profile,spec=spec,examples=examples,
        noise_family='v23',independent_test=False,reused_dev=True))
    complete(root,'prepare',dict(rows=len(ids)*len(spec['snr_db'])*spec['repeats']))

def caption(a,root,budget):
    import torch
    from transformers import Blip2Processor,Blip2ForConditionalGeneration
    from docci_pretrained_checks_v22.neural import cuda
    from docci_longclip_joint_v8.backbones import signature,verify_frozen
    index,data,_=prepared(a.prepared)
    data=[r for r in data if r['snr_db'] in CAPTION_SNR]
    lock=read(a.asset_lock)
    if lock.get('kind')!='caption': raise ValueError('Sealed caption assets required')
    folder=verify_tree(lock['tree']);cfg=read(folder/'config.json')
    if cfg.get('model_type')!='blip-2' or cfg.get('image_token_index') is not None:
        raise ValueError('Use the pinned pre-image-token 2023 BLIP-2 snapshot, not current main')
    device=cuda();budget.gpu=True
    processor=Blip2Processor.from_pretrained(str(folder),local_files_only=True)
    model=Blip2ForConditionalGeneration.from_pretrained(str(folder),local_files_only=True,
        torch_dtype=torch.float16,low_cpu_mem_usage=False).to(device).requires_grad_(False).eval()
    frozen=signature(model);by={case_key(r):r for r in data};blank=Image.new('RGB',(512,512),(128,128,128))
    def generate(im,prompt):
        budget();batch=processor(images=im,text=prompt,return_tensors='pt',truncation=False)
        n=int(batch['input_ids'].shape[-1]);limit=getattr(model.config.text_config,'max_position_embeddings',2048)
        if n+PROTOCOL['caption_max_new_tokens']+model.config.num_query_tokens>limit:
            raise ValueError('Full received text exceeds context; no truncation')
        batch={k:v.to(device,dtype=torch.float16) if v.is_floating_point() else v.to(device) for k,v in batch.items()}
        with torch.no_grad():
            ids=model.generate(**batch,do_sample=False,num_beams=1,max_new_tokens=PROTOCOL['caption_max_new_tokens'])
        values=ids[0].tolist();eos=model.config.text_config.eos_token_id
        eos_set={eos} if isinstance(eos,int) else set(eos)
        natural=any(v in eos_set for v in values[1:])
        return processor.batch_decode(ids,skip_special_tokens=True)[0].strip(),dict(input_tokens=n,
            returned_tokens=len(values),natural_EOS=natural,capacity_stop=not natural,
            maximum_new_tokens=512,do_sample=False,num_beams=1)
    with (root/'rows.jsonl').open('x',encoding='utf-8') as f:
        for i,r in enumerate(data):
            donor=by[r['donor_id'],r['snr_db'],r['repeat']]
            with Image.open(local(a.prepared,r['received_image'])) as im: own=im.convert('RGB')
            with Image.open(local(a.prepared,donor['received_image'])) as im: wrong=im.convert('RGB')
            cap,joint=prompts(r['received_caption'])
            for control,im,prompt in (('received_text',None,None),('caption_only',own,cap),
                    ('joint_correct',own,joint),('joint_wrong_image',wrong,joint),('joint_blank_image',blank,joint),
                    ('joint_wrong_text',own,prompts(donor['received_caption'])[1])):
                output,telemetry=(r['received_caption'],{}) if im is None else generate(im,prompt)
                line(f,dict(example_id=r['example_id'],snr_db=r['snr_db'],repeat=r['repeat'],control=control,
                    output_text=output,output_sha256=digest(output.encode()),telemetry=telemetry,
                    prompt_sha256=None if prompt is None else digest(prompt.encode()),
                    conditioning_image_sha256=None if im is None else digest(np.asarray(im).tobytes()),
                    received_caption_sha256=r['received_caption_sha256'],source_text_at_RX=False,
                    total_symbols=13376,caption_only_symbols=4128 if control=='caption_only' else None))
            print('V23 CAPTION {}/{} all six controls'.format(i+1,len(data)),flush=True)
    verify_frozen((model,),(frozen,))
    atomic(root/'identity.json',dict(prepared_sha256=sha(Path(a.prepared)/'COMPLETE.json'),asset_lock_sha256=sha(a.asset_lock),
        overlap_audit=lock.get('overlap_audit','unknown'),preprocessing='pinned processor resize/crop; not native512',
        caption_only_is_not_joint_recovery=True,instruction_following_not_guaranteed=True))
    complete(root,'caption',dict(rows=len(data)*len(CONTROLS),facts='PENDING'))

def score(a,root,budget):
    import torch
    from docci_pretrained_checks_v22.neural import cuda
    from docci_evidence_v23.core import verify_files
    from docci_text_guided_image_v20.perceptual import Perceptual
    _,_,examples=prepared(a.prepared);target=Path(a.target);receipt=verify_files(target)
    if receipt['stage'] not in ('prepare','digital'): raise ValueError('Only direct/digital image stages')
    if receipt['stage']=='digital' and read(target/'identity.json')['prepared_sha256']!=sha(Path(a.prepared)/'COMPLETE.json'):
        raise ValueError('Different preparation')
    device=cuda();budget.gpu=True;lpips=Perceptual(a.lpips_assets).to(device)
    with torch.no_grad(),(root/'rows.jsonl').open('x',encoding='utf-8') as f:
        for i,r in enumerate(rows(target/'rows.jsonl')):
            budget();ex=examples[r['example_id']];name=r.get('image_file',r.get('received_image'))
            with Image.open(local(target,name)) as im: image=np.asarray(im.convert('RGB')).copy()
            with Image.open(local(a.prepared,ex['source_image'])) as im: truth=np.asarray(im.convert('RGB')).copy()
            l,t,right,bottom=ex['rectangle']
            def tensor(x): return torch.as_tensor(x[t:bottom,l:right].copy(),device=device).permute(2,0,1)[None].float()/255.
            result=metric(truth,image,ex['rectangle']);result['LPIPS']=float(lpips(tensor(image),tensor(truth)).mean())
            line(f,dict(example_id=r['example_id'],snr_db=r['snr_db'],repeat=r['repeat'],
                control=r.get('control','continuous'),image_sha256=sha(local(target,name)),**result))
            if i%20==0: print('V23 SCORE {}/{}'.format(i+1,receipt['metadata']['rows']),flush=True)
    atomic(root/'identity.json',dict(prepared_sha256=sha(Path(a.prepared)/'COMPLETE.json'),target_sha256=sha(target/'COMPLETE.json'),
        target_path=str(target.resolve()),LPIPS='frozen Alex v0.1; uint8 PNG; identical valid rectangle'))
    complete(root,'score',dict(rows=i+1))

def calibrate(a,root,budget):
    import torch
    from docci_text_guided_image_v20.data import Items,tensors
    from docci_text_guided_image_v20.engine import rgb_decode
    count=8 if a.profile=='smoke' else 32
    (root/'arrays').mkdir();(root/'truth').mkdir();(root/'images').mkdir()
    with parents(a,root,budget) as (image,vae,text,lpips,device,dev,selected):
        items=Items(a.cache,'train');order=sorted(range(len(items)),key=lambda i:seed(42,'V23-TRAIN-calibration',items.entries[i]['example_id']))[:count]
        with (root/'rows.jsonl').open('x',encoding='utf-8') as f:
            for i,ix in enumerate(order):
                budget();item=items[ix];src=tensors(item,device);eid=item['example_id']
                if item['cohort']!='train' or item['official_split']!='train': raise ValueError('Calibration uses TRAIN only')
                source_image=root/'truth'/(eid+'.png');Image.fromarray(item['rgb']).save(source_image)
                z=src['latent'];wire=image.encode(z,None);clean=image.receive(wire,None,wire.new_tensor([40.]),wire.new_tensor([1.]))
                for snr in CAPTION_SNR:
                    y,est,rel,stats=observed(wire,eid,snr,0,'image');received=image.receive(y,None,est,rel)
                    rgb=rgb_decode(vae,received).clamp(0,1)
                    pixels=(rgb*255).round()/255.
                    reencoded=vae.encode(pixels*2-1).latent_dist.mean*float(vae.config.scaling_factor)
                    after=rgb_decode(vae,reencoded).clamp(0,1)
                    def array(v): return v[0].float().cpu().numpy()
                    path=root/'arrays'/(case_name(eid,snr,0)+'.npz')
                    with path.open('xb') as dest:
                        np.savez(dest,source_latent=array(z),received_latent=array(received),
                            clean_received_latent=array(clean),reencoded_latent=array(reencoded))
                    def png(v): return (v[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
                    direct_image=root/'images'/(case_name(eid,snr,0)+'__direct.png')
                    roundtrip_image=root/'images'/(case_name(eid,snr,0)+'__roundtrip.png')
                    Image.fromarray(png(rgb)).save(direct_image);Image.fromarray(png(after)).save(roundtrip_image)
                    line(f,dict(example_id=eid,cohort='train',snr_db=snr,repeat=0,array_file=path.relative_to(root).as_posix(),
                        source_image=source_image.relative_to(root).as_posix(),source_image_sha256=sha(source_image),
                        direct_image=direct_image.relative_to(root).as_posix(),direct_image_sha256=sha(direct_image),
                        roundtrip_image=roundtrip_image.relative_to(root).as_posix(),roundtrip_image_sha256=sha(roundtrip_image),
                        rectangle=list(map(int,item['rectangle'])),
                        array_sha256=sha(path),source_error=residual_stats(array(received-z)),
                        incremental_channel_error=residual_stats(array(received-clean)),
                        roundtrip_error=residual_stats(array(reencoded-received)),
                        direct=metric(item['rgb'],png(rgb),item['rectangle']),roundtrip=metric(item['rgb'],png(after),item['rectangle']),
                        channel_stats=stats,source_latent_used_only_after_receive_for_diagnosis=True))
                print('V23 CALIBRATE TRAIN {}/{}; no diffusion/no training'.format(i+1,count),flush=True)
    atomic(root/'identity.json',dict(cohort='train',count=count,snr_db=CAPTION_SNR,
        VAE='same pinned sd-vae-ft-mse, NOT independent SD VAE compatibility validation',
        timestep_mapping=None,diffusion_executed=False,Gaussianity_proven=False,
        latent_statistics_are_oracle_diagnostics_not_receiver_inputs=True))
    complete(root,'calibrate',dict(rows=count*len(CAPTION_SNR),array_files=count*len(CAPTION_SNR)))
