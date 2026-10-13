"""Paired held-out optical frames, retained failures and RX-only interventions."""
import hashlib
import json
import math
import time
import numpy as np
from PIL import Image
import torch
from docci_longclip_joint_v8.core import atomic_json,digest,geometry
from docci_longclip_joint_v8.data import tensors
from docci_longclip_joint_v8.model import generate
from docci_longclip_joint_v8.metrics import metrics
from docci_longclip_joint_v8.channel import send,physical_config,resource
from fso_fluctuating_v2.core import paths
from fso_jscc_v1.core import seed_for
from optical_realism_v3.constellation import constellation_svg
from utils.metrics import batch_image_metrics

CONTROLS=('normal','zero_common','fusion_off','shuffle_common','shuffle_image','shuffle_text')


def array_hash(x): return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


@torch.no_grad()
def evaluate(model,states,items,c,models,tok,device,budget,root,smoke=False):
    root.mkdir(exist_ok=False); (root/'frames').mkdir(); (root/'images').mkdir(); (root/'constellations').mkdir()
    physical=physical_config()
    if smoke: physical['conditions']=physical['conditions'][:1]
    res=resource(c,physical); started=time.monotonic(); bart,vae,_=models
    manifest=dict(controls=list(CONTROLS),arms=['separate','joint'],stages=sorted(set(k[1] for k in states)),
        conditions=['clean']+[v['id'] for v in physical['conditions']],example_ids=[x['example_id'] for x in items],
        channel_config=physical,resources=res,paired_scope='same path/noise seeds and public caps; learned TX waveforms and actual energies may differ',
        interventions_scope='same stored RX frame; counterfactual controls, NOT new resource-saving transmissions',
        common_split_separate='first16 image-only, last16 text-only; no cross-private fusion',joint_common='all32 jointly processed',
        no_diffusion=True,raw_text_transmitted=False,analog_repeats=1,failures_retained=True)
    atomic_json(root/'matrix.json',manifest)
    for item in items: Image.fromarray(item['rgb']).save(root/'images'/('source__'+item['example_id']+'.png'))
    weather={weather:paths(physical,weather,len(items),'v8-qual-dev-holdout') for weather in set(v['weather'] for v in physical['conditions'])}
    with (root/'rows.jsonl').open('x',encoding='utf-8') as log:
        for arm,stage in sorted(states):
            model.configure(arm); model.restore(states[(arm,stage)])
            wires=[]
            for item in items:
                budget(); wire=model.encode(tensors(item,device)).float()
                if not torch.isfinite(wire).all() or wire.shape!=(1,geometry(c)['total_reals']): raise ValueError('Actual TX payload differs')
                if abs(float(wire.square().mean())-1)>1e-4: raise ValueError('Full TX mean-square power differs from1')
                wires.append(wire[0].cpu().numpy())
            for condition in [None]+physical['conditions']:
                cid='clean' if condition is None else condition['id']; frames=[]
                for i,(item,wire) in enumerate(zip(items,wires)):
                    budget(); noise_seed=seed_for(c['seed'],'v8-holdout-noise',cid,item['example_id'])
                    if condition is None:
                        received=wire.copy(); stats=dict(receiver_available=True,pilot_snr_db=40.,analog_reliability=1.,
                            resources=res,payload_complex_symbols=res['payload_complex_symbols'],analog_repeats=1,
                            raw_utf8_wire_bytes=0,clean_sentinel_not_measured_snr=True,physical_symbol_snr_db=None)
                    else:
                        received,stats,transport=send(c,physical,condition,weather[condition['weather']][i],noise_seed,wire)
                        if i<c['save_images']: constellation_svg(transport.constellations,root/'constellations'/('{}__{}__{}__{}.svg'.format(arm,stage,cid,item['example_id'])))
                    name='frames/{}__{}__{}__{}.npz'.format(arm,stage,cid,item['example_id']); file=root/name
                    with file.open('xb') as f:
                        np.savez_compressed(f,observed=np.asarray(received,np.float32),available=np.asarray(stats['receiver_available'],np.bool_),
                            pilot_snr_db=np.asarray(stats['pilot_snr_db'],np.float32),reliability=np.asarray(stats['analog_reliability'],np.float32))
                    frames.append(dict(name=name,sha256=digest(file),received=received,stats=stats,noise_seed=noise_seed,
                        wire_sha256=array_hash(wire),received_sha256=array_hash(received)))
                for i,(item,frame) in enumerate(zip(items,frames)):
                    donor=frames[(i+1)%len(frames)]; own=frame['stats']
                    received=torch.as_tensor(frame['received'],device=device)[None]
                    available=torch.tensor([own['receiver_available']],device=device)
                    snr=received.new_tensor([own['pilot_snr_db']]); rel=received.new_tensor([own['analog_reliability']])
                    target=torch.as_tensor(item['rgb'].copy(),device=device).permute(2,0,1)[None].float()/255.
                    left,top,right,bottom=[int(v) for v in item['rectangle']]
                    for control in CONTROLS:
                        budget(); t0=time.monotonic()
                        out=model.receive(received,available,snr,rel,control,torch.as_tensor(donor['received'],device=device)[None])
                        result=generate(model,out['text'],tok)
                        decoded=((vae.decode(out['latent']/float(vae.config.scaling_factor)).sample+1)/2).clamp(0,1) if bool(available[0]) else torch.zeros_like(target)
                        values=batch_image_metrics(decoded[:,:,top:bottom,left:right],target[:,:,top:bottom,left:right])
                        rgb=(decoded[0].permute(1,2,0).cpu().numpy()*255).round().clip(0,255).astype(np.uint8)
                        image_file=None; image_sha=None
                        if control=='normal' or i<c['save_images']:
                            image_file='images/{}__{}__{}__{}__{}.png'.format(arm,stage,cid,item['example_id'],control)
                            Image.fromarray(rgb).save(root/image_file); image_sha=digest(root/image_file)
                        row=dict(arm=arm,stage=stage,condition=cid,example_id=item['example_id'],official_split=item['official_split'],control=control,
                            source_text=item['text'],covered_text=item['text'],source_text_sha256=item['text_sha256'],source_image_sha256=item['image_sha256'],
                            source_tokens=item['plan']['source_tokens'],clip_tokens=item['plan']['clip_tokens'],input_truncated=False,image_size=c['image_size'],
                            native_image_size=item['native_size'],evaluation_rectangle=item['rectangle'].tolist(),
                            received_caption=result['text'],stopped_by_capacity=result['stopped_by_capacity'],receiver_failure=result['receiver_failure'],
                            **metrics(item['text'],item['text'],result['text']))
                        row.update({k:float(v[0]) for k,v in values.items()})
                        row.update(frame_file=frame['name'],frame_sha256=frame['sha256'],received_sha256=frame['received_sha256'],tx_wire_sha256=frame['wire_sha256'],
                            noise_seed=frame['noise_seed'],donor_example_id=items[(i+1)%len(items)]['example_id'] if control.startswith('shuffle_') else None,
                            donor_received_sha256=donor['received_sha256'] if control.startswith('shuffle_') else None,stats=own,
                            restored_image_file=image_file,restored_image_sha256=image_sha,quantized_image_sha256=array_hash(rgb),
                            inference_wall_seconds=time.monotonic()-t0,metric_scope='full-source text and valid full-frame resized image rectangle; black padding excluded')
                        log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); log.flush()
                print('DOCCI EVAL {} {} {} {}/{} finished'.format(arm,stage,cid,len(items),len(items)),flush=True)
    atomic_json(root/'EVAL_COMPLETE.json',dict(rows_sha256=digest(root/'rows.jsonl'),wall_seconds=time.monotonic()-started,
        rows=len(states)*len(manifest['conditions'])*len(items)*len(CONTROLS),actual_raw_text_never_RX_input=True))


@torch.no_grad()
def native_diagnostics(model,items,models,tok,device,budget,path):
    """Oracle source features; labelled diagnostics, never a transmitted baseline."""
    bart,vae,_=models
    with path.open('x',encoding='utf-8') as log:
        for item in items:
            budget(); s=tensors(item,device); mask=s['ids'].ne(tok.pad_token_id)
            text=generate(model,dict(memory=s['bart_hidden'].float(),receiver_mask=mask,available=torch.ones(1,device=device,dtype=torch.bool)),tok)
            image=((vae.decode(s['latent'].float()/float(vae.config.scaling_factor)).sample+1)/2).clamp(0,1)
            target=torch.as_tensor(item['rgb'].copy(),device=device).permute(2,0,1)[None].float()/255.
            l,t,r,b=[int(v) for v in item['rectangle']]; scores=batch_image_metrics(image[:,:,t:b,l:r],target[:,:,t:b,l:r])
            row=dict(example_id=item['example_id'],source_text=item['text'],received_caption=text['text'],input_truncated=False,
                stopped_by_capacity=text['stopped_by_capacity'],scope='native frozen BART full encoder memory and VAE source latent; ORACLE no transmission; not theoretical upper bound',
                **metrics(item['text'],item['text'],text['text']))
            row.update({k:float(v[0]) for k,v in scores.items()}); log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
