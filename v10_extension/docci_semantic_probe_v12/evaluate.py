"""Shared image observations and explicit oracle controls; no actual-text shortcut."""
import hashlib
import json
import numpy as np
from PIL import Image
import torch
from docci_semantic_probe_v12.core import atomic_json, digest, damage_numpy, seed_for
from utils.metrics import batch_image_metrics


def array_hash(value): return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def frames(items, c, folder):
    folder.mkdir(exist_ok=False); result={}
    for condition in c['image_eval_conditions']:
        for item in items:
            seed=seed_for(c['seed'],'EVAL',condition,item['example_id'])
            observed=damage_numpy(item['wire'],condition,seed)
            name=condition+'__'+item['example_id']+'.npz'
            with (folder/name).open('xb') as f: np.savez_compressed(f,observed=observed,transmitted=item['wire'])
            result[(condition,item['example_id'])]=dict(file='frames/'+name,sha256=digest(folder/name),
                tx_sha256=array_hash(item['wire']),rx_sha256=array_hash(observed),seed=seed,observed=observed)
    return result


@torch.no_grad()
def images(model, base, items, vae, c, device, budget, root, received, tag):
    folder=root/tag; folder.mkdir(exist_ok=False); (folder/'images').mkdir()
    oracle = model is not None and model.arm == 'A1_oracle_text'
    controls=['matched','shuffled','zero'] if oracle else ['none']
    matrix=dict(task='image',tag=tag,example_ids=[x['example_id'] for x in items],conditions=c['image_eval_conditions'],
        controls=controls,image_complex_symbols=4096,analog_repeats=1,FSO=False,
        ORACLE_text_untransmitted=oracle,text_channel_cost=None,
        scope='receiver-fusion diagnostic, NOT equal-total-resource joint communication',
        LPIPS=None,LPIPS_status='NOT_MEASURED: no additional pretrained model/dependency downloaded',
        primary_metric='PSNR',primary_condition=c['image_primary_condition'],
        synthetic_mask_scope='source-independent25% real-coordinate erasure, all8192 real coordinates still charged',
        clean40_is_condition_sentinel_NOT_measured_SNR=True)
    atomic_json(folder/'matrix.json',matrix); rows=[]
    with (folder/'rows.jsonl').open('x',encoding='utf-8') as log:
        for condition in c['image_eval_conditions']:
            for i,item in enumerate(items):
                frame=received[(condition,item['example_id'])]; donor=items[(i+1)%len(items)]
                wire=torch.as_tensor(frame['observed'],device=device)[None]
                for control in controls:
                    budget(); semantic=None; donor_id=None; sem_hash=None
                    if oracle:
                        values=donor['semantic'] if control=='shuffled' else item['semantic']
                        if control=='zero': values=np.zeros_like(values)
                        semantic=torch.as_tensor(values,device=device)[None]
                        donor_id=donor['example_id'] if control=='shuffled' else None; sem_hash=array_hash(values)
                    if tag=='P0_parent':
                        latent=base.receive(wire,torch.ones(1,device=device,dtype=torch.bool),wire.new_tensor([40.]),wire.new_tensor([1.]))['latent']
                    else: latent=model.receive(wire,semantic)
                    decoded=((vae.decode(latent/float(vae.config.scaling_factor)).sample+1)/2).clamp(0,1)
                    target=torch.as_tensor(item['rgb'].copy(),device=device).permute(2,0,1)[None].float()/255.
                    l,t,r,b=[int(v) for v in item['rectangle']]
                    scores=batch_image_metrics(decoded[:,:,t:b,l:r],target[:,:,t:b,l:r])
                    rgb=(decoded[0].permute(1,2,0).cpu().numpy()*255).round().clip(0,255).astype(np.uint8)
                    files={}
                    if i<c['save_images']:
                        name='images/'+condition+'__'+item['example_id']+'__'+control+'.png'
                        source='images/source__'+item['example_id']+'.png'
                        if not (folder/source).exists(): Image.fromarray(item['rgb']).save(folder/source)
                        Image.fromarray(rgb).save(folder/name)
                        files=dict(image_file=tag+'/'+name,image_file_sha256=digest(folder/name),
                            source_view_file=tag+'/'+source,source_view_sha256=digest(folder/source))
                    row=dict(tag=tag,control=control,condition=condition,example_id=item['example_id'],official_split=item['official_split'],
                        frame_file=frame['file'],frame_sha256=frame['sha256'],transmitted_sha256=frame['tx_sha256'],observed_sha256=frame['rx_sha256'],
                        damage_seed=frame['seed'],source_image_sha256=item['image_sha256'],source_text_sha256=item['text_sha256'],
                        source_text=item['text'],source_text_is_evaluation_or_ORACLE_only=True,rectangle=item['rectangle'].tolist(),
                        source_native_size=item['native_size'],quantized_image_sha256=array_hash(rgb),
                        semantic_sha256=sem_hash,semantic_donor_example_id=donor_id,
                        oracle_text_not_transmitted=oracle,receiver_failure=False,
                        payload_complex_symbols=4096,analog_repeats=1,physical_snr_db=None,BER=None,LPIPS=None,
                        **{k:float(v[0]) for k,v in scores.items()},**files)
                    rows.append(row); log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); log.flush()
                if i==0 or (i+1)%10==0 or i+1==len(items): print('IMAGE EVAL {} {} {}/{}'.format(tag,condition,i+1,len(items)),flush=True)
    atomic_json(folder/'EVAL_COMPLETE.json',dict(rows=len(rows),rows_sha256=digest(folder/'rows.jsonl'),matrix_sha256=digest(folder/'matrix.json')))
    return rows
