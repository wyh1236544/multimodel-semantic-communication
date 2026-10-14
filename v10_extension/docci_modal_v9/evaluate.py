"""Full-source private-branch reconstruction; source oracles are separate."""
import hashlib
import json
import time
import numpy as np
from PIL import Image
import torch
from docci_modal_v9.core import atomic_json,digest,geometry,metrics
from docci_modal_v9.data import batch
from docci_modal_v9.model import generate
from docci_modal_v9.channel import resource,send,physical_config
from fso_fluctuating_v2.core import paths
from fso_jscc_v1.core import seed_for
from optical_realism_v3.constellation import constellation_svg
from utils.metrics import batch_image_metrics

CONTROLS=('matched','zero','shuffled')


def array_hash(x): return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


@torch.no_grad()
def score(model,item,output,backbone,tok,device,folder,stem,save_image):
    if model.branch=='text':
        result=generate(model,output,tok)
        return dict(source_text=item['text'],source_text_sha256=item['text_sha256'],source_tokens=item['source_tokens'],input_truncated=False,
            received_caption=result['text'],**result,**metrics(item['text'],item['text'],result['text']))
    target=torch.as_tensor(item['rgb'].copy(),device=device).permute(2,0,1)[None].float()/255.
    decoded=((backbone.decode(output['latent']/float(backbone.config.scaling_factor)).sample+1)/2).clamp(0,1) if bool(output['available'][0]) else torch.zeros_like(target)
    l,t,r,b=[int(v) for v in item['rectangle']]; scores=batch_image_metrics(decoded[:,:,t:b,l:r],target[:,:,t:b,l:r])
    rgb=(decoded[0].permute(1,2,0).cpu().numpy()*255).round().clip(0,255).astype(np.uint8)
    record=dict(image_size=model.c['image_size'],native_image_size=item['native_size'],evaluation_rectangle=item['rectangle'].tolist(),
        source_image_sha256=item['image_sha256'],receiver_failure=not bool(output['available'][0]),
        restored_image_file=None,restored_image_sha256=None,quantized_image_sha256=array_hash(rgb),**{k:float(v[0]) for k,v in scores.items()})
    if save_image:
        name='images/'+stem+'.png'; Image.fromarray(rgb).save(folder/name)
        source_name='images/source__'+item['example_id']+'.png'
        if not (folder/source_name).exists(): Image.fromarray(item['rgb']).save(folder/source_name)
        record.update(restored_image_file=name,restored_image_sha256=digest(folder/name),source_view_file=source_name,source_view_sha256=digest(folder/source_name))
    return record


@torch.no_grad()
def native(model,items,backbone,tok,device,budget,folder,cohort):
    folder.mkdir(exist_ok=False); (folder/'images').mkdir(); rows=[]
    for i,item in enumerate(items):
        budget(); source=batch(items,[i],model.branch,device)
        if model.branch=='text':
            output=dict(memory=source['bart_hidden'].float(),receiver_mask=source['ids'].ne(tok.pad_token_id),available=torch.ones(1,device=device,dtype=torch.bool))
        else: output=dict(latent=source['latent'].float(),available=torch.ones(1,device=device,dtype=torch.bool))
        row=dict(example_id=item['example_id'],official_split=item['official_split'],cohort=cohort,branch=model.branch,
            scope='ORACLE original backbone source memory/latent; no compression or transmission; NOT theoretical upper bound',
            **score(model,item,output,backbone,tok,device,folder,'native__'+item['example_id'],True))
        rows.append(row)
    with (folder/'rows.jsonl').open('x',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    atomic_json(folder/'NATIVE_COMPLETE.json',dict(rows=len(rows),rows_sha256=digest(folder/'rows.jsonl'),no_transmission=True))
    return rows


@torch.no_grad()
def evaluate(model,items,c,backbone,tok,device,budget,folder,stage,cohort,physical=False,save_images=True):
    folder.mkdir(exist_ok=False); (folder/'frames').mkdir(); (folder/'images').mkdir(); (folder/'constellations').mkdir()
    cfg=physical_config(); conditions=[None]+(cfg['conditions'] if physical else [])
    weather={w:paths(cfg,w,len(items),'v9-independent-qual-dev') for w in set(v['weather'] for v in cfg['conditions'])} if physical else {}
    res=resource(c,model.branch,cfg); wires=[]; started=time.monotonic(); rows=[]
    for i in range(len(items)):
        budget(); wire=model.encode(batch(items,[i],model.branch,device))[0].float().cpu().numpy()
        if wire.shape!=(geometry(c,model.branch)['total_reals'],) or not np.isfinite(wire).all() or abs(float(np.mean(wire**2))-1)>1e-4: raise ValueError('Private payload shape/power differs')
        wires.append(wire)
    matrix=dict(branch=model.branch,stage=stage,cohort=cohort,example_ids=[i['example_id'] for i in items],controls=list(CONTROLS),
        conditions=['clean']+[x['id'] for x in conditions if x],resources=res,raw_text_never_RX=True,Common_or_other_modality_used=False,
        failures_retained=True,zero_scope='post-reconstruction BART memory' if model.branch=='text' else 'received image wire',
        control_scope='same stored received frame; NOT new transmissions',clean_snr40_is_sentinel=True)
    atomic_json(folder/'matrix.json',matrix)
    with (folder/'rows.jsonl').open('x',encoding='utf-8') as log:
        for condition in conditions:
            cid='clean' if condition is None else condition['id']; frames=[]
            for item,wire in zip(items,wires):
                budget(); seed=seed_for(c['seed'],'v9-noise',cid,item['example_id']); transport=None
                if condition is None:
                    y=wire.copy(); stats=dict(receiver_available=True,pilot_snr_db=40.,analog_reliability=1.,resources=res,
                        clean_sentinel_not_measured_snr=True,payload_complex_symbols=res['payload_complex_symbols'],analog_repeats=1,raw_utf8_wire_bytes=0,
                        physical_symbol_snr_db=None,pre_fec_ber=None,post_fec_ber=None)
                else:
                    index=len(frames); y,stats,transport=send(c,model.branch,cfg,condition,weather[condition['weather']][index],seed,wire)
                name='frames/'+cid+'__'+item['example_id']+'.npz'
                with (folder/name).open('xb') as f:
                    np.savez_compressed(f,observed=y.astype(np.float32),available=np.asarray(stats['receiver_available'],np.bool_),
                        pilot_snr_db=np.asarray(stats['pilot_snr_db'],np.float32),reliability=np.asarray(stats['analog_reliability'],np.float32))
                if transport and len(frames)<c['save_images']: constellation_svg(transport.constellations,folder/'constellations'/ (cid+'__'+item['example_id']+'.svg'))
                frames.append(dict(name=name,sha256=digest(folder/name),received=y,stats=stats,noise_seed=seed,tx_hash=array_hash(wire),rx_hash=array_hash(y)))
            for i,(item,frame) in enumerate(zip(items,frames)):
                donor=frames[(i+1)%len(frames)]; stats=frame['stats']
                y=torch.as_tensor(frame['received'],device=device)[None]; ok=torch.tensor([stats['receiver_available']],device=device)
                snr=y.new_tensor([stats['pilot_snr_db']]); rel=y.new_tensor([stats['analog_reliability']])
                for control in CONTROLS:
                    budget(); observed=torch.zeros_like(y) if control=='zero' and model.branch=='image' else y
                    if control=='shuffled': observed=torch.as_tensor(donor['received'],device=device)[None]
                    output=model.receive(observed,ok,snr,rel)
                    if control=='zero' and model.branch=='text': output['memory']=torch.zeros_like(output['memory'])
                    row=dict(branch=model.branch,stage=stage,cohort=cohort,example_id=item['example_id'],official_split=item['official_split'],condition=cid,control=control,
                        frame_file=frame['name'],frame_sha256=frame['sha256'],received_sha256=frame['rx_hash'],tx_wire_sha256=frame['tx_hash'],
                        donor_example_id=items[(i+1)%len(items)]['example_id'] if control=='shuffled' else None,
                        donor_received_sha256=donor['rx_hash'] if control=='shuffled' else None,stats=stats,noise_seed=frame['noise_seed'],
                        **score(model,item,output,backbone,tok,device,folder,cid+'__'+item['example_id']+'__'+control,save_images and i<c['save_images']))
                    rows.append(row); log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); log.flush()
            print('PRIVATE EVAL {} {} {} {} finished'.format(model.branch,stage,cohort,cid),flush=True)
    atomic_json(folder/'EVAL_COMPLETE.json',dict(rows=len(rows),rows_sha256=digest(folder/'rows.jsonl'),matrix_sha256=digest(folder/'matrix.json'),wall_seconds=time.monotonic()-started))
    return rows
