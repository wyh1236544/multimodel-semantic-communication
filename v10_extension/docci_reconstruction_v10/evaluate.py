"""Same-frame controls; full source WER and actual images, all failures retained."""
import hashlib
import json
import numpy as np
from PIL import Image
import torch
from docci_modal_v9.evaluate import score as image_score,native as old_native
from docci_reconstruction_v10.core import atomic_json,digest,geometry,metrics
from docci_reconstruction_v10.data import batch
from docci_reconstruction_v10.model import generate
from docci_reconstruction_v10.channel import resource,send,physical_config
from fso_fluctuating_v2.core import paths
from fso_jscc_v1.core import seed_for
from optical_realism_v3.constellation import constellation_svg


def array_hash(x): return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


@torch.no_grad()
def score(model,item,out,backbone,tok,device,folder,stem,preview):
    if model.branch=='image': return image_score(model,item,out,backbone,tok,device,folder,stem,preview)
    result=generate(model,out,tok)
    return dict(source_text=item['text'],source_text_sha256=item['text_sha256'],source_tokens=item['source_tokens'],input_truncated=False,
        received_caption=result['text'],**result,**metrics(item['text'],item['text'],result['text']))


@torch.no_grad()
def native(model,items,backbone,tok,device,budget,folder,cohort):
    # Whole original source-memory reference, NOT segmentation oracle or transmission.
    return old_native(model,items,backbone,tok,device,budget,folder,cohort)


@torch.no_grad()
def native_chunks(model,items,backbone,tok,device,budget,folder):
    folder.mkdir(exist_ok=False); rows=[]
    for item in items:
        budget(); memory=torch.as_tensor(item['chunk_hidden'],device=device)[None].float()
        ids=torch.as_tensor(item['chunk_ids'],device=device)[None]
        output=dict(memory=memory,receiver_mask=ids.ne(tok.pad_token_id),available=torch.ones(1,device=device,dtype=torch.bool))
        row=dict(example_id=item['example_id'],official_split=item['official_split'],cohort='validation',branch='text',
            scope='ORACLE original chunk BART memories + source masks; NOT transmitted receiver; no compression',
            **score(model,item,output,backbone,tok,device,folder,'native_chunks',False))
        rows.append(row)
    with (folder/'rows.jsonl').open('x',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    atomic_json(folder/'NATIVE_COMPLETE.json',dict(rows=len(rows),rows_sha256=digest(folder/'rows.jsonl'),no_transmission=True))
    return rows


@torch.no_grad()
def evaluate(model,items,c,backbone,tok,device,budget,folder,stage,cohort,physical=False,preview=True):
    folder.mkdir(exist_ok=False)
    for name in ('frames','images','constellations'): (folder/name).mkdir()
    cfg=physical_config(); conditions=[None]+(cfg['conditions'] if physical else []); res=resource(c,model.arm,cfg)
    weather={w:paths(cfg,w,len(items),'v10-independent-qual-dev') for w in set(v['weather'] for v in cfg['conditions'])} if physical else {}
    wires=[]; rows=[]
    for i in range(len(items)):
        budget(); wire=model.encode(batch(items,[i],model.branch,device))[0].float().cpu().numpy()
        if wire.shape!=(geometry(c,model.arm)['total_reals'],) or not np.isfinite(wire).all() or abs(float(np.mean(wire**2))-1)>1e-4: raise ValueError('Private wire size/power differs')
        wires.append(wire)
    matrix=dict(arm=model.arm,branch=model.branch,stage=stage,cohort=cohort,example_ids=[i['example_id'] for i in items],
        controls=['matched','zero','shuffled'],conditions=['clean']+[x['id'] for x in conditions if x],resources=res,
        source_count_or_mask_at_RX=False,Common_used=False,failures_retained=True,
        zero_scope='post-reconstruction BART memory' if model.branch=='text' else 'received image wire',
        control_scope='same stored received frames; no retransmission',clean_snr40_is_sentinel=True)
    atomic_json(folder/'matrix.json',matrix)
    with (folder/'rows.jsonl').open('x',encoding='utf-8') as log:
        for condition in conditions:
            cid='clean' if condition is None else condition['id']; frames=[]
            for i,(item,wire) in enumerate(zip(items,wires)):
                budget(); seed=seed_for(c['seed'],'v10-noise',cid,item['example_id']); transport=None
                if condition is None:
                    y=wire.copy(); stats=dict(receiver_available=True,pilot_snr_db=40.,analog_reliability=1.,resources=res,
                        clean_sentinel_not_measured_snr=True,payload_complex_symbols=res['payload_complex_symbols'],analog_repeats=1,
                        raw_utf8_wire_bytes=0,pre_fec_ber=None,post_fec_ber=None)
                else: y,stats,transport=send(c,model.arm,cfg,condition,weather[condition['weather']][i],seed,wire)
                name='frames/'+cid+'__'+item['example_id']+'.npz'
                with (folder/name).open('xb') as f: np.savez_compressed(f,observed=y.astype(np.float32),available=np.asarray(stats['receiver_available'],np.bool_),pilot_snr_db=np.asarray(stats['pilot_snr_db'],np.float32),reliability=np.asarray(stats['analog_reliability'],np.float32))
                if transport and i<c['save_images']: constellation_svg(transport.constellations,folder/'constellations'/(cid+'__'+item['example_id']+'.svg'))
                frames.append(dict(name=name,sha=digest(folder/name),y=y,stats=stats,seed=seed,tx_hash=array_hash(wire),rx_hash=array_hash(y)))
            for i,(item,frame) in enumerate(zip(items,frames)):
                donor=frames[(i+1)%len(frames)]; stats=frame['stats']; y=torch.as_tensor(frame['y'],device=device)[None]
                ok=torch.tensor([stats['receiver_available']],device=device); snr=y.new_tensor([stats['pilot_snr_db']]); rel=y.new_tensor([stats['analog_reliability']])
                for ctrl in matrix['controls']:
                    budget(); observed=torch.as_tensor(donor['y'],device=device)[None] if ctrl=='shuffled' else y
                    if ctrl=='zero' and model.branch=='image': observed=torch.zeros_like(y)
                    out=model.receive(observed,ok,snr,rel)
                    if ctrl=='zero' and model.branch=='text': out['memory']=torch.zeros_like(out['memory'])
                    row=dict(arm=model.arm,branch=model.branch,stage=stage,cohort=cohort,example_id=item['example_id'],official_split=item['official_split'],condition=cid,control=ctrl,
                        frame_file=frame['name'],frame_sha256=frame['sha'],received_sha256=frame['rx_hash'],tx_wire_sha256=frame['tx_hash'],noise_seed=frame['seed'],stats=stats,
                        donor_example_id=items[(i+1)%len(items)]['example_id'] if ctrl=='shuffled' else None,donor_received_sha256=donor['rx_hash'] if ctrl=='shuffled' else None,
                        **score(model,item,out,backbone,tok,device,folder,cid+'__'+item['example_id']+'__'+ctrl,preview and i<c['save_images']))
                    rows.append(row); log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); log.flush()
            print('EVAL {} {} {} {} finished'.format(model.arm,stage,cohort,cid),flush=True)
    atomic_json(folder/'EVAL_COMPLETE.json',dict(rows=len(rows),rows_sha256=digest(folder/'rows.jsonl'),matrix_sha256=digest(folder/'matrix.json')))
    return rows
