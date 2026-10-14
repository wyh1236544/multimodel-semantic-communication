"""Same legacy public clean frame/greedy controls; exact faster metric arithmetic."""
import hashlib
import json
import numpy as np
import torch
from docci_reconstruction_v10.data import batch
from docci_reconstruction_v10.model import generate
from docci_reconstruction_v10.channel import resource
from docci_text_continue_v11.core import atomic_json,digest,geometry
from docci_text_continue_v11.metrics import metrics


def ah(value): return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


@torch.no_grad()
def evaluate(model,items,c,tok,device,budget,folder,stage,cohort):
    folder.mkdir(exist_ok=False); (folder/'frames').mkdir(); rows=[]; wires=[]; res=resource(c,'T1_chunks')
    if len(items)<2: raise ValueError('Shuffled control needs two distinct sources')
    for i in range(len(items)):
        budget(); wire=model.encode(batch(items,[i],'text',device))[0].float().cpu().numpy()
        if wire.shape!=(geometry(c,'T1_chunks')['total_reals'],) or not np.isfinite(wire).all() or abs(float(np.mean(wire**2))-1)>1e-4:
            raise ValueError('Public clean wire shape/power differs')
        wires.append(wire)
    matrix=dict(arm='T1_chunks',branch='text',stage=stage,cohort=cohort,example_ids=[x['example_id'] for x in items],
        controls=['matched','zero','shuffled'],conditions=['clean'],resources=res,
        source_count_or_mask_at_RX=False,Common_used=False,failures_retained=True,
        zero_scope='post-reconstruction BART memory',control_scope='same stored received frames; no retransmission',
        clean_snr40_is_sentinel=True,metric_implementation='exact bit-vector Levenshtein; unchanged definitions')
    atomic_json(folder/'matrix.json',matrix); frames=[]
    for item,wire in zip(items,wires):
        file='frames/clean__'+item['example_id']+'.npz'
        with (folder/file).open('xb') as f: np.savez_compressed(f,observed=wire,available=np.asarray(True,np.bool_),
            pilot_snr_db=np.asarray(40.,np.float32),reliability=np.asarray(1.,np.float32))
        frames.append(dict(file=file,sha=digest(folder/file),hash=ah(wire)))
    with (folder/'rows.jsonl').open('x',encoding='utf-8') as log:
        for i,item in enumerate(items):
            donor=(i+1)%len(items); wire=torch.as_tensor(wires[i],device=device)[None]
            stats=dict(receiver_available=True,pilot_snr_db=40.,analog_reliability=1.,resources=res,
                clean_sentinel_not_measured_snr=True,payload_complex_symbols=9216,analog_repeats=1,
                raw_utf8_wire_bytes=0,pre_fec_ber=None,post_fec_ber=None)
            for control in matrix['controls']:
                budget(); observed=torch.as_tensor(wires[donor],device=device)[None] if control=='shuffled' else wire
                output=model.receive(observed,torch.tensor([True],device=device),wire.new_tensor([40.]),wire.new_tensor([1.]))
                if control=='zero': output['memory']=torch.zeros_like(output['memory'])
                result=generate(model,output,tok)
                row=dict(arm='T1_chunks',branch='text',stage=stage,cohort=cohort,example_id=item['example_id'],
                    official_split=item['official_split'],condition='clean',control=control,
                    frame_file=frames[i]['file'],frame_sha256=frames[i]['sha'],received_sha256=frames[i]['hash'],
                    tx_wire_sha256=frames[i]['hash'],noise_seed=None,stats=stats,
                    donor_example_id=items[donor]['example_id'] if control=='shuffled' else None,
                    donor_received_sha256=frames[donor]['hash'] if control=='shuffled' else None,
                    source_text=item['text'],source_text_sha256=item['text_sha256'],source_tokens=item['source_tokens'],
                    input_truncated=False,received_caption=result['text'],**result,**metrics(item['text'],result['text']))
                rows.append(row); log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); log.flush()
            if i==0 or (i+1)%10==0 or i+1==len(items):
                print('EVAL {} {} {}/{} (all3 controls)'.format(stage,cohort,i+1,len(items)),flush=True)
    atomic_json(folder/'EVAL_COMPLETE.json',dict(rows=len(rows),rows_sha256=digest(folder/'rows.jsonl'),matrix_sha256=digest(folder/'matrix.json')))
    return rows
