"""CPU packet simulation and separate frozen-GPU extrapolation; no optimizer."""
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from docci_digital_compare_v19.common import read,lines,atomic,progress,sha,score,seal,trials,CHECKPOINTS

def capacity(sources,c):
    from docci_digital_compare_v19.digital import compressions,CODECS
    lengths=[]
    for r in sources:
        values=compressions(r['text'])
        d={k:len(v)*8 for k,v in zip(CODECS,values)};d['best_stdlib']=min(d.values())
        lengths.append(dict(example_id=r['example_id'],**d))
    table=[]
    for n in c['capacity_budgets']:
        for snr in c['snr_db']:
            for gap in c['capacity_gaps_db']:
                cap=n*math.log2(1+10**((snr-gap)/10))
                table.append(dict(symbol_budget=n,snr_db=snr,assumed_gap_db=gap,budget_bits=cap,
                    length_fit_fraction={k:sum(r[k]<=cap for r in lengths)/len(lengths) for k in list(CODECS)+['best_stdlib']}))
    return dict(lengths=lengths,table=table,actual_transmission_success=False,
                interpretation='length-fit optimistic asymptotic budget only; no header/pilots/finite-block error, not measured success',
                gap_2db='illustrative sensitivity assumption, not calibrated code performance',
                compression_libraries=dict(zlib=__import__('zlib').ZLIB_VERSION))

def digital(prepared,root,c,budget,smoke=False):
    from docci_text_channel_v18.core import noise,trial_seed
    from docci_digital_compare_v19.digital import evaluate_packet
    sources=read(Path(prepared)/'sources.json')[:2 if smoke else 100]
    root.mkdir(exist_ok=False);started=time.monotonic();schedule=list(trials(c,smoke));count=0
    total=len(sources)*len(schedule)*len(c['profiles'])
    atomic(root/'identity.json',dict(kind='digital',smoke=smoke,ids=[r['example_id'] for r in sources],
        config=c,trials=[list(x) for x in schedule],total_rows=total,
        source_sha256=sha(Path(prepared)/'sources.json'),source_size_is_not_RX_input=True))
    if not smoke:atomic(root/'capacity.json',capacity(sources,c))
    points=[]
    with (root/'rows.jsonl').open('x',encoding='utf-8') as out:
        for snr,repeat in schedule:
            for row in sources:
                epsilon=noise(18432,snr,trial_seed(c,'validation',0,row['example_id'],snr,repeat))
                for profile in c['profiles']:
                    budget();result,y=evaluate_packet(row['text'],profile,c,snr,epsilon)
                    result.update(example_id=row['example_id'],arm=profile,snr_db=snr,repeat=repeat,
                        provenance='V19_actual_packet',**score(row['text'],result['received_caption']))
                    out.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');out.flush();count+=1
                    if row==sources[0] and repeat==0:
                        p=c['pilot_symbols']+c['header_symbols']
                        points.append(dict(arm=profile,snr_db=snr,example_id=row['example_id'],
                            received_IQ=[[float(z.real),float(z.imag)] for z in y[p:p+256]],
                            equalization='none: h=1 known symbol-level AWGN',selection='fixed first ID, first256body symbols, repeat0'))
                    if count==1 or count%50==0 or count==total:
                        elapsed=time.monotonic()-started
                        progress(root/'progress.json',dict(rows=count,total=total,elapsed_seconds=elapsed,ETA_seconds=elapsed/count*(total-count)))
                        print('V19 DIGITAL {}/{} snr={} ETA={:.1f}s'.format(count,total,snr,elapsed/count*(total-count)),flush=True)
    atomic(root/'constellations.json',points)
    atomic(root/'COMPLETE.json',dict(rows=count,simulation_seconds=time.monotonic()-started,
        actual_packet_receiver_executed=True,no_training=True,smoke=smoke))
    seal(root)

def negative(prepared,root,c,budget,smoke,bart_path):
    import torch
    from types import SimpleNamespace
    from docci_text_channel_v18.engine import make,state_hash,backbone_hash
    from docci_text_channel_v18.channel import transmit
    from docci_text_channel_v18.data import Items
    from docci_text_channel_v18.core import trial_seed
    from docci_baseline_v14.model import decode
    identity=read(Path(prepared)/'identity.json');v18=Path(identity['v18_root'])
    device=torch.device('cuda:0');entries=read(Path(prepared)/'sources.json')[:2 if smoke else 100]
    items=Items(entries,1);args=SimpleNamespace(features=identity['features'],bart_path=bart_path)
    old={(r['arm'],r['example_id'],r['snr_db'],r['repeat']):r for r in lines(Path(prepared)/'old_rows.jsonl')}
    # Smoke verifies the exact previous saved models/output, plus the new OOD path.
    schedule=[(None,0),(0,0),(-5,0)] if smoke else list(trials(c,negative=True))
    root.mkdir(exist_ok=False);started=time.monotonic();count=0;total=len(entries)*len(schedule)*3
    atomic(root/'identity.json',dict(kind='negative',smoke=smoke,config=c,checkpoints=CHECKPOINTS,
        ids=[r['example_id'] for r in entries],trials=[list(x) for x in schedule],total_rows=total,
        source_sha256=sha(Path(prepared)/'sources.json'),training_executed=False,
        negative_SNR_was_NOT_in_training=True,physical_GPU=1))
    audits={}
    with (root/'rows.jsonl').open('x',encoding='utf-8') as out,torch.no_grad():
        for arm,h in CHECKPOINTS.items():
            budget();cp=None if arm=='frozen' else v18/('train_'+arm)/'last.pth'
            if cp is not None and sha(cp)!=h:raise ValueError('Pinned checkpoint differs before deserialization')
            model,tok,_,_=make(args,device,cp);model.requires_grad_(False).eval()
            before=state_hash(model);backbone=backbone_hash(model)
            if any(p.requires_grad for p in model.parameters()):raise ValueError('Inference-only model must be frozen')
            for snr,repeat in schedule:
                for i in range(len(items)):
                    budget();row=items[i]
                    ids=torch.as_tensor(row['ids'][None],device=device);hidden=torch.as_tensor(row['hidden'][None],device=device)
                    wire=model.encode(ids,hidden);observed,stats=transmit(wire,snr,trial_seed(c,'validation',0,row['example_id'],snr,repeat))
                    prediction=decode(model.bart,model.receive(observed),tok,'greedy')
                    if snr in (None,0):
                        saved=old[(arm,row['example_id'],snr,repeat)]
                        if prediction['chunk_generated_token_ids']!=saved['chunk_generated_token_ids'] or prediction['received_caption']!=saved['received_caption']:raise ValueError('Real V18 saved-output anchor differs')
                    result=dict(example_id=row['example_id'],arm=arm,snr_db=snr,repeat=repeat,
                        own_stats=stats,epsilon_sha256=stats['epsilon_sha256'],provenance='V19_frozen_OOD_inference',
                        **prediction,**score(row['text'],prediction['received_caption']))
                    out.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');out.flush();count+=1
                    if count==1 or count%20==0 or count==total:
                        elapsed=time.monotonic()-started
                        progress(root/'progress.json',dict(rows=count,total=total,elapsed_seconds=elapsed,ETA_seconds=elapsed/count*(total-count)))
                        print('V19 INFERENCE {} {}/{} snr={} ETA={:.1f}s'.format(arm,count,total,snr,elapsed/count*(total-count)),flush=True)
            if state_hash(model)!=before or backbone_hash(model)!=backbone:raise ValueError('Inference mutated weights')
            if cp is not None and sha(cp)!=h:raise ValueError('Historical checkpoint changed')
            audits[arm]=dict(checkpoint_sha256=h,codec_unchanged=True,BART_unchanged=True,
                            all_parameters_frozen=True,anchor_rows_checked=4 if smoke else 0)
            del model,tok,ids,hidden,wire,observed;torch.cuda.empty_cache()
    atomic(root/'COMPLETE.json',dict(rows=count,inference_seconds=time.monotonic()-started,
        audits=audits,neural_generation_executed=True,no_training=True,smoke=smoke))
    seal(root)
