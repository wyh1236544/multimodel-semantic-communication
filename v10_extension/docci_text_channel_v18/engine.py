"""Two-view CE: train BOTH TX compressor and RX reconstructor; frozen BART."""
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from docci_text_channel_v18.core import (ARMS, PARAMETERS, PARENT_SHA, atomic, sha, read, lines,
    batches, channel_for, trial_seed, trials, seal)
from docci_text_channel_v18.channel import transmit, objective, tensor_sha

def state_hash(model):
    from docci_baseline_v14.core import parameter_hash
    return parameter_hash(model)

def backbone_hash(model):
    from docci_longclip_joint_v8.backbones import signature
    return signature(model.bart)

def make(args,device,checkpoint=None):
    from docci_modal_v9.backbones import load,historical_backbone
    from docci_baseline_v14.model import TextModel
    bart,tok,identity=load('text',args,device)
    historical_backbone('text',identity['files'])
    if identity['files']!=read(Path(args.features)/'backbone.json')['files']: raise ValueError('Actual offline BART files differ from original feature cache backbone')
    parent=Path(args.features)/'released64/train/last.pth'
    if sha(parent)!=PARENT_SHA: raise ValueError('Approved V14 checkpoint differs before deserialization')
    record=torch.load(str(parent),map_location='cpu')
    if (record.get('schema'),record.get('task'),record.get('arm'),record.get('updates'))!=('v14-checkpoint','text','released64',5120): raise ValueError('Wrong parent schema')
    torch.manual_seed(42)
    model=TextModel(bart,'released64').to(device)
    model.restore(record['state']); del record
    initial=state_hash(model)
    if checkpoint is not None:
        info=read(Path(checkpoint).parent/'TRAIN_COMPLETE.json')
        if sha(checkpoint)!=info['checkpoint_sha256']: raise ValueError('V18 checkpoint hash differs')
        record=torch.load(str(checkpoint),map_location='cpu')
        if record.get('schema')!='v18-text-checkpoint' or record['parent_sha256']!=PARENT_SHA: raise ValueError('Wrong V18 checkpoint')
        model.restore(record['state']); del record
    active={n:p for n,p in model.named_parameters() if p.requires_grad}
    if sum(p.numel() for p in active.values())!=PARAMETERS or any('bart.' in n for n in active): raise ValueError('Expected only original 25618432 codec parameters trainable')
    model.eval()
    return model,tok,identity,initial

def fit(args,c,items,root,device,budget,smoke):
    if args.arm not in ARMS: raise ValueError('Unknown arm')
    root.mkdir(exist_ok=False)
    model,tok,backbone,initial=make(args,device)
    frozen=backbone_hash(model)
    active={n:p for n,p in model.named_parameters() if p.requires_grad}
    start_state=model.saved(); grad_seen=set()
    total=2 if smoke else math.ceil(len(items)/c['accumulation'])*c['epochs']
    atomic(root/'identity.json',dict(arm=args.arm,smoke=smoke,parent_sha256=PARENT_SHA,
        config=c,initial_hashes=initial,trainable_parameters=PARAMETERS,
        TRAIN_ids=[r['example_id'] for r in items.entries],updates=total,
        source_features='immutable native BART encoder cache; NOT frozen TX wire',
        receiver_inputs='observed 18432 real symbols only',two_forward_backward_views_per_description=True,
        BART_frozen=True,exact_optimizer_resume=False,checkpoint_policy='fixed final',new_optimizer=True))
    optimizer=torch.optim.AdamW(list(active.values()),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    stream=list(batches(len(items),c['accumulation'],c['epochs'],c['seed']))[:total]
    started=time.monotonic(); torch.cuda.reset_peak_memory_stats(device)
    seen=set()
    with (root/'history.jsonl').open('x',encoding='utf-8') as log:
        for step,(epoch,indices) in enumerate(stream):
            budget(); optimizer.zero_grad(set_to_none=True)
            lr=c['learning_rate']*min(1.,(step+1)/min(100,max(1,total//10)))*(.1+.9*(1-step/total))
            for g in optimizer.param_groups: g['lr']=lr
            denominator=sum(items.entries[i]['source_tokens']+16 for i in indices)
            clean_ce,second_ce=0.,0.; observations=[]
            for i in indices:
                row=items[i]; eid=row['example_id']; seen.add(eid)
                ids=torch.as_tensor(row['ids'][None],device=device)
                hidden=torch.as_tensor(row['hidden'][None],device=device)
                channel=None if args.arm=='clean_continue' else channel_for(c,epoch,eid)
                stats=[]
                # Two separately recomputed graphs, even in the clean control.
                # No retain_graph, cached TX features, or skipped encoder gradient.
                for view,snr,weight in [('clean',None,c['clean_weight']),('second',channel,1-c['clean_weight'])]:
                    budget(); wire=model.encode(ids,hidden)
                    observed,s=transmit(wire,snr,trial_seed(c,'train',epoch,eid,snr))
                    summed,n=objective(model,ids,hidden,observed)
                    if n!=row['source_tokens']+16 or not torch.isfinite(summed): raise FloatingPointError('Invalid CE/token denominator')
                    (summed*(weight/denominator)).backward()
                    if view=='clean': clean_ce+=float(summed.detach())/denominator
                    else: second_ce+=float(summed.detach())/denominator
                    stats.append(dict(view=view,weight=weight,**s))
                observations.append(dict(example_id=eid,views=stats))
            for n,p in active.items():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all(): raise FloatingPointError('Nonfinite gradient '+n)
                    if bool(p.grad.abs().sum()): grad_seen.add(n)
            norm=torch.nn.utils.clip_grad_norm_(list(active.values()),1.,error_if_nonfinite=True)
            optimizer.step(); elapsed=time.monotonic()-started
            record=dict(update=step+1,total_updates=total,epoch=epoch+1,batch_ids=[items.entries[i]['example_id'] for i in indices],
                valid_tokens_per_view=denominator,clean_CE=clean_ce,second_CE=second_ce,
                loss=c['clean_weight']*clean_ce+(1-c['clean_weight'])*second_ce,
                lr=lr,gradient_norm=float(norm),observations=observations,
                elapsed_seconds=elapsed,ETA_this_arm_train=elapsed/(step+1)*(total-step-1),
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
            if step==0 or (step+1)%c['log_every']==0 or step+1==total:
                atomic(root/'progress.json',record)
                print('V18 {} {}/{} cleanCE={:.4f} secondCE={:.4f} elapsed={:.1f}s ETA_this_arm={:.1f}s'.format(args.arm,step+1,total,clean_ce,second_ce,elapsed,record['ETA_this_arm_train']),flush=True)
    audit={}
    for group,prefixes in model.groups().items():
        names=[n for n in active if n.startswith(prefixes)]
        # saved() keys omit the outer codec prefix.
        audit[group]=dict(parameters=sum(active[n].numel() for n in names),
            nonzero_gradient=any(n in grad_seen for n in names),
            changed=any(not torch.equal(start_state[n[len('codec.'):]],active[n].detach().cpu()) for n in names))
    if not all(v['parameters'] and v['nonzero_gradient'] and v['changed'] for v in audit.values()): raise RuntimeError('TX/RX codec failed real learning gate')
    after=backbone_hash(model)
    if frozen!=after or any(p.requires_grad or p.grad is not None for p in model.bart.parameters()): raise RuntimeError('Frozen BART changed')
    atomic(root/'learning_audit.json',dict(groups=audit,passed=True))
    atomic(root/'backbone_audit.json',dict(before=frozen,after=after,unchanged=True))
    saved=model.saved()
    with (root/'last.pth').open('xb') as f:
        torch.save(dict(schema='v18-text-checkpoint',arm=args.arm,parent_sha256=PARENT_SHA,
            config=c,state=saved,optimizer=optimizer.state_dict(),updates=total,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device)),f)
    actual=torch.load(str(root/'last.pth'),map_location='cpu'); model.restore(actual['state'])
    restored=model.saved()
    if any(not torch.equal(v,restored[k]) for k,v in saved.items()): raise RuntimeError('Saved checkpoint roundtrip differs')
    if sha(Path(args.features)/'released64/train/last.pth')!=PARENT_SHA: raise ValueError('Historical parent changed')
    atomic(root/'TRAIN_COMPLETE.json',dict(updates=total,unique_ids_seen=len(seen),
        checkpoint_sha256=sha(root/'last.pth'),actual_disk_roundtrip=True,fixed_final=True,
        training_seconds=time.monotonic()-started,smoke=smoke))
    seal(root)

def evaluate(args,c,items,root,device,budget,smoke):
    from docci_baseline_v14.model import decode
    from docci_text_continue_v11.metrics import metrics
    root.mkdir(exist_ok=False)
    model,tok,backbone,initial=make(args,device,args.checkpoint)
    model.requires_grad_(False).eval(); frozen=backbone_hash(model); before=state_hash(model)
    old={r['example_id']:r for r in lines(Path(args.features)/'released64/validation/rows.jsonl') if r['strategy']=='greedy' and r['control']=='matched'}
    schedule=list(trials(c,smoke)); total=len(items)*len(schedule)
    atomic(root/'identity.json',dict(stage='evaluation',arm=args.arm,smoke=smoke,config=c,
        checkpoint_sha256=PARENT_SHA if args.checkpoint is None else sha(args.checkpoint),
        parent_sha256=PARENT_SHA,ids=[r['example_id'] for r in items.entries],
        trials=[list(x) for x in schedule],rows=total,FSO=False,digital='NOT_RUN',facts='PENDING'))
    started=time.monotonic(); count=0
    with (root/'rows.jsonl').open('x',encoding='utf-8') as log,torch.no_grad():
        for snr,repeat,control in schedule:
            for i in range(len(items)):
                budget(); row=items[i]; eid=row['example_id']
                ids=torch.as_tensor(row['ids'][None],device=device);hidden=torch.as_tensor(row['hidden'][None],device=device)
                wire=model.encode(ids,hidden)
                observed,own=transmit(wire,snr,trial_seed(c,'validation',0,eid,snr,repeat))
                delivered=observed; donor=None; delivered_stats=own
                if control=='shuffled':
                    r=items[(i+1)%len(items)];donor=r['example_id']
                    donor_wire=model.encode(torch.as_tensor(r['ids'][None],device=device),torch.as_tensor(r['hidden'][None],device=device))
                    delivered,delivered_stats=transmit(donor_wire,snr,trial_seed(c,'validation',0,donor,snr,repeat))
                elif control=='zero': delivered=torch.zeros_like(observed)
                output=decode(model.bart,model.receive(delivered),tok,'greedy')
                if args.arm=='frozen' and snr is None and control=='matched' and output['chunk_generated_token_ids']!=old[eid]['chunk_generated_token_ids']: raise ValueError('Original V14 clean baseline tokens changed')
                record=dict(example_id=eid,source_text=row['text'],source_text_sha256=row['text_sha256'],
                    source_tokens=row['source_tokens'],official_split='qual_dev',arm=args.arm,
                    snr_db=snr,repeat=repeat,control=control,donor_id=donor,
                    own_stats=own,delivered_stats=delivered_stats,received_payload_sha256=tensor_sha(delivered),
                    input_truncated=False,**output,**metrics(row['text'],output['received_caption']))
                log.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');log.flush();count+=1
                if count==1 or count%20==0 or count==total:
                    elapsed=time.monotonic()-started
                    atomic(root/'progress.json',dict(rows=count,total=total,elapsed_seconds=elapsed,ETA_this_evaluation=elapsed/count*(total-count)))
                    print('V18 EVAL {} {}/{} snr={} repeat={} control={} ETA_this_eval={:.1f}s'.format(args.arm,count,total,snr,repeat,control,elapsed/count*(total-count)),flush=True)
    if before!=state_hash(model) or frozen!=backbone_hash(model): raise ValueError('Evaluation mutated model')
    atomic(root/'EVAL_COMPLETE.json',dict(rows=count,neural_generation_executed=True,
        unchanged=True,elapsed_seconds=time.monotonic()-started))
    seal(root)
