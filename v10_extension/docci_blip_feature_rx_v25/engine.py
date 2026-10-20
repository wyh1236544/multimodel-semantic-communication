"""Only two equal-size adapters train. All pretrained and TX/RX weights frozen."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from docci_blip_feature_rx_v25.core import (atomic,line,read,rows,sha,array_sha,stream,seal,
    regimes,REGIMES,CHANNELS,donor_map,CONTROLS)
from docci_blip_feature_rx_v25.prepare import Dataset,observation
from docci_blip_feature_rx_v25.model import Adapter,self_tokens,ce,generate

def frozen_bart(a,device):
    from docci_modal_v9.backbones import load,historical_backbone
    bart,tok,identity = load('text',a,device)
    historical_backbone('text',identity['files'])
    bart.requires_grad_(False).eval()
    return bart,tok

def signature(model):
    from docci_longclip_joint_v8.backbones import signature as actual
    return actual(model)

def frozen_check(bart,before):
    if signature(bart)!=before or any(p.requires_grad or p.grad is not None for p in bart.parameters()):
        raise RuntimeError('Frozen BART changed or acquired parameter gradients')

def fit(a,c,root,device,budget):
    items = Dataset(a.cache,'train')
    if len(items)!=c['train'] or any(r['cohort']!='train' or r['official_split']!='train' for r in items.entries):
        raise ValueError('Optimizer may only read fixed TRAIN')
    bart,tok = frozen_bart(a,device); before = signature(bart)
    torch.manual_seed(42); adapter = Adapter(bart,c).to(device).eval()
    active = dict(adapter.named_parameters()); initial = {n:p.detach().cpu().clone() for n,p in active.items()}
    atomic(root/'identity.json',dict(profile=c['profile'],arm=a.arm,config=c,
        cache_index_sha256=sha(Path(a.cache)/'index.json'),train_ids=[r['example_id'] for r in items.entries],
        trainable_parameters=sum(p.numel() for p in active.values()),trainable_names=sorted(active),
        initial_parameters={n:array_sha(p.numpy()) for n,p in initial.items()},
        BART_frozen=True,BLIP_frozen=True,TX_RX_frozen=True,source_masks_at_RX=False,
        matching_gate=False,checkpoint_policy='fixed final; no validation selection',
        training_regimes=['clean','both0','text0'],loss='token-weighted CE only',
        generated_BLIP_text_used_in_training=False))
    optimizer = torch.optim.AdamW(list(active.values()),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    seen,grad_seen = set(),set(); start = time.monotonic(); torch.cuda.reset_peak_memory_stats(device)
    try:
        with (root/'history.jsonl').open('x',encoding='utf-8') as log:
            for step,epoch,ix,rs in stream(c):
                budget(); optimizer.zero_grad(set_to_none=True)
                lr = c['learning_rate']*min(1.,(step+1)/min(50,max(1,c['updates']//10)))*(.1+.9*(1-step/c['updates']))
                for group in optimizer.param_groups: group['lr'] = lr
                denom = sum(items.entries[i]['source_tokens']+16 for i in ix)
                total,count = 0.,0
                for i,regime in zip(ix,rs):
                    budget(); item = items[i]; seen.add(item['example_id'])
                    memory,gate,visual,_,_ = observation(item,regime,device)
                    aux = self_tokens(memory,gate) if a.arm=='self' else visual
                    ids = torch.as_tensor(item['ids'][None],device=device)
                    summed,n = ce(bart,adapter,memory,aux,gate,ids)
                    if not torch.isfinite(summed): raise FloatingPointError('Nonfinite CE')
                    (summed/denom).backward(); total += float(summed.detach())/denom; count += n
                if count!=denom: raise ValueError('Full-text token weighting differs')
                for name,p in active.items():
                    if p.grad is not None:
                        if not torch.isfinite(p.grad).all(): raise FloatingPointError('Nonfinite adapter gradient '+name)
                        if bool(p.grad.abs().sum()): grad_seen.add(name)
                norm = torch.nn.utils.clip_grad_norm_(list(active.values()),1.,error_if_nonfinite=True)
                optimizer.step(); elapsed = time.monotonic()-start
                record = dict(update=step+1,total_updates=c['updates'],epoch=epoch+1,
                    ids=[items.entries[i]['example_id'] for i in ix],regimes=rs,valid_target_tokens=count,
                    CE=total,lr=lr,gradient_norm=float(norm),elapsed_seconds=elapsed,
                    ETA_train_only_seconds=elapsed/(step+1)*(c['updates']-step-1),
                    peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
                line(log,record)
                if step==0 or (step+1)%20==0 or step+1==c['updates']:
                    print('V25 {} update={}/{} CE={:.4f} elapsed={:.1f}s ETA_train_only={:.1f}s'.format(
                        a.arm,step+1,c['updates'],total,elapsed,record['ETA_train_only_seconds']),flush=True)
        audits = {}
        for group,prefixes in adapter.groups().items():
            names = [n for n in active if n.startswith(prefixes)]
            audits[group] = dict(parameters=sum(active[n].numel() for n in names),
                nonzero_gradient=any(n in grad_seen for n in names),
                changed=any(not torch.equal(initial[n],active[n].detach().cpu()) for n in names))
        if not all(x['parameters'] and x['changed'] and x['nonzero_gradient'] for x in audits.values()):
            raise RuntimeError('Actual adapter learning groups failed')
        frozen_check(bart,before)
        atomic(root/'learning_audit.json',dict(groups=audits,BART_before=before,BART_after=signature(bart),
            frozen_unchanged=True,pretrained_parameters_trained=0,passed=True))
        state = {n:v.detach().cpu().clone() for n,v in adapter.state_dict().items()}
        path = root/'last.pth'
        with path.open('xb') as f:
            torch.save(dict(schema='v25-adapter',arm=a.arm,config=c,state=state,
                cache_index_sha256=sha(Path(a.cache)/'index.json'),optimizer=optimizer.state_dict(),updates=c['updates']),f)
        saved = torch.load(str(path),map_location='cpu'); adapter.load_state_dict(saved['state'],strict=True)
        if any(not torch.equal(v,adapter.state_dict()[n].detach().cpu()) for n,v in state.items()):
            raise RuntimeError('Actual saved checkpoint roundtrip differs')
        atomic(root/'TRAIN_RESULT.json',dict(updates=c['updates'],unique_training_ids=len(seen),
            checkpoint_sha256=sha(path),actual_disk_roundtrip=True,fixed_final=True,
            train_seconds=time.monotonic()-start,smoke=c['profile']=='smoke'))
        seal(root,'train')
    finally: adapter.close()

def load_adapter(bart,c,path,arm,cache,device):
    path = Path(path); info = read(path.parent/'TRAIN_RESULT.json')
    if sha(path)!=info['checkpoint_sha256']: raise ValueError('Trained adapter checksum differs')
    ck = torch.load(str(path),map_location='cpu')
    if (ck['schema']!='v25-adapter' or ck['arm']!=arm or ck['config']!=c or
            ck['updates']!=c['updates'] or ck['cache_index_sha256']!=sha(Path(cache)/'index.json')):
        raise ValueError('Checkpoint identity differs')
    model = Adapter(bart,c).to(device).eval(); model.load_state_dict(ck['state'],strict=True)
    model.requires_grad_(False)
    return model

def evaluate(a,c,root,device,budget):
    from docci_text_continue_v11.metrics import metrics
    from docci_caption_calibration_v24.core import text_audit
    items = Dataset(a.cache,'dev'); ids = [r['example_id'] for r in items.entries]; donors = donor_map(ids)
    lookup = {eid:i for i,eid in enumerate(ids)}
    bart,tok = frozen_bart(a,device); before = signature(bart)
    models = {}; start = time.monotonic()
    try:
        for arm in ('self','visual'):
            models[arm] = load_adapter(bart,c,Path(a.training_root)/('train_'+arm)/'last.pth',arm,a.cache,device)
        captions = {(r['example_id'],r['image_snr_db']):r for r in rows(Path(a.cache)/'caption_only.jsonl')}
        with np.load(Path(a.cache)/'gray.npz',allow_pickle=False) as z:
            gray = torch.as_tensor(z['visual'][None],device=device)
        atomic(root/'identity.json',dict(config=c,profile=c['profile'],dev_ids=ids,donors=donors,
            cache_index_sha256=sha(Path(a.cache)/'index.json'),regimes=list(regimes(c)),controls=CONTROLS,
            primary='both0',source_text_at_RX=False,source_image_at_RX=False,
            image_symbols=4128,text_symbols=9248,total_symbols=13376,FSO=False,
            negative_snr_is_outside_frozen_codec_and_adapter_training=True))
        count = 0
        with (root/'rows.jsonl').open('x',encoding='utf-8') as log,torch.no_grad():
            for regime in regimes(c):
                for i in range(len(items)):
                    budget(); item = items[i]; memory,gate,visual,it,tt = observation(item,regime,device)
                    donor = items[lookup[donors[item['example_id']]]]
                    wrong = torch.as_tensor(donor['visual'][it:it+1],device=device)
                    base = generate(bart,tok,memory)
                    for arm,controls in CONTROLS.items():
                        for control in controls:
                            budget(); aux = None
                            if arm=='base': result = base
                            elif arm=='caption':
                                cap = captions[item['example_id'],REGIMES[regime][0]]
                                result = dict(received_caption=cap['output_text'],caption_tokens=cap['telemetry']['new_token_ids'],
                                    caption_only=True,caption_input_sha256=cap['image_sha256'])
                            else:
                                aux = self_tokens(memory,gate) if arm=='self' else (
                                    wrong if control=='shuffled' else gray if control=='gray' else
                                    torch.zeros_like(visual) if control=='zero' else visual)
                                result = generate(bart,tok,memory,models[arm],aux,gate,control!='disabled')
                                if control=='disabled' and result!=base:
                                    raise ValueError('Disabled branch changed original generation')
                            record = dict(example_id=item['example_id'],cohort='dev',source_split=item['official_split'],
                                source_text=item['text'],source_text_sha256=item['text_sha256'],source_tokens=item['source_tokens'],
                                regime=regime,image_snr_db=REGIMES[regime][0],text_snr_db=REGIMES[regime][1],
                                arm=arm,control=control,donor_id=donors[item['example_id']] if control=='shuffled' else None,
                                memory_sha256=array_sha(memory.cpu().numpy()),
                                auxiliary_sha256=None if aux is None else array_sha(aux.cpu().numpy()),
                                image_stats=item['image_stats'][it],text_stats=item['text_stats'][tt],
                                source_text_at_RX=False,source_mask_at_RX=False,source_image_at_RX=False,
                                same_text_observation_all_controls=True,training_updates=0,
                                shared_trial_allocated_symbols=13376,
                                minimum_input_symbols=(4128 if arm=='caption' else 9248 if arm in ('base','self') or
                                    control in ('gray','zero','disabled') else 13376),
                                **result,**metrics(item['text'],result['received_caption']),
                                output_audit=text_audit(result['received_caption']))
                            line(log,record); count += 1
                    print('V25 EVAL {} {}/{} all 8 controls elapsed={:.1f}s'.format(
                        regime,i+1,len(items),time.monotonic()-start),flush=True)
        frozen_check(bart,before)
        atomic(root/'EVAL_RESULT.json',dict(rows=count,wall_seconds=time.monotonic()-start,
            BART_before=before,BART_after=signature(bart),frozen_unchanged=True,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,facts='PENDING',semantic_success=False))
        seal(root,'evaluation')
    finally:
        for model in models.values(): model.close()
