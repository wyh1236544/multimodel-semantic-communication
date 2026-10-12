"""Two fresh decoder-only TRAIN overfits; never production decoder selection."""
import copy
import json
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from final_compare_v1.metrics import caption_metrics
from utils.checkpoint import load_checkpoint
from residual_fusion_v2.core import diagnostic_decision,signature,seed_for
from residual_fusion_v2.model import precompression_memory,generate_record


def diagnose(core,items,c,device,directory,receive,noise_context,budget,atomic_json,atomic_checkpoint,smoke):
    root=directory/'text_diagnostics'; root.mkdir()
    tok=items[0]['tokenizer']; memories={}; scores={}; proofs={}
    for label in ('D0_precompression','D1_T32_clean'):
        memories[label]=[]
        for item in items:
            budget()
            memory=precompression_memory(core,item,device) if label=='D0_precompression' else receive(item,None)[0]['text_memory']
            memories[label].append(memory.detach().cpu())
    with (root/'samples.jsonl').open('x',encoding='utf-8') as output:
        for label in ('D0_precompression','D1_T32_clean'):
            folder=root/label; folder.mkdir(); decoder=copy.deepcopy(core.language).to(device)
            decoder.requires_grad_(True); decoder.eval()
            before={n:signature(p) for n,p in decoder.named_parameters()}
            opt=torch.optim.AdamW(decoder.parameters(),lr=c['diagnostic_lr'],weight_decay=1e-5)
            covered=set(); start=time.monotonic()
            with (folder/'history.jsonl').open('x') as history:
                for step in range(c['diagnostic_updates']):
                    budget()
                    i=int(np.random.default_rng(seed_for(c['seed'],'text-overfit',step)).integers(len(items)))
                    item=items[i]; memory=memories[label][i].to(device)
                    opt.zero_grad(set_to_none=True)
                    loss=decoder.teacher_loss(item['target_ids'].to(device),item['target_length'].to(device),memory)
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite diagnostic CE')
                    loss.backward(); norm=torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.)
                    if not torch.isfinite(norm): raise FloatingPointError('Nonfinite diagnostic gradient')
                    covered.update(n for n,p in decoder.named_parameters() if p.grad is not None and bool(p.grad.abs().max()>0))
                    opt.step()
                    history.write(json.dumps(dict(label=label,step=step+1,image_id=item['image_id'],text_ce=float(loss.detach()),
                        initialization='identical frozen B4 language weights',schedule_seed=seed_for(c['seed'],'text-overfit',step)))+'\n')
                    if (step+1)%100==0:
                        history.flush(); print('DIAG {} update={}/{} teacherCE={:.4f}'.format(label,step+1,c['diagnostic_updates'],float(loss)),flush=True)
                    if time.monotonic()-start>c['arm_max_hours']*3600: raise TimeoutError('Diagnostic time budget')
            checkpoint=folder/'decoder_delta.pth'; atomic_checkpoint(checkpoint,dict(label=label,delta={n:v.detach().cpu() for n,v in decoder.state_dict().items()}))
            saved=load_checkpoint(checkpoint); decoder.load_state_dict(saved['delta'],strict=True)
            if any(signature(v)!=signature(saved['delta'][n]) for n,v in decoder.state_dict().items()): raise RuntimeError('Diagnostic checkpoint reload differs')
            changed=[n for n,p in decoder.named_parameters() if signature(p)!=before[n]]
            if not covered or not changed: raise RuntimeError('Diagnostic decoder was not actually trained')
            predictions=[]
            for i,item in enumerate(items):
                budget(); memory=memories[label][i].to(device); generation=generate_record(decoder,memory,tok)
                metrics=caption_metrics(item['covered_text'],generation['received_caption'])
                with torch.no_grad(): ce=float(decoder.teacher_loss(item['target_ids'].to(device),item['target_length'].to(device),memory))
                row=dict(label=label,condition='clean',image_id=item['image_id'],source_split='train',
                    source_caption_evaluation_only=item['text'],covered_reference_evaluation_only=item['covered_text'],
                    source_feature_diagnostic_not_deployable=label=='D0_precompression',input_truncated=item['truncated'],
                    teacher_forced_ce_diagnostic_only=ce,**generation,**metrics)
                predictions.append(row); output.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); output.flush()
            scores[label]=float(np.mean([r['normalized_wer'] for r in predictions]))
            proofs[label]=dict(updates=c['diagnostic_updates'],train_ids=[r['image_id'] for r in items],
                trainable_parameters=sum(p.numel() for p in decoder.parameters()),observed_gradient_names=sorted(covered),
                changed_names=changed,checkpoint_roundtrip_verified=True,initial_parameter_sha256=before,
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                wall_seconds=time.monotonic()-start,actual_covered_wer=scores[label])
            atomic_json(folder/'COMPLETE.json',proofs[label])
            if label=='D1_T32_clean':
                # D2 is the SAME fitted D1 decoder, no further noisy training.
                for condition in noise_context['conditions']:
                    for i,item in enumerate(items):
                        budget(); ctx=noise_context['make'](condition,i,len(items)); out,_,_,frame=receive(item,ctx)
                        generation=generate_record(decoder,out['text_memory'],tok)
                        row=dict(label='D2_T32_FSO',condition=condition['id'],image_id=item['image_id'],source_split='train',
                            source_caption_evaluation_only=item['text'],covered_reference_evaluation_only=item['covered_text'],
                            receiver_available=bool(frame.header_ok[0]),channel_stats=frame.last_stats,
                            diagnostic_not_validation=True,**generation,**caption_metrics(item['covered_text'],generation['received_caption']))
                        output.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); output.flush()
                        del out,frame
            del decoder,opt
    decision=diagnostic_decision(scores,c['diagnostic_wer_gate'],smoke)
    atomic_json(root/'gate.json',dict(decision,diagnostic_samples=len(items),proofs=proofs))
    print('TEXT DIAG gate={} required={} actualWER={}'.format(decision['observed_pass'],decision['required'],scores),flush=True)
    return decision
