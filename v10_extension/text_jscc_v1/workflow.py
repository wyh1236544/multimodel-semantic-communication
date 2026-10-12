"""Actual greedy diagnostics and audited training; diagnostic heads never deployed."""
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from text_jscc_v1.core import seed_for,metrics,summarize,overfit_gate,digest
from text_jscc_v1.model import tensors,generate,grad_norms,legacy_wire,legacy_receive
from text_jscc_v1.legacy import precompression
from text_jscc_v1.channel import proxy


def signature(value):
    import hashlib
    a=value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str(a.dtype).encode()+json.dumps(list(a.shape)).encode()+a.tobytes()).hexdigest()


def state(model): return {n:v.detach().cpu().clone() for n,v in model.state_dict().items()}


def row(item,label,condition,generation,stats=None):
    result=dict(arm=label,image_id=item['image_id'],condition=condition,source_split=item['split'],
        source_caption_evaluation_only=item['text'],covered_reference_evaluation_only=item['covered_text'],
        source_sha256=item['source_sha256'],source_utf8_bytes=item['source_utf8_bytes'],
        source_content_tokens=item['source_content_tokens'],covered_content_tokens=item['covered_content_tokens'],
        input_truncated=item['truncated'],source_text_at_receiver=False,source_length_at_receiver=False,
        raw_utf8_wire_bytes=0)
    result.update(stats or {}); result.update(generation)
    result.update(metrics(item['text'],item['covered_text'],generation['received_caption']))
    return result


@torch.no_grad()
def native_memories(model,items,device):
    out=[]
    for item in items:
        ids,lens=tensors([item],device); wire=model.encode(ids,lens)
        out.append(model.receive(wire,wire.new_ones(1),wire.new_full((1,),40.),wire.new_ones(1)).cpu())
    return out


def memory_tests(decoder,memories,items,tok,device,label,budget):
    rows=[]; groups={}
    with torch.no_grad():
        for intervention in ('matched','zero','shuffled'):
            rr=[]
            for i,item in enumerate(items):
                budget(); donor=(i+1)%len(items)
                memory=memories[donor if intervention=='shuffled' else i].to(device)
                if intervention=='zero': memory=torch.zeros_like(memory)
                r=row(item,label,intervention,generate(decoder,memory,tok))
                r['donor_image_id']=items[donor]['image_id'] if intervention=='shuffled' else None
                r['diagnostic_only']=True; rr.append(r); rows.append(r)
            groups[intervention]=summarize(rr)
    return rows,groups


def decoder_diagnostics(core,items,tok,cfg,device,folder,budget,atomic_json,atomic_checkpoint):
    folder.mkdir(); all_rows=[]; proofs={}
    for label in ('D0_precompression','D1_compressed_clean'):
        sub=folder/label; sub.mkdir(); memories=[]
        with torch.no_grad():
            for item in items:
                budget()
                if label=='D0_precompression': memory=precompression(core,item,device)
                else:
                    wire=legacy_wire(core,item,device)
                    memory=legacy_receive(core,wire,1.,wire.new_full((1,),40.),wire.new_ones(1,dtype=torch.bool))
                memories.append(memory.detach().cpu())
        decoder=copy.deepcopy(core.language).to(device); decoder.requires_grad_(True); decoder.eval()
        initial=state(decoder); opt=torch.optim.AdamW(decoder.parameters(),lr=cfg['diagnostic_lr'],weight_decay=1e-5)
        gradients=set(); start=time.monotonic()
        with (sub/'history.jsonl').open('x') as log:
            for step in range(cfg['diagnostic_updates']):
                budget(); i=step%len(items); ids,lens=tensors([items[i]],device)
                opt.zero_grad(set_to_none=True); loss=decoder.teacher_loss(ids,lens,memories[i].to(device))
                if not torch.isfinite(loss): raise FloatingPointError('Diagnostic CE nonfinite')
                loss.backward(); norm=torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.)
                if not torch.isfinite(norm): raise FloatingPointError('Diagnostic gradient nonfinite')
                gradients.update(n for n,p in decoder.named_parameters() if p.grad is not None and bool(p.grad.abs().max()>0))
                opt.step(); log.write(json.dumps(dict(step=step+1,image_id=items[i]['image_id'],teacher_ce=float(loss)))+'\n')
                if (step+1)%100==0: log.flush(); print('TEXT DIAG {} update={}/{} CE={:.4f}'.format(label,step+1,cfg['diagnostic_updates'],float(loss)),flush=True)
        file=sub/'decoder.pth'; atomic_checkpoint(file,dict(version='text-only-diagnostic',weights=state(decoder)))
        from utils.checkpoint import load_checkpoint
        decoder.load_state_dict(load_checkpoint(file)['weights'],strict=True)
        rr,groups=memory_tests(decoder,memories,items,tok,device,label,budget); all_rows+=rr
        changed=[n for n,p in decoder.named_parameters() if not torch.equal(p.detach().cpu(),initial[n])]
        if not gradients or not changed: raise RuntimeError('Diagnostic head did not train')
        proof=dict(groups=groups,gradient_names=sorted(gradients),changed_names=changed,
            initial_sha256={n:signature(v) for n,v in initial.items()},checkpoint_sha256=digest(file),
            updates=cfg['diagnostic_updates'],wall_seconds=time.monotonic()-start,
            checkpoint_reload_verified=True,train_only_not_production=True)
        proofs[label]=proof; atomic_json(sub/'proof.json',proof)
        del decoder,opt,initial
    with (folder/'samples.jsonl').open('x',encoding='utf-8') as f:
        for r in all_rows: f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    atomic_json(folder/'summary.json',proofs)
    return proofs


def train(model,items,cfg,device,folder,epochs,lr,label,budget,atomic_json,atomic_checkpoint,bank=None,updates=None):
    folder.mkdir(exist_ok=False); before=state(model); seen={k:0. for k in grad_norms(model)}
    torch.cuda.reset_peak_memory_stats(device); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-5)
    model.eval(); start=time.monotonic(); steps=0; task_steps=0; covered=set()
    if updates is not None:
        schedule=[[(j+k)%len(items) for k in range(min(cfg['batch_size'],len(items)))] for j in range(updates)]
    else:
        schedule=[]
        for epoch in range(epochs):
            order=np.random.default_rng(seed_for(cfg['seed'],'text-training-order',epoch)).permutation(len(items)).tolist()
            schedule.extend(order[i:i+cfg['batch_size']] for i in range(0,len(order),cfg['batch_size']))
    with (folder/'history.jsonl').open('x') as log:
        for step,indices in enumerate(schedule):
            budget(); batch=[items[i] for i in indices]; ids,lens=tensors(batch,device)
            opt.zero_grad(set_to_none=True); wire=model.encode(ids,lens); losses=[]; contexts=[]
            for j,item in enumerate(batch):
                seed=seed_for(cfg['seed'],'text-train-channel',step,item['image_id'])
                rng=np.random.default_rng(seed); use_noise=bank is not None and rng.random()>=cfg['clean_anchor_fraction']
                if use_noise:
                    # Walk correlated windows within a condition. Every item gets independent residual replay.
                    conditions=sorted({r['condition'] for r in bank['records'] if r['partition']=='train'})
                    cid=conditions[(step//4)%len(conditions)]
                    choices=[r['index'] for r in bank['records'] if r['partition']=='train' and r['condition']==cid]
                    index=choices[(step*cfg['batch_size']+j)%len(choices)]
                    received,rel,snr,ok=proxy(wire[j:j+1],bank,index,seed)
                else: received,rel,snr,ok,index=wire[j:j+1],1.,40.,True,None
                memory=model.receive(received,wire.new_tensor([rel]),wire.new_tensor([snr]),wire.new_tensor([ok]))
                loss=model.language.teacher_loss(ids[j:j+1],lens[j:j+1],memory)*float(ok)
                losses.append(loss); contexts.append(dict(image_id=item['image_id'],noise_seed=seed,bank_index=index,task_loss_applied=bool(ok)))
            papr_loss=model.waveform_penalty(wire,cfg['papr_target_db'])
            loss=torch.stack(losses).mean()+cfg['papr_weight']*papr_loss
            if not torch.isfinite(loss): raise FloatingPointError('Text task loss nonfinite')
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm): raise FloatingPointError('Text gradient nonfinite')
            for key,value in grad_norms(model).items(): seen[key]+=value
            covered.update(n for n,p in model.named_parameters() if p.grad is not None and bool(p.grad.abs().max()>0))
            opt.step(); steps+=1; task_steps+=sum(r['task_loss_applied'] for r in contexts)
            rec=dict(arm=label,optimizer_step=steps,teacher_ce=float(torch.stack(losses).mean().detach()),total_loss=float(loss.detach()),papr_penalty=float(papr_loss.detach()),items=contexts,elapsed_seconds=time.monotonic()-start)
            log.write(json.dumps(rec)+'\n')
            if steps%100==0:
                log.flush(); eta=(time.monotonic()-start)/steps*(len(schedule)-steps)
                atomic_json(folder.parent/'progress.json',dict(stage=label,update=steps,total_updates=len(schedule),remaining_stage_minutes=eta/60))
                print('TEXT TRAIN {} update={}/{} CE={:.4f} elapsed={:.1f}s stageETA={:.1f}min'.format(label,steps,len(schedule),float(loss),time.monotonic()-start,eta/60),flush=True)
    changes={n:float((p.detach().cpu()-before[n]).abs().max()) for n,p in model.named_parameters()}
    required=('encoder.','compressor.','tx.','rx.','receiver.','language.decoder.')
    updates_by_group={prefix:max([value for name,value in changes.items() if name.startswith(prefix)] or [0.]) for prefix in required}
    if any(v<=0 for v in seen.values()) or any(v<=0 for v in updates_by_group.values()) or any(not any(n.startswith(prefix) for n in covered) for prefix in required):
        raise RuntimeError('Required source/compression/TX/RX/decoder did not learn')
    file=folder/'last.pth'; saved=state(model); atomic_checkpoint(file,dict(version='V10-text-only-jscc-v1',arm=label,weights=saved,config=cfg))
    from utils.checkpoint import load_checkpoint
    model.load_state_dict(load_checkpoint(file)['weights'],strict=True)
    if any(signature(v)!=signature(saved[n]) for n,v in model.state_dict().items()): raise RuntimeError('Checkpoint reload differs')
    proof=dict(arm=label,optimizer_steps=steps,source_exposures=sum(map(len,schedule)),valid_task_exposures=task_steps,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),gradient_energy=seen,
        gradient_names=sorted(covered),parameter_changes=changes,required_prefix_updates=updates_by_group,initial_sha256={n:signature(v) for n,v in before.items()},
        checkpoint_sha256=digest(file),checkpoint_reload_verified=True,wall_seconds=time.monotonic()-start,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
    atomic_json(folder/'proof.json',proof)
    return saved,proof


class SemanticScorer:
    """Evaluation-only chunked CLIP text cosine; not an independent semantic judge."""
    def __init__(self,path,device,tokenizer):
        from transformers import CLIPModel
        self.model=CLIPModel.from_pretrained(path,local_files_only=True).to(device).eval()
        self.model.requires_grad_(False); self.device=device; self.tok=tokenizer; self.cache={}

    @torch.no_grad()
    def feature(self,text):
        if text in self.cache: return self.cache[text]
        raw=self.tok(text,add_special_tokens=False,truncation=False)['input_ids']; vectors=[]; weights=[]
        for start in range(0,max(1,len(raw)),75):
            part=raw[start:start+75]; ids=[self.tok.bos_token_id]+part+[self.tok.eos_token_id]
            mask=[1]*len(ids)+[0]*(77-len(ids)); ids += [self.tok.pad_token_id]*(77-len(ids))
            v=self.model.get_text_features(input_ids=torch.tensor([ids],device=self.device),attention_mask=torch.tensor([mask],device=self.device)).float()
            vectors.append(torch.nn.functional.normalize(v,dim=-1)*max(len(part),1)); weights.append(max(len(part),1))
        result=torch.nn.functional.normalize(sum(vectors)/sum(weights),dim=-1).cpu()
        self.cache[text]=result
        return result

    def score(self,a,b): return float((self.feature(a)*self.feature(b)).sum()) if b else 0.
