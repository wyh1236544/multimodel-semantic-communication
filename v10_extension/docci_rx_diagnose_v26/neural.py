"""Pure evaluation: frozen BART/adapters; passive hook observes actual residual."""
from contextlib import contextmanager
import math
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import shift_tokens_right
from docci_rx_diagnose_v26.core import (read,rows,sha,atomic,line,array_sha,selected,
    CONTROLS,REGIMES,CHANNELS,PINS)
from docci_blip_feature_rx_v25.model import generate,self_tokens,content_gate

class Trace:
    """Gold masks weight recorded diagnostics ONLY; never change receiver outputs."""
    def __init__(self,adapter,valid,content,enabled):
        self.adapter=adapter;self.masks=dict(all=valid,content=content)
        self.enabled=adapter is not None and enabled;self.values=[];self.handles=[]
    def __enter__(self):
        if self.enabled:
            for i,layer in enumerate(self.adapter.cross):
                self.handles.append(layer.register_forward_hook(self.hook(i)))
        return self
    def hook(self,i):
        def record(module,inputs,output):
            x,_,gate=inputs
            if x.shape[:2]!=self.masks['all'].shape:raise ValueError('TF trace public shape differs')
            delta=(output-x).double();hidden=x.double()
            scopes={}
            for name,mask in self.masks.items():
                m=mask[:,:,None].double()
                hs=(hidden.square()*m).sum((1,2)).cpu().tolist()
                ds=(delta.square()*m).sum((1,2)).cpu().tolist()
                ns=mask.sum(1).cpu().tolist()
                scopes[name]=[dict(positions=int(n),hidden_l2_sq=h,delta_l2_sq=d,
                    effective_ratio=math.sqrt(d/h) if n and h else None) for h,d,n in zip(hs,ds,ns)]
            self.values.append(dict(layer=i,alpha=float(module.alpha),tanh_alpha=float(module.alpha.tanh()),
                received_content_gates=gate.detach().cpu().tolist(),scopes=scopes))
            # Returning None from the hook leaves the original output untouched.
        return record
    def __exit__(self,*args):
        for handle in self.handles:handle.remove()
        self.handles=[]

@torch.no_grad()
def teacher(bart,memory,adapter,aux,gate,ids,enabled=True):
    labels=ids.flatten(0,1).masked_fill(ids.flatten(0,1).eq(bart.config.pad_token_id),-100)
    labels=labels[:,:int(labels.ne(-100).sum(1).max())]
    masks=dict(all=labels.ne(-100),content=labels.gt(2),eos=labels.eq(2),bos=labels.eq(0))
    shifted=shift_tokens_right(labels,bart.config.pad_token_id,bart.config.decoder_start_token_id)
    @contextmanager
    def context():
        if adapter is None:yield
        else:
            with adapter.use(memory,aux,gate,enabled):yield
    with Trace(adapter,masks['all'],masks['content'],enabled) as trace,context():
        logits=bart(encoder_outputs=BaseModelOutput(last_hidden_state=memory.flatten(0,1)),
            attention_mask=torch.ones(memory.shape[:3],device=memory.device,dtype=torch.long).flatten(0,1),
            decoder_input_ids=shifted,use_cache=False,return_dict=True).logits
    if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite actual TF logits')
    losses=F.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=-100,reduction='none').reshape_as(labels).double()
    predicted=logits.argmax(-1);slots=[]
    for i in range(9):
        counts={k:int(m[i].sum()) for k,m in masks.items()}
        nll={k:float(losses[i][m[i]].sum()) for k,m in masks.items()}
        correct=int((predicted[i].eq(labels[i])&masks['content'][i]).sum())
        slots.append(dict(slot=i,counts=counts,nll=nll,content_correct=correct))
    counts={k:sum(s['counts'][k] for s in slots) for k in masks}
    sums={k:math.fsum(s['nll'][k] for s in slots) for k in masks}
    return dict(slots=slots,counts=counts,nll=sums,
        TF_all_CE=sums['all']/counts['all'],TF_content_CE=sums['content']/counts['content'] if counts['content'] else None,
        TF_content_accuracy=sum(s['content_correct'] for s in slots)/counts['content'] if counts['content'] else None,
        trace=dict(enabled=trace.enabled,layers=trace.values),teacher_forced_gold_prefix=True,
        gold_masks_used_for_measurement_only=True,source_mask_at_receiver=False,training_updates=0)

def run(a,c,root,device,budget):
    from docci_blip_feature_rx_v25.engine import frozen_bart,load_adapter,signature,frozen_check
    from docci_blip_feature_rx_v25.prepare import Dataset,observation
    from docci_text_continue_v11.metrics import metrics
    from docci_caption_calibration_v24.core import text_audit
    from docci_rx_diagnose_v26.parents import after
    parent=Path(a.parent);cache=parent/'cache';start=time.monotonic()
    choices,donors=selected(read(cache/'index.json'),c['profile'])
    sets={group:Dataset(cache,group) for group in choices}
    lookups={group:{r['example_id']:i for i,r in enumerate(ds.entries)} for group,ds in sets.items()}
    old_rows={ (r['example_id'],r['regime'],r['arm'],r['control']):r for r in rows(parent/'evaluation/rows.jsonl') }
    bart,tok=frozen_bart(a,device);before=signature(bart);models={};models_before={}
    with np.load(cache/'gray.npz',allow_pickle=False) as z:gray=torch.as_tensor(z['visual'][None],device=device)
    try:
        for arm in ('self','visual'):
            models[arm]=load_adapter(bart,read(cache/'config.json'),parent/('train_'+arm)/'last.pth',arm,cache,device)
            models_before[arm]=signature(models[arm])
        atomic(root/'CHECKPOINTS.json',dict(parent_pins=PINS,BART_before=before,
            adapters={arm:dict(tensor_sha256=models_before[arm],parameters=sum(p.numel() for p in model.parameters()),
                layers=[dict(layer=i,alpha=float(x.alpha),tanh_alpha=float(x.alpha.tanh())) for i,x in enumerate(model.cross)]) for arm,model in models.items()},
            trainable_parameters=0,BLIP_loaded=False,VAE_loaded=False,source_codec_loaded=False))
        print('V26 LOAD existing frozen BART + two V25 adapters. No BLIP/VAE or optimizer.',flush=True)
        # Actual saved-checkpoint anchors; no trace active and no decoding changes.
        anchors=[]
        for origin in choices['dev'][:2]:
            item=sets['dev'][lookups['dev'][origin['example_id']]]
            memory,gate,visual,it,tt=observation(item,'both0',device)
            donor=sets['dev'][lookups['dev'][donors[item['example_id']]]]
            wrong=torch.as_tensor(donor['visual'][it:it+1],device=device)
            for arm,ctrl in (('base','matched'),('self','matched'),('visual','matched'),('visual','shuffled')):
                budget();model=models.get(arm);aux=self_tokens(memory,gate) if arm=='self' else wrong if ctrl=='shuffled' else visual
                value=generate(bart,tok,memory,model,aux,gate)
                old=old_rows[item['example_id'],'both0',arm,ctrl]
                if any(value[k]!=old[k] for k in value):raise ValueError('Actual V25 saved-parent generation anchor differs')
                anchors.append(dict(example_id=item['example_id'],arm=arm,control=ctrl,tokens_equal=True))
        atomic(root/'ANCHORS.json',dict(actual_anchors=anchors,passed=True,training_updates=0))
        refs=[];own_count=0;calls=0;generated=0;reused=0
        with (root/'tf_rows.jsonl').open('x') as tf,(root/'generations.jsonl').open('x') as gf,torch.no_grad():
            for cohort,origins in choices.items():
                for position,origin in enumerate(origins):
                    item=sets[cohort][lookups[cohort][origin['example_id']]]
                    ref=dict(origin,gold_ids=item['ids'].tolist(),received_gates=item['gate'].tolist())
                    refs.append(ref)
                    donor=sets[cohort][lookups[cohort][donors[item['example_id']]]]
                    for regime in c['regimes']:
                        budget();memory,gate,visual,it,tt=observation(item,regime,device)
                        fresh_gate=content_gate(bart,memory)
                        if not torch.allclose(gate,fresh_gate,rtol=1e-5,atol=1e-6):raise ValueError('Cached receiver-only EOS gate differs')
                        wrong=torch.as_tensor(donor['visual'][it:it+1],device=device)
                        ids=torch.as_tensor(item['ids'][None],device=device)
                        for arm,ctrl in CONTROLS:
                            budget();model=models.get(arm)
                            aux=None if arm=='base' else self_tokens(memory,gate) if arm=='self' else (
                                wrong if ctrl=='shuffled' else gray if ctrl=='gray' else torch.zeros_like(visual) if ctrl=='zero' else visual)
                            own=dict(example_id=item['example_id'],cohort=cohort,official_split=item['official_split'],
                                source_text=item['text'],source_text_sha256=item['text_sha256'],source_tokens=item['source_tokens'],
                                regime=regime,arm=arm,control=ctrl,donor_id=donors[item['example_id']] if ctrl=='shuffled' else None,
                                memory_sha256=array_sha(memory.cpu().numpy()),auxiliary_sha256=None if aux is None else array_sha(aux.cpu().numpy()),
                                gate_sha256=array_sha(gate.cpu().numpy()),image_snr_db=REGIMES[regime][0],text_snr_db=REGIMES[regime][1],
                                training_updates=0)
                            measured=teacher(bart,memory,model,aux,gate,ids,ctrl!='disabled')
                            line(tf,dict(own,**measured));calls+=1
                            if cohort=='dev':
                                old=old_rows[item['example_id'],regime,arm,ctrl]
                                output={k:old[k] for k in ('received_caption','chunk_generated_token_ids','generated_token_ids','stopped_by_capacity')}
                                provenance='V25_VERIFIED_DEV_REUSE';reused+=1
                            else:
                                # This free-generation call has NO gold IDs, masks, source text or lengths.
                                output=generate(bart,tok,memory,model,aux,gate,ctrl!='disabled')
                                provenance='V26_NEW_TRAIN_GREEDY';generated+=1
                            line(gf,dict(own,**output,**metrics(item['text'],output['received_caption']),
                                output_audit=text_audit(output['received_caption']),generation_provenance=provenance,
                                source_text_at_RX=False,source_mask_at_RX=False,source_image_at_RX=False))
                    own_count+=1
                    print('V26 {} {}/{} TF + {} greedy; elapsed={:.1f}s'.format(cohort,position+1,len(origins),
                        'V25-reuse' if cohort=='dev' else 'new TRAIN',time.monotonic()-start),flush=True)
        atomic(root/'REFERENCES.json',dict(records=refs,donors=donors,profile=c['profile'],
            gray_visual_slice_sha256=read(cache/'gray.json')['visual_slice_sha256'],
            parent_index_sha256=sha(cache/'index.json')))
        frozen_check(bart,before)
        for arm,model in models.items():
            if signature(model)!=models_before[arm] or any(p.requires_grad or p.grad is not None for p in model.parameters()):
                raise RuntimeError('Read-only adapter changed or acquired gradients')
        after(parent)
        atomic(root/'RESULT.json',dict(profile=c['profile'],TF_calls=calls,new_TRAIN_greedy=generated,
            reused_DEV_greedy=reused,wall_seconds=time.monotonic()-start,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            BART_before=before,BART_after=signature(bart),adapter_tensors_unchanged=True,
            adapter_before=models_before,adapter_after={arm:signature(model) for arm,model in models.items()},
            optimizer_created=False,training_updates=0,semantic_success=False,facts='PENDING'))
    finally:
        for model in models.values():model.close()
