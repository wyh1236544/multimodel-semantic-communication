"""Privileged gold-prefix diagnosis ONLY; never a transmitted recovery score."""
import json
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput
from docci_reconstruction_v10.data import batch
from docci_reconstruction_v10.model import prediction
from docci_reconstruction_v10.channel import clean
from docci_text_continue_v11.core import atomic_json,digest
from docci_text_continue_v11.evaluate import ah


def mean(value,mask): return float(value[mask].mean()) if bool(mask.any()) else None


@torch.no_grad()
def diagnose(model,items,c,tok,device,budget,folder,free_rows):
    folder.mkdir(exist_ok=False); index={r['example_id']:r for r in free_rows if r['control']=='matched'}; rows=[]
    for i,item in enumerate(items):
        budget(); src=batch(items,[i],'text',device); wire=model.encode(src)
        if ah(wire[0].float().cpu().numpy())!=index[item['example_id']]['tx_wire_sha256']: raise ValueError('Diagnostic received wire differs')
        output=model.receive(*clean(wire)); ids,hidden=model.sources(src); mask=ids.ne(tok.pad_token_id)
        memory=output['memory'].flatten(0,1); rxmask=output['receiver_mask'].flatten(0,1)
        tf,labels=prediction(model.bart,ids,mask,memory,rxmask)
        oracle,oracle_labels=prediction(model.bart,ids,mask,hidden.float(),mask)
        if not torch.equal(labels,oracle_labels): raise ValueError('Native/gold target differs')
        content=labels.ne(-100)&labels.ne(tok.bos_token_id)&labels.ne(tok.eos_token_id)
        eosmask=labels.eq(tok.eos_token_id); valid=labels.ne(-100)
        ce=F.cross_entropy(tf.transpose(1,2),labels,reduction='none',ignore_index=-100)
        oce=F.cross_entropy(oracle.transpose(1,2),labels,reduction='none',ignore_index=-100)
        prob=tf.softmax(-1)[...,tok.eos_token_id]
        # Compare only positions with an actual self-generated prefix, up to its FIRST EOS.
        sequences=index[item['example_id']]['chunk_generated_token_ids']; length=labels.shape[1]
        prefix=torch.full((9,length),tok.pad_token_id,dtype=torch.long,device=device); available=torch.zeros_like(content)
        for j,seq in enumerate(sequences):
            end=next((k for k,t in enumerate(seq[1:],1) if t==tok.eos_token_id),len(seq)-1)
            count=min(length,end); prefix[j,:count]=torch.tensor(seq[:count],device=device); available[j,:count]=True
        # Actual generated-prefix availability, not gold length or a PAD-value filter.
        # A PAD token chosen before EOS stays in the prefix, as in unrestricted generation.
        free=model.bart(encoder_outputs=BaseModelOutput(last_hidden_state=memory),attention_mask=rxmask.long(),
            decoder_input_ids=prefix,decoder_attention_mask=available.long(),use_cache=False,return_dict=True).logits
        fce=F.cross_entropy(free.transpose(1,2),labels,reduction='none',ignore_index=-100); common=available&content
        slots=[]
        for j in range(9):
            gold_content=int(content[j].sum()); epos=int(torch.nonzero(eosmask[j],as_tuple=False)[0,0].item())
            slots.append(dict(slot=j,gold_content_BPE=gold_content,
                TF_content_CE=mean(ce[j],content[j]),native_TF_content_CE=mean(oce[j],content[j]),TF_content_accuracy=mean(tf[j].argmax(-1).eq(labels[j]).float(),content[j]),
                TF_pEOS_before_gold_end=mean(prob[j],content[j]),TF_argmax_early_EOS=int((tf[j].argmax(-1).eq(tok.eos_token_id)&content[j]).sum()),
                TF_pEOS_at_gold_end=float(prob[j,epos]),
                common_content_positions=int(common[j].sum()),TF_CE_common=mean(ce[j],common[j]),free_prefix_CE_common=mean(fce[j],common[j])))
        row=dict(example_id=item['example_id'],source_text_sha256=item['text_sha256'],source_tokens=item['source_tokens'],
            actual_wire_sha256=index[item['example_id']]['tx_wire_sha256'],diagnostic_only=True,
            gold_prefix_is_privileged_NOT_actual_RX=True,native_source_memory_NOT_transmitted=True,
            slot_count=9,TF_total_CE=mean(ce,valid),TF_content_CE=mean(ce,content),native_TF_content_CE=mean(oce,content),
            TF_content_accuracy=mean(tf.argmax(-1).eq(labels).float(),content),
            common_content_positions=int(common.sum()),TF_CE_common=mean(ce,common),free_prefix_CE_common=mean(fce,common),slots=slots)
        rows.append(row)
    with (folder/'rows.jsonl').open('x',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    atomic_json(folder/'DIAGNOSTIC_COMPLETE.json',dict(rows=len(rows),rows_sha256=digest(folder/'rows.jsonl'),
        scope='post-hoc positional conditional diagnosis; source/gold prefixes only here, NEVER actual receiver inputs or optimizer loss'))
    return rows
