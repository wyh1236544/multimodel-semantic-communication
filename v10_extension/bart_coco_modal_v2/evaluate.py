"""Actual compact full-waveform evaluation, all failures and independent task fields."""
import json
import time
import torch
from bart_coco_modal_v2.core import geometry,seed_for,tensor_signature,metrics
from bart_coco_modal_v2.data import source_wire
from bart_coco_modal_v2.channel import clean,transport
from fso_fluctuating_v2.core import paths
from optical_realism_v3.constellation import constellation_svg
from utils.metrics import batch_image_metrics
from eval_v10 import save_image


@torch.no_grad()
def tasks(model,received,item,tokenizer,vae,device):
    out=model.receive_payload(*received); available=bool(received[3][0])
    row=dict(received_caption=None,psnr=None,ssim=None,input_truncated=None,stopped_by_capacity=None)
    image=None
    if model.modality!='image_only':
        result=model.generate(out['text_memory'],tokenizer,available=available)
        row.update(received_caption=result['text'],stopped_by_capacity=result['stopped_by_capacity'],
            input_truncated=item['truncated'],covered_text=item['covered_text'],
            **metrics(item['text'],item['covered_text'],result['text']))
    if model.modality!='text_only':
        scale=float(getattr(vae.config,'scaling_factor',.18215))
        image=((vae.decode(out['latent']/scale).sample+1)/2).clamp(0,1) if available else torch.zeros_like(item['image'].to(device))
        scores=batch_image_metrics(image,item['image'].to(device))
        row.update(psnr=float(scores['psnr'][0]),ssim=float(scores['ssim'][0]))
    return row,image


@torch.no_grad()
def evaluate(model,items,cfg,cc,device,root,tokenizer,vae,budget,atomic_json,writer,events):
    mode=model.modality; total=0
    for condition in [None]+cc['conditions']:
        cid='clean' if condition is None else condition['id']
        weather=None if condition is None else paths(cc,condition['weather'],len(items),('coco-modal-validation',condition['weather']))
        for i,item in enumerate(items):
            budget(); start=time.monotonic(); seed=seed_for(cfg['seed'],item['image_id'],cid,'coco-modal-eval')
            x=source_wire(model,item,device); t=None
            if condition is None:
                received=clean(x); stats=dict(receiver_available=True,clean_control_no_physical_transmission=True,
                    physical_symbol_snr_db=None,pilot_snr_db=None,scheduled_seconds=None,launch_signal_energy_j=None,
                    pre_fec_ber=None,post_fec_ber=None,raw_utf8_wire_bytes=0)
            else:
                t=transport(cc,condition,weather[i],seed,mode); pilot=float(t.probe())
                y,rel,stats=t.continuous(x[0].cpu().numpy()); ok=bool(stats['analog_receiver_available'])
                received=(torch.as_tensor(y[None],device=device,dtype=x.dtype),x.new_tensor([rel]),x.new_tensor([pilot]),torch.tensor([ok],device=device))
                stats.update(receiver_available=ok,pilot_snr_db=pilot,clean_control_no_physical_transmission=False)
            score,image=tasks(model,received,item,tokenizer,vae,device)
            key='{}__{}__{}'.format(mode,cid,item['image_id'])
            row=dict(image_id=item['image_id'],annotation_id=item['annotation_id'],source_sha256=item['source_sha256'],
                source_split='validation',mode=mode,condition=cid,frame_key=key,noise_seed=seed,
                source_text=item['text'],source_caption_reference_only=mode=='image_only',
                text_used_for_source=item['text_used_for_source'],image_used_for_source=item['image_used_for_source'],
                source_utf8_bytes=item['source_utf8_bytes'] if mode!='image_only' else None,
                source_rgb8_bytes=item.get('source_rgb8_bytes'),geometry=geometry(mode),stats=stats,
                payload_sha256=tensor_signature(x),received_sha256=tensor_signature(received[0]),
                offline_seconds=time.monotonic()-start,**score)
            if i<cfg['save_images']:
                if image is not None:
                    source_name='images/source_'+str(item['image_id'])+'.png'
                    if not (root/source_name).exists(): save_image(item['image'][0],root/source_name)
                    name='images/'+key+'.png'; save_image(image[0].cpu(),root/name)
                    row.update(image_file=name,source_image_file=source_name)
                if t is not None:
                    name='constellations/'+key+'.svg'; constellation_svg(t.constellations,root/name); row['constellation_file']=name
            if t is not None:
                for event in t.events: events.write(json.dumps(dict(event,frame_key=key),allow_nan=False)+'\n')
            writer.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); writer.flush(); events.flush(); total+=1
        atomic_json(root/'progress.json',dict(stage='compact_full_waveform_eval',mode=mode,condition=cid,
            mode_rows=total,mode_expected_rows=len(items)*(len(cc['conditions'])+1)))
        print('COCO MODAL EVAL {} {} mode_rows={}'.format(mode,cid,total),flush=True)
    return total


@torch.no_grad()
def common_controls(model,items,cfg,device,root,tokenizer,vae,budget):
    # Fixed observed symbols and resource: no new transmission, no original-text RX input.
    subset=items[:min(cfg['save_images'],len(items))]
    if len(subset)<2: return []
    wires=[source_wire(model,item,device) for item in subset]; rows=[]
    for control in ('matched','zero_common','shuffled_common'):
        for i,item in enumerate(subset):
            budget(); x=wires[i].clone()
            if control=='zero_common': x[:,:32]=0.
            if control=='shuffled_common': x[:,:32]=wires[(i+1)%len(wires)][:,:32]
            score,_=tasks(model,clean(x),item,tokenizer,vae,device)
            rows.append(dict(image_id=item['image_id'],mode=model.modality,control=control,
                source_text=item['text'],source_split='validation',source_sha256=item['source_sha256'],
                diagnostic_fixed_received_tensor_not_new_physical_transmission=True,**score))
    return rows
