"""Actual greedy text, original images and complete fresh coherent waveforms."""
import json
from pathlib import Path
import torch
from clip_bart_joint_v5.core import make_row
from clip_bart_joint_v5.data import source, clean_args
from clip_bart_joint_v5.channel import send, resource
from bart_latent_codec_v4.model import generate
from fso_fluctuating_v2.core import paths, seed_for
from utils.metrics import batch_image_metrics
from optical_realism_v3.constellation import constellation_svg


@torch.no_grad()
def evaluate(model, items, c, cc, vae, scaling, tok, device, budget, folder, phase, atomic_json):
    from PIL import Image
    import numpy as np
    folder.mkdir(exist_ok=False); (folder/'images').mkdir(); (folder/'constellations').mkdir()
    atomic_json(folder/'eval_design.json', dict(arm=model.arm, phase=phase, conditions=['clean']+[v['id'] for v in cc['conditions']],
        validation_ids=[v['image_id'] for v in items], resources=resource(cc), channel_config=cc,
        controls=['normal', 'zero_common', 'no_common_fusion', 'shuffled_common'] if model.arm == 'J1_joint' else ['normal'],
        same_received_frame_controls_are_offline_interventions=True, single_channel_draw=True))
    def save_png(pixel, filename):
        values=(pixel[0].clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
        Image.fromarray(values).save(folder/filename)
    conditions = [None]+list(cc['conditions'])
    with (folder/'samples.jsonl').open('x',encoding='utf-8') as writer, (folder/'events.jsonl').open('x') as events:
        for condition in conditions:
            cid='clean' if condition is None else condition['id']
            trace=None if condition is None else paths(cc,condition['weather'],len(items),('v5-eval-weather',condition['weather']))
            received=[]
            for i,item in enumerate(items):
                budget(); x=source(model,item,device)
                if condition is None: args=clean_args(x); stats=None
                else:
                    y,stats,transport=send(cc,condition,trace[i],seed_for(c['seed'],'v5-waveform',item['image_id'],cid),x[0].cpu().numpy())
                    y=torch.as_tensor(y[None],device=device)
                    args=(y,torch.tensor([stats['receiver_available']],device=device),y.new_tensor([stats['pilot_snr_db']]),y.new_tensor([stats['analog_reliability']]))
                    for event in transport.events:
                        events.write(json.dumps(dict(event,image_id=item['image_id'],condition=cid),allow_nan=False)+'\n')
                    events.flush()
                    if i<c['save_images']: constellation_svg(transport.constellations,folder/'constellations'/('{}__{}.svg'.format(cid,item['image_id'])))
                received.append((args,stats))
            controls=['normal','zero_common','no_common_fusion','shuffled_common'] if model.arm=='J1_joint' else ['normal']
            for control in controls:
                for i,item in enumerate(items):
                    budget(); args,stats=received[i]; donor=(i+1)%len(items)
                    if control=='shuffled_common':
                        wire=args[0].clone(); wire[:,:512]=received[donor][0][0][:,:512]
                        args=(wire,)+args[1:]; intervention='normal'
                    else: intervention=control
                    out=model.receive(*args,intervention=intervention)
                    result=generate(model.text.bart,out['text'],tok)
                    pixel=((vae.decode(out['latent']/scaling).sample+1)/2).clamp(0,1)*args[1].to(out['latent'])[:,None,None,None]
                    measures={k:float(v[0]) for k,v in batch_image_metrics(pixel,item['image'].to(device)).items()}
                    r=make_row(item,model.arm,phase,cid,result,measures,control=control,stats=stats,
                        donor_image_id=items[donor]['image_id'] if control=='shuffled_common' else None,
                        receiver_intervention_not_new_transmission=control!='normal')
                    if i<c['save_images']:
                        name='images/{}__{}__{}.png'.format(cid,item['image_id'],control); save_png(pixel,name); r['image_file']=name
                        original='images/{}__source.png'.format(item['image_id'])
                        if not (folder/original).exists(): save_png(item['image'],original)
                        r['source_image_file']=original
                    writer.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n'); writer.flush()
            if condition is None and model.arm=='J1_joint':
                # Wrong TEXT at transmitter; both CLIP/BART inputs from a donor,
                # image remains own. Distinct source-content diagnostic, not RX ablation.
                for i,item in enumerate(items):
                    budget(); donor=(i+1)%len(items); x=source(model,item,device,items[donor]); args=clean_args(x)
                    out=model.receive(*args); result=generate(model.text.bart,out['text'],tok)
                    pixel=((vae.decode(out['latent']/scaling).sample+1)/2).clamp(0,1)
                    measures={k:float(v[0]) for k,v in batch_image_metrics(pixel,item['image'].to(device)).items()}
                    r=make_row(item,model.arm,phase,'clean',result,measures,control='wrong_text_source',stats=None,
                        donor_image_id=items[donor]['image_id'],receiver_intervention_not_new_transmission=False,
                        evaluation_target_is_original_matched_pair=True)
                    writer.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n'); writer.flush()
            print('EVAL {} {} {} finished'.format(model.arm,phase,cid),flush=True)
