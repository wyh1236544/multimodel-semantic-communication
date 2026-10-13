"""One received waveform per pair/condition, reused by EVERY decoder/control."""
import json
import numpy as np
import torch
from bidirectional_rx_v7.core import array_hash,mask_policy,variants
from clip_bart_joint_v5.data import clean_args
from clip_bart_joint_v5.core import make_row
from clip_bart_joint_v5.channel import send,resource
from clip_bart_joint_v6.model import masked_wire
from bart_latent_codec_v4.model import generate
from fso_fluctuating_v2.core import paths,seed_for
from utils.metrics import batch_image_metrics
from optical_realism_v3.constellation import constellation_svg
from bidirectional_rx_v7.features import receiver_features


@torch.no_grad()
def evaluate(adapter,base,states,items,c,cc,vae,scaling,clip,processor,tok,device,budget,folder,phases,atomic_json):
    from PIL import Image
    folder.mkdir(exist_ok=False); (folder/'images').mkdir(); (folder/'frames').mkdir(); (folder/'constellations').mkdir()
    settings=[('clean',None,None),('clean_drop_image',None,1),('clean_drop_text',None,2)]
    settings += [(v['id'],v,None) for v in cc['conditions']]
    selection=variants(phases); image_count=min(c['save_images'],len(items))
    atomic_json(folder/'eval_design.json',dict(conditions=[x[0] for x in settings],variants=selection,
        validation_ids=[v['image_id'] for v in items],resources=resource(cc),channel_config=cc,
        same_physical_waveform_all_decoders=True,synthetic_conditions=['clean_drop_image','clean_drop_text'],
        clean_snr_sentinel_db=40.,no_oracle_receiver=True))
    def png(pixel,name):
        im=(pixel[0].clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8); Image.fromarray(im).save(folder/name)
    with (folder/'samples.jsonl').open('x',encoding='utf-8') as writer,(folder/'frames.jsonl').open('x') as fw,(folder/'events.jsonl').open('x') as ew:
        for cid,condition,drop in settings:
            trace=None if condition is None else paths(cc,condition['weather'],len(items),('v7-eval-weather',condition['weather']))
            received=[]
            for i,item in enumerate(items):
                budget(); wire=item['wire'].to(device); stats=None; policy=None
                if condition is None:
                    if drop is not None: policy=mask_policy(c,drop,item['image_id']); wire=masked_wire(wire,policy)
                    args=clean_args(wire)
                else:
                    y,stats,transport=send(cc,condition,trace[i],seed_for(c['seed'],'v7-waveform',item['image_id'],cid),wire[0].cpu().numpy())
                    y=torch.as_tensor(y[None],device=device); args=(y,torch.tensor([stats['receiver_available']],device=device),y.new_tensor([stats['pilot_snr_db']]),y.new_tensor([stats['analog_reliability']]))
                    for event in transport.events: ew.write(json.dumps(dict(event,image_id=item['image_id'],condition=cid),allow_nan=False)+'\n')
                    ew.flush()
                    if i<image_count: constellation_svg(transport.constellations,folder/'constellations'/('{}__{}.svg'.format(cid,item['image_id'])))
                name='frames/{}__{}.npz'.format(cid,item['image_id'])
                # Strict public RX cache: no clean source memory/text/length/mask.
                np.savez_compressed(folder/name,observed=args[0][0].cpu().numpy(),available=np.asarray(bool(args[1][0])),
                    pilot_snr_db=np.asarray(float(args[2][0])),reliability=np.asarray(float(args[3][0])))
                fh=array_hash(args[0][0].cpu().numpy())
                frame=dict(condition=cid,image_id=item['image_id'],frame_file=name,observed_sha256=fh,receiver_available=bool(args[1][0]),
                    pilot_snr_db=float(args[2][0]),reliability=float(args[3][0]),stats=stats,synthetic_private_mask=policy)
                fw.write(json.dumps(frame,allow_nan=False)+'\n'); fw.flush()
                own=receiver_features(base.receive(*args),vae,scaling,clip,processor); received.append((args,own,frame))
            for arm,phase,control in selection:
                if arm!='B0_frozen': adapter.configure(arm); adapter.restore(states[(arm,phase)])
                for i,item in enumerate(items):
                    budget(); args,own,frame=received[i]; donor=(i+1)%len(items)
                    out=own if arm=='B0_frozen' else adapter(own,args[2],args[3],control,received[donor][1] if control.startswith('shuffle_') else None)
                    result=generate(base.text.bart,out['text'],tok)
                    pixel=((vae.decode(out['latent']/scaling).sample+1)/2).clamp(0,1)*args[1].to(out['latent'])[:,None,None,None]
                    measures={k:float(v[0]) for k,v in batch_image_metrics(pixel,item['image'].to(device)).items()}
                    r=make_row(item,arm,phase,cid,result,measures,control=control,stats=frame['stats'],frame_file=frame['frame_file'],
                        observed_sha256=frame['observed_sha256'],pilot_snr_db=frame['pilot_snr_db'],
                        donor_image_id=items[donor]['image_id'] if control.startswith('shuffle_') else None,
                        receiver_intervention_not_new_transmission=control!='normal',synthetic_condition=drop is not None)
                    if i<image_count:
                        name='images/{}__{}__{}__{}__{}.png'.format(cid,item['image_id'],arm,phase,control); png(pixel,name); r['image_file']=name
                        original='images/{}__source.png'.format(item['image_id'])
                        if not (folder/original).exists(): png(item['image'],original)
                        r['source_image_file']=original
                    writer.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n'); writer.flush()
                print('EVAL {} {} {} {} finished'.format(arm,phase,control,cid),flush=True)
