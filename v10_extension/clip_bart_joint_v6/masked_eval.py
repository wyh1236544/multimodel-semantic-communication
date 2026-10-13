"""Fixed unseen-validation receiver-feature erasures; NOT a physical channel."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from clip_bart_joint_v5.data import source,clean_args
from clip_bart_joint_v6.model import masked_wire
from clip_bart_joint_v6.core import private_mask,make_row
from bart_latent_codec_v4.model import generate
from utils.metrics import batch_image_metrics


@torch.no_grad()
def evaluate(model,items,c,vae,scaling,tok,device,budget,path):
    image_dir=Path(path).with_suffix(''); image_dir.mkdir()
    wires=[source(model,item,device) for item in items]
    with Path(path).open('x',encoding='utf-8') as log:
        for control in ('normal','zero_common','no_common_fusion','shuffled_common'):
            for i,item in enumerate(items):
                budget(); donor=(i+1)%len(items); policy=private_mask(c,3,item['image_id'],'clean')
                wire=masked_wire(wires[i],policy)
                if control=='shuffled_common': wire[:,:512]=wires[donor][:,:512]
                intervention='normal' if control=='shuffled_common' else control
                out=model.receive(*clean_args(wire),intervention=intervention)
                result=generate(model.text.bart,out['text'],tok)
                pixels=((vae.decode(out['latent']/scaling).sample+1)/2).clamp(0,1)
                met={k:float(v[0]) for k,v in batch_image_metrics(pixels,item['image'].to(device)).items()}
                r=make_row(item,model.arm,'clean','receiver_feature_dropout',result,met,control=control,
                    private_mask=policy,not_a_physical_channel=True,
                    donor_image_id=items[donor]['image_id'] if control=='shuffled_common' else None)
                if i<c['save_images']:
                    filename=image_dir/('{}__{}.png'.format(item['image_id'],control))
                    array=(pixels[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
                    Image.fromarray(array).save(filename); r['image_file']=str(filename.relative_to(path.parent))
                log.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n'); log.flush()
