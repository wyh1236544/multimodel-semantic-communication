"""Exact frozen B4 reference; no old decoder or image model is fine-tuned."""
import copy
import json
from pathlib import Path
from text_jscc_v1_fix1.core import digest


def load_reference(args,device):
    from v10.runtime import load_config
    from v10.model import build_model
    from utils.checkpoint import load_checkpoint
    from token_guidance.model import Guidance
    from fso_jscc_v1.model import ContinuousChannel,parameters
    from fso_jscc_v1.core import kernels
    from vae_tail_ablation_v1.model import apply_parent
    from joint_jscc_arch_v1.model import ArchitectureModel
    from joint_jscc_arch_v1.report import verify
    parent=Path(args.source_run).resolve(strict=True); verify(parent,complete=True)
    design=json.loads((parent/'design.json').read_text()); identity=json.loads((parent/'run_identity.json').read_text())
    if design['smoke'] or design['config']['max_text_tokens']!=128: raise ValueError('Completed main B4 required')
    b4file=parent/'train_B4_separate/fso_delta.pth'; b2file=Path(args.legacy_run)/'train_B2_fso_ft/last_delta.pth'
    hashes=dict(base_checkpoint_sha256=digest(args.checkpoint),a_delta_sha256=digest(args.a_delta),
        split_sha256=digest(args.split_manifest),b4_sha256=digest(b4file),b2_sha256=digest(b2file))
    if any(identity[k]!=hashes[k] for k in ('base_checkpoint_sha256','a_delta_sha256','split_sha256')) or identity['init_fso_delta_sha256']!=hashes['b2_sha256']:
        raise ValueError('Nonexact B4/base/A/B2/split identity')
    config=load_config(args); record=load_checkpoint(args.checkpoint)
    if record.get('split_sha256')!=hashes['split_sha256']: raise ValueError('Wrong base split')
    config['v10']=copy.deepcopy(record['config']['v10']); config['v10'].update(blocks=1,csi='pilot',fixed_mode=0)
    config['model']['vae_use_fp16']=False; del record
    model,exact=build_model(config,device,args.checkpoint)
    if exact!={'v10_exact':True}: raise ValueError('Nonexact initial model')
    model.float(); guide=Guidance(model); a=load_checkpoint(args.a_delta)
    if a.get('version')!='V10-token-guidance-quick-v1' or a.get('variant')!='A' or set(a['delta'])!={n for n,_ in guide.trainable('A')}:
        raise ValueError('Wrong original A delta')
    model.load_state_dict(dict(model.state_dict(),**a['delta']),strict=True); del a
    guide.set('A'); model.base.channel=ContinuousChannel(model.base.channel,design['channel_config']).to(device)
    apply_parent(model,parameters(model),load_checkpoint(b2file)); model.requires_grad_(False); model.eval()
    core=ArchitectureModel(model.base,design['config'],kernels(design['channel_config'])).to(device).float()
    saved=load_checkpoint(b4file)
    if (saved.get('version'),saved.get('arm'),saved.get('phase'))!=('V10-joint-jscc-architecture-v1','B4_separate','fso') or saved['architecture_config']!=design['config']:
        raise ValueError('Wrong B4 checkpoint')
    core.load_state_dict(saved['delta'],strict=True); core.configure('B4_separate'); core.requires_grad_(False); core.eval()
    return model,core,design['channel_config'],hashes


def precompression(core,item,device):
    import torch
    from joint_jscc_arch_v1.model import attention_mask
    from joint_jscc_arch_v1.core import source_groups
    t=item['text_features'].to(device); mask=item['text_mask'].to(device)
    latent=torch.zeros(1,4,16,16,device=device)
    v=core.image_projection(latent).flatten(2).transpose(1,2)+core.image_position+core.image_type
    t=t+core.text_position[:,:t.shape[1]]+core.text_type
    tokens=torch.cat((core.common_queries.expand(1,-1,-1),v,t),1)
    pad=torch.cat((torch.zeros(1,288,device=device,dtype=torch.bool),~mask.bool()),1)
    h=core.source_norm(core.source_transformer(tokens,mask=attention_mask(source_groups(t.shape[1]),True,device),src_key_padding_mask=pad))
    return h[:,288:][:,mask[0].bool()]
