"""Load exactly ONE branch's authentic offline backbone, not all joint models."""
from pathlib import Path
from docci_modal_v9.core import digest
from docci_longclip_joint_v8.backbones import signature,verify_frozen


def file_identity(folder):
    folder=Path(folder).resolve(strict=True)
    files=sorted(f for f in folder.iterdir() if f.is_file() and f.suffix in ('.bin','.json','.txt','.safetensors'))
    if not files: raise ValueError('Offline backbone files missing')
    return {f.name:digest(f) for f in files}


def historical_backbone(branch,files):
    expected=(dict(config_json='c4022c931216afcb778991477332e5312aca8be478027ee3a8bf34b7004c44e2',
        pytorch_model_bin='6e4fdb5e04114696ec4f7cc0cb5fc2a22a826f1c9eea234ca550b5390341a2fd',
        merges_txt='1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5',
        vocab_json='9e7f63c2d15d666b52e21d250d2e513b87c9b713cfa6987a82ed89e5e6e50655') if branch=='text' else
        dict(config_json='92d3dfb746fca211a2c9e019e285f8597412211728dce3c5bcf4eda0f2d62e7e',
        diffusion_pytorch_model_bin='1b4889b6b1d4ce7ae320a02dedaeff1780ad77d415ea0d744b476155c6377ddc'))
    for key,sha in expected.items():
        name=key.rsplit('_',1)[0]+'.'+key.rsplit('_',1)[1]
        if files.get(name)!=sha: raise ValueError('Historical backbone differs: '+name+'; no cross-backbone warm start')
    if branch=='image' and 'diffusion_pytorch_model.safetensors' in files and files['diffusion_pytorch_model.safetensors']!='a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815':
        raise ValueError('Alternative VAE weight differs')


def load(branch,args,device):
    if branch=='text':
        from bart_clip_joint_v1.backbones import load_bart
        model,tok,record=load_bart(args.bart_path,device)
        if model.config.d_model!=768 or model.config.max_position_embeddings<1024: raise ValueError('Complete BART-base768/native1024 required')
        record.update(files=file_identity(args.bart_path),VAE_CLIP_LongCLIP_loaded=False)
        return model,tok,record
    from diffusers import AutoencoderKL
    model,info=AutoencoderKL.from_pretrained(args.vae_path,local_files_only=True,output_loading_info=True,low_cpu_mem_usage=False)
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')): raise ValueError('Incomplete actual VAE')
    if model.config.latent_channels!=4 or len(model.config.block_out_channels)!=4: raise ValueError('Stride8/latent4 VAE required')
    model=model.to(device).float().requires_grad_(False).eval()
    return model,None,dict(files=file_identity(args.vae_path),parameters=sum(p.numel() for p in model.parameters()),BART_CLIP_LongCLIP_loaded=False)
