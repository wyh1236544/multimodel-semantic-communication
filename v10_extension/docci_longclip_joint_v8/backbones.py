"""Strict existing-environment/offline loading. No downloads or random fallbacks."""
from pathlib import Path
from docci_longclip_joint_v8.core import digest, LONGCLIP_SHA256


def load_models(args, device):
    import torch
    from transformers import BartForConditionalGeneration, BartTokenizer, CLIPTokenizer
    from diffusers import AutoencoderKL
    from docci_longclip_joint_v8.vendor_longclip import build_model
    bart_path=Path(args.bart_path).resolve(strict=True); vae_path=Path(args.vae_path).resolve(strict=True)
    token_path=Path(args.clip_tokenizer_path).resolve(strict=True); clip_path=Path(args.longclip_path).resolve(strict=True)
    if digest(clip_path)!=LONGCLIP_SHA256: raise ValueError('Long-CLIP-B differs from official SHA256; refuse pickle load')
    bart,info=BartForConditionalGeneration.from_pretrained(str(bart_path),local_files_only=True,output_loading_info=True)
    missing=set(info.get('missing_keys',[]))
    if missing-{'lm_head.weight'} or any(info.get(k) for k in ('unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError('Incomplete BART checkpoint: '+str(info))
    if bart.lm_head.weight.data_ptr()!=bart.model.shared.weight.data_ptr(): raise ValueError('BART pretrained LM head not tied')
    if bart.config.d_model!=768 or bart.config.max_position_embeddings<1024: raise ValueError('Expected BART-base768 with native1024 positions')
    tokenizer=BartTokenizer.from_pretrained(str(bart_path),local_files_only=True)
    ctok=CLIPTokenizer.from_pretrained(str(token_path),local_files_only=True)
    if (ctok.bos_token_id,ctok.eos_token_id,len(ctok))!=(49406,49407,49408): raise ValueError('Official OpenAI CLIP BPE tokenizer required')
    if (tokenizer.bos_token_id,tokenizer.eos_token_id,tokenizer.pad_token_id)!=(0,2,1): raise ValueError('Official BART tokenizer required')
    # Only this checksum-verified official tensor checkpoint is deserialized.
    state=torch.load(str(clip_path),map_location='cpu')
    if not isinstance(state,dict) or 'positional_embedding_res' not in state: raise ValueError('Not a Long-CLIP checkpoint')
    longclip=build_model(state,load_from_clip=False).float().eval(); del state
    if longclip.context_length!=248 or longclip.visual.input_resolution!=224 or longclip.visual.conv1.kernel_size!=(16,16):
        raise ValueError('Expected official Long-CLIP-B ViT-B/16, 248 context, native224')
    if tuple(longclip.text_projection.shape)!=(512,512): raise ValueError('Long-CLIP-B projection differs')
    vae,vi=AutoencoderKL.from_pretrained(str(vae_path),local_files_only=True,output_loading_info=True,low_cpu_mem_usage=False)
    if any(vi.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')): raise ValueError('Incomplete VAE checkpoint')
    if len(vae.config.block_out_channels)!=4 or vae.config.latent_channels!=4: raise ValueError('VAE stride8/latent4 required')
    models=(bart.to(device).float(),vae.to(device).float(),longclip.to(device))
    for m in models: m.requires_grad_(False).eval()
    def identity(folder):
        files=sorted(p for p in folder.iterdir() if p.is_file() and p.suffix in ('.json','.txt','.bin','.safetensors'))
        if not files: raise ValueError('No complete offline model files')
        return {p.name:digest(p) for p in files}
    record=dict(bart_path=str(bart_path),vae_path=str(vae_path),longclip_path=str(clip_path),clip_tokenizer_path=str(token_path),
        bart_files=identity(bart_path),vae_files=identity(vae_path),clip_tokenizer_files={p.name:digest(p) for p in token_path.iterdir() if p.is_file() and p.suffix in ('.json','.txt')},
        longclip_sha256=LONGCLIP_SHA256,all_pretrained_weights_frozen=True,original_B4_or_COCO_checkpoint_required=False,
        parameters={name:sum(p.numel() for p in m.parameters()) for name,m in zip(('bart','vae','longclip'),models)},
        native_longclip_image_size=224,longclip_context_tokens=248)
    return models,tokenizer,ctok,record


def signature(model):
    import hashlib
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def verify_frozen(models, signatures):
    for model,before in zip(models,signatures):
        if any(p.requires_grad or p.grad is not None for p in model.parameters()) or signature(model)!=before:
            raise RuntimeError('Pretrained model changed or acquired gradients')
