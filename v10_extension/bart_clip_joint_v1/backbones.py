"""Strict offline official BART/CLIP loading; no random replacement language head."""
from bart_clip_joint_v1.core import offline_snapshot


def load_bart(path, device):
    from transformers import BartForConditionalGeneration, BartTokenizer
    record = offline_snapshot(path, 'bart')
    tokenizer = BartTokenizer.from_pretrained(path, local_files_only=True)
    model, info = BartForConditionalGeneration.from_pretrained(path, local_files_only=True,
        output_loading_info=True, from_tf=False)
    missing = set(info.get('missing_keys', []))
    if missing-set(('lm_head.weight', 'final_logits_bias')) or info.get('unexpected_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError('Incomplete pretrained BART: '+str(info))
    if model.lm_head.weight.data_ptr() != model.model.shared.weight.data_ptr(): raise RuntimeError('BART LM head is not tied')
    if (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) != (0, 2, 1): raise ValueError('Wrong BART tokenizer')
    if model.model.encoder.embed_tokens.weight.data_ptr() != model.model.shared.weight.data_ptr() or model.model.decoder.embed_tokens.weight.data_ptr() != model.model.shared.weight.data_ptr():
        raise RuntimeError('Pretrained BART shared embeddings are not preserved')
    record.update(loading_info=info, tied_pretrained_lm_head=True, full_pretrained_encoder_decoder=True,
                  parameters=sum(p.numel() for p in model.parameters()), d_model=model.config.d_model)
    return model.to(device).float().requires_grad_(False).eval(), tokenizer, record


def load_clip(path, device):
    from transformers import CLIPModel, CLIPProcessor
    record = offline_snapshot(path, 'clip')
    model, info = CLIPModel.from_pretrained(path, local_files_only=True, output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('Complete pretrained CLIP vision/text towers required: '+str(info))
    processor = CLIPProcessor.from_pretrained(path, local_files_only=True)
    record.update(loading_info=info, parameters=sum(p.numel() for p in model.parameters()))
    return model.to(device).float().requires_grad_(False).eval(), processor, record
