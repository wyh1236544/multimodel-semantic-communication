"""Native HF4.30.2 BLIP-2 vision/Q-Former path, before OPT projection.

No generated words are used as adapter inputs. No BART-alignment claim.
"""
import hashlib

def extract(model,processor,image,device,dtype,budget):
    import torch
    budget()
    batch = processor(images=image,return_tensors='pt')
    if set(batch)!={'pixel_values'}: raise ValueError('Visual path must have no text/token inputs')
    pixels = batch['pixel_values'].to(device,dtype=dtype)
    if not torch.isfinite(pixels).all(): raise FloatingPointError('Invalid pixels')
    with torch.no_grad():
        states = model.vision_model(pixel_values=pixels,return_dict=True).last_hidden_state
        mask = torch.ones(states.shape[:-1],dtype=torch.long,device=states.device)
        query = model.query_tokens.expand(states.shape[0],-1,-1)
        value = model.qformer(query_embeds=query,encoder_hidden_states=states,
            encoder_attention_mask=mask,return_dict=True).last_hidden_state
    if not torch.isfinite(value).all(): raise FloatingPointError('Nonfinite actual Q-Former states')
    p = pixels.detach().float().cpu().contiguous().numpy()
    return value.detach().float(),dict(shape=list(value.shape),dtype=str(value.dtype),
        input_has_text=False,source_image_at_RX=False,level='QFormer.last_hidden_state before language_projection',
        processed_pixels_float32_sha256=hashlib.sha256(p.tobytes()).hexdigest(),
        processed_pixels_shape=list(pixels.shape),finite=True)
