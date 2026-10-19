"""Offline, frozen BLIP-2 with public generate(), raw-token and finite-state audits."""
import hashlib
import json
import time
from docci_caption_calibration_v24.core import (generation_kwargs, processor_kwargs,
    token_audit, text_audit)

def state_digest(model, budget):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        budget()
        a = value.detach().contiguous().cpu().numpy()
        h.update(name.encode()); h.update(str((a.shape, a.dtype)).encode()); h.update(a.tobytes())
    return h.hexdigest()

class Probe:
    def __init__(self, model):
        self.handles = []
        self.values = {}
        for name in ('vision_model', 'qformer', 'language_projection', 'language_model'):
            self.handles.append(getattr(model, name).register_forward_hook(self.hook(name)))
    def hook(self, name):
        def capture(module, args, output):
            import torch
            value = output if name == 'language_projection' else (
                output.logits if name == 'language_model' else output.last_hidden_state)
            if not torch.isfinite(value).all().item():
                raise FloatingPointError('Nonfinite actual '+name+' output')
            if name in self.values:
                self.values[name]['checked_forwards'] += 1
                return
            v = value.detach().float()
            self.values[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
                finite=True, checked_forwards=1, mean=float(v.mean()), std=float(v.std(unbiased=False)),
                min=float(v.min()), max=float(v.max()))
        return capture
    def close(self):
        for h in self.handles:
            h.remove()

def generate_one(model, processor, image, prompt, stage, device, dtype, budget):
    import torch
    batch = processor(**processor_kwargs(image, prompt))
    input_ids = batch.get('input_ids')
    input_tokens = 1 if input_ids is None else int(input_ids.shape[-1])
    kwargs = generation_kwargs(stage)
    if input_tokens+model.config.num_query_tokens+kwargs['max_new_tokens'] > model.config.text_config.max_position_embeddings:
        raise ValueError('No text/context truncation permitted')
    batch = {k: v.to(device, dtype=dtype) if v.is_floating_point() else v.to(device)
        for k, v in batch.items()}
    pixels = batch['pixel_values']
    if not torch.isfinite(pixels).all().item():
        raise FloatingPointError('Nonfinite processed pixels')
    p = pixels.detach().float().cpu().contiguous().numpy()
    tensor_info = dict(shape=list(p.shape), dtype=str(pixels.dtype), finite=True,
        min=float(p.min()), max=float(p.max()), mean=float(p.mean()), std=float(p.std()),
        float32_sha256=hashlib.sha256(p.tobytes()).hexdigest())
    probe = Probe(model)
    start = time.monotonic(); budget()
    try:
        with torch.no_grad():
            result = model.generate(**batch, **kwargs)
        budget()
        sequence = [int(x) for x in result.sequences[0].tolist()]
        eos = model.generation_config.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else list(eos)
        telemetry = token_audit(sequence, len(result.scores), eos_ids, kwargs['max_new_tokens'])
        scores = []
        for j, logits in enumerate(result.scores):
            # Processed scores can legally contain -inf for masked tokens. Raw logits
            # were independently required finite by the language-model forward hook.
            x = logits[0].detach().float()
            if torch.isnan(x).any().item() or torch.isposinf(x).any().item() or not torch.isfinite(x).any().item():
                raise FloatingPointError('Invalid generation scores')
            prob = torch.softmax(x, dim=-1)
            vals, ids = prob.topk(5)
            scores.append(dict(step=j, selected_token_id=telemetry['new_token_ids'][j],
                top5_ids=[int(v) for v in ids.tolist()], top5_probabilities=[float(v) for v in vals.tolist()],
                eos_probability=float(prob[eos_ids].sum()),
                masked_negative_infinity_count=int(torch.isneginf(x).sum())))
        if set(probe.values) != {'vision_model', 'qformer', 'language_projection', 'language_model'}:
            raise ValueError('Actual forward probes missing')
        text = processor.batch_decode(result.sequences, skip_special_tokens=True)[0].strip()
        raw = processor.batch_decode(result.sequences, skip_special_tokens=False)[0]
        telemetry.update(input_token_ids=None if input_ids is None else input_ids[0].tolist(),
            implicit_BOS=input_ids is None, eos_token_ids=eos_ids,
            bos_token_id=model.config.text_config.bos_token_id,
            pad_token_id=model.generation_config.pad_token_id,
            processed_pixels=tensor_info, forward_probes=probe.values, score_trace=scores,
            elapsed_seconds=time.monotonic()-start,
            cuda_allocated_gib=None if str(device) == 'cpu' else torch.cuda.memory_allocated()/2**30,
            cuda_reserved_gib=None if str(device) == 'cpu' else torch.cuda.memory_reserved()/2**30)
        return dict(output_text=text, raw_decoded_text=raw, text_audit=text_audit(text), telemetry=telemetry)
    finally:
        probe.close()

def load_model(folder, dtype, device):
    import torch
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    processor = Blip2Processor.from_pretrained(str(folder), local_files_only=True)
    model, info = Blip2ForConditionalGeneration.from_pretrained(str(folder),
        local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=False,
        output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('BLIP-2 loading identity mismatch: '+json.dumps(info))
    model.to(device).requires_grad_(False).eval()
    return model, processor, info
