"""Optional, separate received-information-only diffusion preview.

Does not reuse AWGN-calibrated V9-B verifier thresholds on shifted FSO data.
Every candidate is labelled unselected, not claimed to improve the source image.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import torch
from v10.io import atomic_json
from v10.data import sha256
from v10.paths import write_path

def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--eval-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--diffusion-model-path', required=True)
    p.add_argument('--strength-list', nargs='+', type=float, default=[.35])
    p.add_argument('--max-candidates', type=int, default=16)
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--guidance-scale', type=float, default=6.)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-minutes', type=float, default=30.)
    a = p.parse_args()
    if min(a.max_candidates, a.steps, a.max_minutes) <= 0 or any(not 0 < s <= 1 for s in a.strength_list):
        raise ValueError('Invalid generation budget')
    source, dest = Path(a.eval_dir).resolve(), write_path(a.output_dir)
    if not (source/'COMPLETE.json').is_file():
        raise ValueError('Only completed evaluation runs may be previewed')
    rows = []
    for line in (source/'samples.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row['receiver_mode'] == 'reconstruction' or row['header_rejected'] or not row.get('image_file'):
            continue
        # Whitelist receiver-available fields; no source captions or true SNR.
        prompt = ' '.join(row['received_caption'].split())
        if len(prompt.split()) < 3:
            continue
        image = (source/row['image_file']).resolve()
        if source not in image.parents or not image.is_file():
            raise ValueError('Invalid preview image path')
        rows.append({'image': str(image), 'prompt': prompt, 'mode': row['receiver_mode']})
        if len(rows)*len(a.strength_list) >= a.max_candidates:
            break
    if not rows:
        raise ValueError('No eligible saved receiver images. Increase --save-images during evaluation if needed')
    dest.mkdir(parents=True, exist_ok=False)
    from diffusers import StableDiffusionImg2ImgPipeline
    from PIL import Image
    device = torch.device(a.device)
    if device.type == 'cuda':
        free, _ = torch.cuda.mem_get_info(device)
        if free < 6*2**30:
            raise RuntimeError('Less than 6 GiB GPU memory free; do not stop unrelated processes')
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(a.diffusion_model_path,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
            local_files_only=True).to(device)
    pipe.enable_attention_slicing()
    started, results = time.monotonic(), []
    atomic_json(dest/'identity.json', {'args': vars(a), 'input_samples_sha256': sha256(source/'samples.jsonl'),
                'prompt_source': 'received_caption', 'selection': 'not_applied', 'source_caption_used': False})
    for index, row in enumerate(rows):
        for strength in a.strength_list:
            if len(results) >= a.max_candidates:
                break
            if time.monotonic()-started > a.max_minutes*60:
                raise TimeoutError('Generation budget exhausted; partial images retained')
            image = Image.open(row['image']).convert('RGB').resize((512, 512))
            output = pipe(prompt=row['prompt'], image=image, strength=strength,
                          num_inference_steps=a.steps, guidance_scale=a.guidance_scale,
                          generator=torch.Generator(device=device).manual_seed(a.seed+index)).images[0]
            name = 'candidate_{:04d}_strength{}.png'.format(index, strength)
            output.save(dest/name)
            results.append(dict(row, strength=strength, file=name, status='unselected_preview'))
            print('candidate={}/{} {}'.format(len(results), a.max_candidates, name), flush=True)
    atomic_json(dest/'candidates.json', results)
    atomic_json(dest/'COMPLETE.json', {'candidates': len(results), 'wall_seconds': time.monotonic()-started})

if __name__ == '__main__':
    main()
