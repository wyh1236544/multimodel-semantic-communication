"""Frozen CUDA inference only. Local assets, no automatic network or training."""
import json
import math
from pathlib import Path
import time
import numpy as np
from PIL import Image
from docci_pretrained_checks_v22.core import (CONFIG, SCHEMA, atomic, read, rows, sha,
    digest, noise, seed, case_key, case_name, donor_map, prepared, local, complete, verify_tree)

def cuda():
    import os
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise RuntimeError('Use physical GPU1 explicitly; never stop another job')
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required')
    free, total = torch.cuda.mem_get_info()
    print('GPU1 free={:.2f}GiB total={:.2f}GiB'.format(free/2**30, total/2**30), flush=True)
    if free < 20*2**30:
        raise RuntimeError('Need 20GiB free at start; no automatic changes')
    return torch.device('cuda:0')

def observed(wire, eid, snr, repeat, kind):
    import torch
    from docci_text_guided_image_v20.core import array_sha
    n = 8192 if kind == 'image' else 18432
    if tuple(wire.shape) != (1, n) or abs(float(wire.square().mean())-1) > .0002:
        raise ValueError('Frozen TX shape/power differs')
    e = noise(eid, snr, repeat, kind, n)
    p = noise(eid, snr, repeat, kind+'_pilot', 64)
    n0 = float(np.mean(p.astype(float)**2))
    estimate = 40. if snr is None else max(-40., min(40., -10*math.log10(max(n0, 1e-12))))
    y = wire+torch.as_tensor(e[None], device=wire.device)
    stats = dict(nominal_snr_db=snr, pilot_noise_variance=n0,
        estimated_snr_db=None if snr is None else estimate,
        realized_snr_db=None if snr is None else -10*math.log10(float(np.mean(e.astype(float)**2))),
        noise_sha256=array_sha(e), pilot_sha256=array_sha(p),
        signal_power=float(wire.square().mean()), post_noise_renormalization=False)
    return y, wire.new_tensor([estimate]), wire.new_tensor([1/(1+n0)]), stats

def prepare(a, root, budget):
    import torch
    from transformers import BartTokenizer
    from docci_text_aid_v21.core import CACHE_SHA, PARENTS
    from docci_text_aid_v21.runtime import anchors
    from docci_text_guided_image_v20_fix1.compat import source_check, validate_gate, pooling_patch
    from docci_text_guided_image_v20 import model as old_model
    from docci_text_guided_image_v20.data import Items, tensors
    from docci_text_guided_image_v20.engine import make, rgb_decode, state_hash, signatures, check_frozen
    from docci_baseline_v14.model import decode
    source_check()
    old = Path(a.v20_root)
    if read(old/'PIPELINE.json')['status'] != 'COMPLETE' or sha(old/'cache/index.json') != CACHE_SHA:
        raise ValueError('Pinned completed V20 full run/cache required')
    validate_gate(read(old/'FIX1_GPU_GATE.json'))
    checkpoint = old/'train_A_self/last.pth'
    if sha(checkpoint) != PARENTS['A_self']:
        raise ValueError('Pinned saved V20 A differs before deserialization')
    a.cache = str(old/'cache')
    count, repeats = CONFIG['profiles'][a.profile]
    items = Items(a.cache, 'dev')
    if len(items) != 100:
        raise ValueError('Expected original 100 V20 dev, not TRAIN or qual_test')
    selected = sorted(items.entries, key=lambda r: seed(42, 'V22-fixed-dev', r['example_id']))[:count]
    if any(r['cohort'] != 'dev' or r['official_split'] != 'train' or r['input_truncated'] for r in selected):
        raise ValueError('Invalid original held-out dev entries')
    ids = [r['example_id'] for r in selected]
    donors = donor_map(ids)
    indices = {r['example_id']: i for i, r in enumerate(items.entries)}
    device = cuda()
    budget.gpu = True
    tok = BartTokenizer.from_pretrained(a.bart_path, local_files_only=True)
    print('V22 LOAD pinned V20 A, frozen VAE/BART/V18/LPIPS; then replay two real anchors', flush=True)
    with pooling_patch(old_model), torch.no_grad():
        image, vae, text, lpips = make(a, device, checkpoint)
        image.requires_grad_(False)
        before, frozen = state_hash(image), signatures(vae, text, lpips)
        # Replay real archived observations before NEW noise/OOD inference.
        anchors(a, dict(selected={'dev': selected, 'train': []}, mask_blocks=0), root,
                'A_self', image, vae, text, lpips, device, budget)
        truth = root/'truth'
        rxdir = root/'receiver'
        truth.mkdir(); rxdir.mkdir()
        examples = []
        with (root/'rows.jsonl').open('x', encoding='utf-8') as f:
            for i, entry in enumerate(selected):
                budget()
                item = items[indices[entry['example_id']]]
                src = tensors(item, device)
                eid = item['example_id']
                source = truth/(eid+'.png')
                Image.fromarray(item['rgb']).save(source)
                examples.append(dict(example_id=eid, original_split=item['official_split'],
                    cohort='dev', source_image=source.relative_to(root).as_posix(),
                    source_image_sha256=sha(source), source_text=item['text'],
                    source_text_sha256=item['text_sha256'], rectangle=list(map(int, item['rectangle']))))
                wire = image.encode(src['latent'], None)
                for snr in CONFIG['snr_db']:
                    for rep in range(repeats):
                        budget()
                        y, estimate, reliability, stats = observed(wire, eid, snr, rep, 'image')
                        latent = image.receive(y, None, estimate, reliability)
                        rgb = rgb_decode(vae, latent).clamp(0, 1)
                        output = rxdir/(case_name(eid, snr, rep)+'.png')
                        Image.fromarray((rgb[0].permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)).save(output)
                        ty, _, _, ts = observed(src['text_wire'], eid, snr, rep, 'text')
                        rx = text.receive(ty)
                        caption = decode(text.bart, rx, tok, 'greedy')['received_caption']
                        row = dict(example_id=eid, donor_id=donors[eid], snr_db=snr, repeat=rep,
                            received_image=output.relative_to(root).as_posix(), received_image_sha256=sha(output),
                            received_caption=caption, received_caption_sha256=digest(caption.encode()),
                            image_stats=stats, text_stats=ts, image_payload_symbols=4096, image_pilot_symbols=32,
                            text_payload_symbols=9216, text_pilot_symbols=32, total_symbols=13376,
                            source_text_at_RX=False, post_noise_renormalization=False,
                            negative_SNR_out_of_training_range=snr is not None and snr < 0)
                        f.write(json.dumps(row, allow_nan=False)+'\n'); f.flush()
                print('V22 PREPARE {}/{} frozen image/text; no training'.format(i+1, count), flush=True)
        if state_hash(image) != before:
            raise RuntimeError('Frozen image model changed')
        check_frozen((vae, text, lpips), frozen)
    atomic(root/'index.json', dict(schema=SCHEMA, config=CONFIG, profile=a.profile, repeats=repeats,
        examples=examples, cache_index_sha256=CACHE_SHA, image_parent_sha256=PARENTS['A_self'],
        reused_development_set=True, independent_test=False, train_updates=0))
    complete(root, 'prepare', dict(profile=a.profile, examples=count, rows=count*3*repeats,
        frozen_models_unchanged=True, actual_saved_parent_anchor=True,
        note='AWGN only; negative SNR is OOD inference; new noise, not old V20 scores'))

def asset(a, kind):
    lock = read(a.asset_lock)
    if lock['kind'] != kind or lock['remote_code_allowed']:
        raise ValueError('Wrong asset lock')
    path = verify_tree(lock['tree'])
    return path, lock

def save_line(f, row):
    f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); f.flush()

def caption(a, root, budget):
    import torch
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    index, data, _ = prepared(a.prepared)
    model_path, lock = asset(a, 'caption')
    if read(model_path/'config.json').get('model_type') != 'blip-2':
        raise ValueError('Only existing pretrained BLIP-2 supported; no remote custom code')
    device = cuda(); budget.gpu = True
    processor = Blip2Processor.from_pretrained(str(model_path), local_files_only=True)
    model = Blip2ForConditionalGeneration.from_pretrained(str(model_path), local_files_only=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=False).to(device).requires_grad_(False).eval()
    from docci_longclip_joint_v8.backbones import signature, verify_frozen
    frozen = signature(model)
    lookup = {case_key(r): r for r in data}
    blank = Image.new('RGB', (512, 512), (128, 128, 128))
    def generate(im, prompt):
        budget()
        batch = processor(images=im, text=prompt, return_tensors='pt', truncation=False)
        length = int(batch['input_ids'].shape[-1])
        maximum = getattr(model.config.text_config, 'max_position_embeddings', 2048)
        if length+CONFIG['caption_max_new_tokens']+model.config.num_query_tokens > maximum:
            raise ValueError('Full received text exceeds captioner context; no silent truncation')
        batch = {k: v.to(device, dtype=torch.float16) if v.is_floating_point() else v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model.generate(**batch, do_sample=False, num_beams=1,
                                 max_new_tokens=CONFIG['caption_max_new_tokens'])
        text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        return text, dict(prompt_tokens=length, generated_tokens=int(out.shape[-1]),
                          max_new_tokens=CONFIG['caption_max_new_tokens'], do_sample=False)
    with (root/'rows.jsonl').open('x', encoding='utf-8') as f:
        for i, r in enumerate(data):
            key = case_key(r)
            wrong = lookup[(r['donor_id'], key[1], key[2])]
            # Only receiver artifacts enter these functions; no source text/image.
            with Image.open(local(a.prepared, r['received_image'])) as im:
                own = im.convert('RGB')
            with Image.open(local(a.prepared, wrong['received_image'])) as im:
                donor = im.convert('RGB')
            prompt = ('Correct this received description using the image. Preserve supported facts, '
                'do not invent details. Output only the complete corrected description. Received description: '
                +json.dumps(r['received_caption'])+' Corrected description:')
            controls = [('received_text', None, None),
                ('caption_only', own, 'Describe the visible objects, attributes, counts and spatial relationships. Do not invent details.'),
                ('joint_correct', own, prompt), ('joint_wrong_image', donor, prompt),
                ('joint_blank_image', blank, prompt)]
            for control, im, question in controls:
                text, telemetry = (r['received_caption'], {}) if im is None else generate(im, question)
                save_line(f, dict(example_id=key[0], snr_db=key[1], repeat=key[2], control=control,
                    output_text=text, output_sha256=digest(text.encode()), telemetry=telemetry,
                    prompt_sha256=None if question is None else digest(question.encode()),
                    received_caption_sha256=r['received_caption_sha256'],
                    conditioning_image_sha256=(None if im is None else
                        digest(np.asarray(im).tobytes())), donor_id=r['donor_id'] if control=='joint_wrong_image' else None,
                    source_text_at_RX=False, training_updates=0))
            print('V22 CAPTION {}/{} all five controls'.format(i+1, len(data)), flush=True)
    verify_frozen((model,), (frozen,))
    atomic(root/'identity.json', dict(prepared_sha256=sha(Path(a.prepared)/'COMPLETE.json'),
        asset_lock_sha256=sha(a.asset_lock), overlap_audit=lock['overlap_audit'],
        visual_preprocessing='pinned BLIP-2 processor; may resize/crop; not native 512 detail preservation',
        blank_control_is_language_rewrite=True, caption_only_is_not_joint_recovery=True))
    complete(root, 'caption', dict(rows=len(data)*5, frozen=True, facts_status='PENDING'))

def prompt_embeddings(pipe, text):
    """All tokens, independent 75-content-token CLIP blocks, concatenated memory.

    This is an explicit heuristic, NOT a Long-CLIP or pretrained long-context claim.
    Overflow fails rather than silently cropping a DOCCI description.
    """
    import torch
    tokenizer = pipe.tokenizer
    body = tokenizer(text, add_special_tokens=False, truncation=False, max_length=1000000)['input_ids']
    width = int(tokenizer.model_max_length)
    if width != 77:
        raise ValueError('Expected native SD CLIP77 tokenizer')
    from docci_pretrained_checks_v22.core import clip_blocks
    ids, plan = clip_blocks(body, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
    device = next(pipe.text_encoder.parameters()).device
    with torch.no_grad():
        value = pipe.text_encoder(torch.tensor(ids, device=device))[0]
    return value.reshape(1, -1, value.shape[-1]), plan

def enhance(a, root, budget):
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline
    index, data, _ = prepared(a.prepared)
    model_path, lock = asset(a, 'diffusion')
    device = cuda(); budget.gpu = True
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(str(model_path), local_files_only=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=False).to(device)
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    for m in (pipe.vae, pipe.unet, pipe.text_encoder):
        m.requires_grad_(False).eval()
    from docci_longclip_joint_v8.backbones import signature, verify_frozen
    models = (pipe.vae, pipe.unet, pipe.text_encoder)
    if pipe.safety_checker is not None:
        pipe.safety_checker.requires_grad_(False).eval()
        models += (pipe.safety_checker,)
    frozen = [signature(m) for m in models]
    lookup = {case_key(r): r for r in data}
    receiver_root = Path(a.prepared)
    receiver_identity = dict(stage='continuous', control='continuous',
        receipt_sha256=sha(receiver_root/'COMPLETE.json'))
    images = {case_key(r): (r['received_image'], r['received_image_sha256']) for r in data}
    if a.receiver_stage:
        from docci_pretrained_checks_v22.report import verify
        stage = Path(a.receiver_stage)
        if verify(stage, a.prepared)['stage'] != 'digital' or not a.receiver_control:
            raise ValueError('Explicit verified digital stage/control required')
        chosen = [r for r in rows(stage/'rows.jsonl') if r['control']==a.receiver_control]
        images = {case_key(r): (r['image_file'], r['image_sha256']) for r in chosen}
        if len(chosen) != len(data) or set(images) != set(lookup):
            raise ValueError('Digital postprocessing must cover identical complete cases')
        receiver_root = stage
        receiver_identity = dict(stage='digital', control=a.receiver_control,
            receipt_sha256=sha(stage/'COMPLETE.json'))
    (root/'images').mkdir()
    with (root/'rows.jsonl').open('x', encoding='utf-8') as f:
        for i, r in enumerate(data):
            key = case_key(r)
            other = lookup[(r['donor_id'], key[1], key[2])]
            with Image.open(local(receiver_root, images[key][0])) as im:
                received = im.convert('RGB')
            if received.size != (512, 512):
                raise ValueError('Do not resize/crop receiver canvas')
            for control, text in (('no_text', ''), ('correct_text', r['received_caption']),
                                  ('wrong_text', other['received_caption'])):
                budget()
                embeddings, plan = prompt_embeddings(pipe, text)
                negative, _ = prompt_embeddings(pipe, '')
                negative = negative.repeat(1, plan['chunks'], 1)
                # Same actual noise seed for all three interventions, no best-of-N.
                value = seed(42, 'V22-diffusion', *key) % (2**63-1)
                generator = torch.Generator(device='cpu').manual_seed(value)
                actual_steps = []
                def callback(step, timestep, latents):
                    budget()
                    actual_steps.append(int(step))
                with torch.no_grad():
                    result = pipe(prompt_embeds=embeddings, negative_prompt_embeds=negative,
                        image=received, strength=CONFIG['diffusion_strength'],
                        num_inference_steps=CONFIG['diffusion_steps'], guidance_scale=CONFIG['guidance_scale'],
                        generator=generator, num_images_per_prompt=1,
                        callback=callback, callback_steps=1)
                image = result.images[0]
                if image.size != received.size:
                    raise ValueError('Generated geometry differs; no evaluation resizing')
                path = root/'images'/(case_name(*key)+'__'+control+'.png')
                image.save(path)
                save_line(f, dict(example_id=key[0], snr_db=key[1], repeat=key[2], control=control,
                    image_file=path.relative_to(root).as_posix(), image_sha256=sha(path),
                    received_image_sha256=images[key][1],
                    prompt_sha256=digest(text.encode()), prompt_plan=plan, random_seed=value,
                    scheduler_steps_requested=CONFIG['diffusion_steps'], actual_callback_steps=len(actual_steps),
                    safety_flag=None if result.nsfw_content_detected is None else bool(result.nsfw_content_detected[0]),
                    donor_id=r['donor_id'] if control=='wrong_text' else None,
                    source_text_at_RX=False, source_image_at_RX=False, training_updates=0))
            print('V22 ENHANCE {}/{} same seed / no, correct, wrong text'.format(i+1, len(data)), flush=True)
    verify_frozen(models, frozen)
    atomic(root/'identity.json', dict(prepared_sha256=sha(Path(a.prepared)/'COMPLETE.json'),
        asset_lock_sha256=sha(a.asset_lock), overlap_audit=lock['overlap_audit'], config=CONFIG,
        receiver=receiver_identity, receiver_root=str(receiver_root.resolve()),
        actual_multimodal_symbols_with_text=13376,
        geometric_transform='none; original 512 letterbox preserved',
        long_prompt_policy='all native CLIP77 chunks concatenated; heuristic not trained long-context',
        best_of_N=False, safety_checker_disabled_by_this_code=False,
        ControlNet_implemented=False, DiffJSCC_reproduction=False))
    complete(root, 'enhance', dict(rows=len(data)*3, frozen=True, semantic_success=False))

def score(a, root, budget):
    """One frozen LPIPS model; exact same uint8 valid-view metrics for all outputs."""
    import torch
    from docci_text_guided_image_v20.perceptual import Perceptual
    from docci_pretrained_checks_v22.core import verify_files, metric
    _, base, examples = prepared(a.prepared)
    target = Path(a.target)
    receipt = verify_files(target)
    if receipt['stage'] not in ('prepare', 'digital', 'enhance'):
        raise ValueError('Only image stages can be scored')
    if receipt['stage'] != 'prepare' and read(target/'identity.json')['prepared_sha256'] != sha(Path(a.prepared)/'COMPLETE.json'):
        raise ValueError('Target uses a different preparation')
    device = cuda(); budget.gpu = True
    lpips = Perceptual(a.lpips_assets).to(device)
    with (root/'rows.jsonl').open('x', encoding='utf-8') as f, torch.no_grad():
        for r in rows(target/'rows.jsonl'):
            budget()
            ex = examples[r['example_id']]
            name = r['received_image'] if receipt['stage']=='prepare' else r['image_file']
            with Image.open(local(target, name)) as im:
                output = np.asarray(im.convert('RGB')).copy()
            with Image.open(local(a.prepared, ex['source_image'])) as im:
                truth = np.asarray(im.convert('RGB')).copy()
            m = metric(truth, output, ex['rectangle'])
            l, t, right, bottom = ex['rectangle']
            def tensor(x):
                return torch.as_tensor(x[t:bottom, l:right].copy(), device=device).permute(2, 0, 1)[None].float()/255.
            m['LPIPS'] = float(lpips(tensor(output), tensor(truth)).mean())
            save_line(f, dict(example_id=r['example_id'], snr_db=r['snr_db'], repeat=r['repeat'],
                control=r.get('control', 'continuous'), image_sha256=sha(local(target, name)), **m))
    atomic(root/'identity.json', dict(prepared_sha256=sha(Path(a.prepared)/'COMPLETE.json'),
        target_sha256=sha(target/'COMPLETE.json'), target_stage=receipt['stage'], target_path=str(target.resolve()),
        LPIPS='Alex v0.1 pinned V20 assets; uint8 saved PNG; original valid rectangle',
        evaluator_independent_of_SD_and_BLIP2=True))
    complete(root, 'score', dict(rows=len(rows(target/'rows.jsonl')), facts_status='PENDING'))
