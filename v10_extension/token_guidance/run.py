"""Four bounded fine-tunings and paired validation. No test or raw-text side channel."""
import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from unittest.mock import patch

EXT = Path(__file__).resolve().parents[1]
PROJECT = EXT.parent
sys.path.insert(0, str(EXT))
from token_guidance.spec import VERSION, VARIANTS, DIAGNOSTICS, SETTINGS, TRAIN_CONDITIONS, conditions, donor_map, expected_rows


def parser():
    from v10.runtime import add_shared_args
    p = add_shared_args(argparse.ArgumentParser(description=__doc__, allow_abbrev=False))
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--train-samples', type=int, default=512)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--val-samples', type=int, default=32)
    p.add_argument('--save-images', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--guidance-lr', type=float, default=5e-5)
    p.add_argument('--grad-accum', type=int, default=2)
    return p


def main(args):
    import torch
    import torch.nn.functional as F
    import numpy as np
    import v10.model as model_module
    from v10.runtime import preflight, load_config
    from v10.io import begin_run, atomic_json, atomic_checkpoint
    from v10.data import build_loader, sha256
    from v10.model import build_model
    from utils.checkpoint import load_checkpoint
    from utils.experiment import seed_everything
    from utils.metrics import batch_image_metrics, token_accuracy, sentence_bleu2, ssim_per_image
    from business_link.metrics import text_metrics
    from business_link.plots import constellation_svg
    from eval_v10 import save_image
    from token_guidance.model import Guidance, TokenFrame, frame_context
    from token_guidance.report import analyze, write_report

    if args.smoke:
        args.train_samples, args.epochs, args.val_samples, args.save_images = 8, 1, 2, 2
    if min(args.train_samples, args.epochs, args.val_samples, args.grad_accum, args.lr, args.guidance_lr) <= 0 or args.val_samples < 2 or args.save_images < 0:
        raise ValueError('Invalid nonpositive budget or too few validation images')
    args.batch_size, args.num_workers, args.no_amp, args.fixed_mode = 1, 0, True, 0
    device = preflight(args)
    config = load_config(args)
    metadata = load_checkpoint(args.checkpoint)
    if (not str(metadata.get('version', '')).startswith('V10-')
            or metadata.get('split_sha256') != sha256(args.split_manifest)):
        raise ValueError('Need original exact V10 checkpoint and matching frozen split')
    config['v10'] = copy.deepcopy(metadata['config']['v10'])
    config['v10'].update(blocks=1, csi='pilot', fixed_mode=0)
    del metadata
    directory, identity = begin_run(args.output_dir, args, config, EXT)
    identity['version'] = VERSION
    identity['new_source_sha256'] = {p.name: sha256(p) for p in sorted(Path(__file__).parent.iterdir()) if p.suffix in ('.py', '.md', '.json')}
    identity['physical_source_sha256'] = {str(p.relative_to(EXT)): sha256(p) for folder in ('business_link', 'physical_link') for p in sorted((EXT/folder).glob('*.py'))}
    atomic_json(directory/'run_identity.json', identity)
    started, rows = time.monotonic(), []
    deadline = started+args.max_hours*3600

    def budget():
        if time.monotonic() >= deadline:
            raise TimeoutError('Overall time budget reached; artifacts retained, no COMPLETE')

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    try:
        print('Loading existing checkpoint; old code, weights and results remain unchanged.', flush=True)
        model, report = build_model(config, device, args.checkpoint)
        if report != {'v10_exact': True}:
            raise ValueError('Strict original checkpoint loading required')
        atomic_json(directory/'checkpoint_load.json', report)
        model.eval()
        guide = Guidance(model)
        # Reuse precisely the SAME initialization, including existing guidance
        # weights, in every factorial training arm. No module reinitialization.
        initial = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}

        def cache(split, limit):
            seed_everything(args.seed)
            loader = build_loader(config, args.split_manifest, split, limit, args.seed)
            loader.dataset.caption_mode = 'first'  # freeze caption, not random epoch paraphrases
            items = []
            with torch.no_grad():
                for batch in loader:
                    budget()
                    images = batch['images'].to(device)
                    latent = model.base.vae_wrapper.encode_images(images).detach().cpu()
                    items.append(dict(image_id=int(batch['image_ids'][0]), text=batch['texts'][0],
                                      images=batch['images'], latent=latent))
            return items

        train = cache('train', args.train_samples)
        validation = cache('validation', args.val_samples)
        val_ids = [x['image_id'] for x in validation]
        donors = donor_map(val_ids)
        items_by_id = {x['image_id']: x for x in validation}
        design = dict(version=VERSION, smoke=args.smoke, train_ids=[x['image_id'] for x in train],
                      validation_ids=val_ids, donor_ids=donors, variants=VARIANTS,
                      conditions=conditions(args.smoke), training_conditions=TRAIN_CONDITIONS,
                      train_samples=args.train_samples, epochs=args.epochs, physical_settings=SETTINGS,
                      primary='image-averaged moderate/strong FSO physical20dB; D-A',
                      loss='1.5 latent MSE + 0.75 pixel MSE + 0.35 (1-SSIM) + 0.5 masked text CE',
                      backward='identity STE for accepted quantized objects; zero sender gradient for CRC erasure',
                      checkpoint='last fixed step, no validation selection', dropout='disabled in all arms',
                      original_checkpoint_sha256=identity['checkpoint_sha256'],
                      text_branch='C/I/T semantic tokens ONLY; raw UTF8 branch disabled',
                      training_seed=args.seed, training_replicates=1,
                      train_items=[dict(image_id=x['image_id'], caption_sha256=hashlib.sha256(x['text'].encode()).hexdigest(), image_sha256=hashlib.sha256(x['images'].numpy().tobytes()).hexdigest(), cached_vae_latent_sha256=hashlib.sha256(x['latent'].numpy().tobytes()).hexdigest()) for x in train],
                      validation_items=[dict(image_id=x['image_id'], caption_sha256=hashlib.sha256(x['text'].encode()).hexdigest(), image_sha256=hashlib.sha256(x['images'].numpy().tobytes()).hexdigest(), cached_vae_latent_sha256=hashlib.sha256(x['latent'].numpy().tobytes()).hexdigest(), rgb8_bytes=x['images'].numel(), source_utf8_bytes=len(x['text'].encode())) for x in validation])
        atomic_json(directory/'design.json', design)

        def forward(item, cond, phase, epoch=0):
            budget()
            context = frame_context(args.seed, item['image_id'], phase, cond, epoch)
            def factory(*values, **kwargs):
                return TokenFrame(*values, context=context, **kwargs)
            # NumPy field/noise identities do not depend on variant or hooks.
            with patch.object(model_module, 'LinkFrame', factory):
                out = model(item['images'].to(device), [item['text']], cond[1], cond[2],
                            vae_latents_override=item['latent'].to(device))
            return out, context

        timings = {}
        delta_paths = {}
        for variant in ('A', 'B', 'C', 'D'):
            budget()
            model.load_state_dict(initial, strict=True)
            params = guide.trainable(variant)
            core = [p for n, p in params if not ('text_to_image_fusion' in n or 'decoder_text_to_image_refiner' in n)]
            specific = [p for n, p in params if ('text_to_image_fusion' in n or 'decoder_text_to_image_refiner' in n)]
            optimizer = torch.optim.AdamW([dict(params=core, lr=args.lr), dict(params=specific, lr=args.guidance_lr)], weight_decay=1e-5)
            arm = directory/('train_'+variant)
            arm.mkdir()
            atomic_json(arm/'trainable.json', {n: p.numel() for n, p in params})
            optimizer.zero_grad(set_to_none=True)
            sync()
            arm_start, steps, optimizer_steps, delivered = time.monotonic(), 0, 0, 0
            gradient_steps = dict(sender=0, receiver=0)
            with (arm/'history.jsonl').open('x', encoding='utf-8') as history:
                for epoch in range(args.epochs):
                    order = list(range(len(train)))
                    random.Random(args.seed+epoch).shuffle(order)
                    for pos, ix in enumerate(order):
                        seed_everything(args.seed+epoch*100003+pos)
                        cond = ('physical',)+TRAIN_CONDITIONS[pos % len(TRAIN_CONDITIONS)]
                        item = train[ix]
                        out, _ = forward(item, cond, 'train', epoch)
                        image = item['images'].to(device)
                        pixel = F.mse_loss(out['image'].float(), image.float())
                        latent = F.mse_loss(out['recovered_vae_latents'].float(), out['vae_latents'].detach().float())
                        structure = 1-ssim_per_image(out['image'], image).mean()
                        logits, ids = out['text_logits'].float(), out['text_input_ids']
                        length = min(logits.size(1), ids.size(1))
                        mask = out['text_attention_mask'][:, :length].float()
                        ce = F.cross_entropy(logits[:, :length].reshape(-1, logits.size(-1)), ids[:, :length].reshape(-1), reduction='none').reshape_as(mask)
                        text = (ce*mask).sum()/mask.sum().clamp_min(1)
                        loss = 1.5*latent+.75*pixel+.35*structure+.5*text
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite loss, variant '+variant)
                        divisor = min(args.grad_accum, len(order)-(pos//args.grad_accum)*args.grad_accum)
                        (loss/divisor).backward()
                        if (pos+1) % args.grad_accum == 0 or pos+1 == len(order):
                            for label, prefix in [('sender', 'base.text_to_image_fusion.'), ('receiver', 'base.decoder_text_to_image_refiner.')]:
                                if any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for n, p in params if n.startswith(prefix)):
                                    gradient_steps[label] += 1
                            norm = torch.nn.utils.clip_grad_norm_([p for _, p in params], 1.)
                            if not torch.isfinite(norm):
                                raise FloatingPointError('Nonfinite gradient, variant '+variant)
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                            optimizer_steps += 1
                        ok = bool(model.last_frame.header_ok[0])
                        delivered += int(ok)
                        steps += 1
                        value = float(loss.detach())
                        history.write(json.dumps(dict(epoch=epoch+1, step=pos+1, image_id=item['image_id'], condition=cond, loss=value, feature_crc_accepted=ok, elapsed_seconds=time.monotonic()-arm_start), allow_nan=False)+'\n')
                        history.flush()
                        del out, loss, ce, logits, pixel, latent, structure, text
                        if steps == 1 or steps % 20 == 0:
                            print('TRAIN {} epoch={}/{} step={}/{} loss={:.4f} delivered={}/{} elapsed={:.1f}s'.format(variant, epoch+1, args.epochs, pos+1, len(order), value, delivered, steps, time.monotonic()-arm_start), flush=True)
                sync()
            if steps != args.train_samples*args.epochs or optimizer_steps != args.epochs*math.ceil(args.train_samples/args.grad_accum):
                raise RuntimeError('Incomplete training arm')
            # Delta preserves original names and requires exact original base SHA.
            delta = {n: p.detach().cpu().clone() for n, p in params}
            updates = {}
            for label, prefix in [('sender', 'base.text_to_image_fusion.'), ('receiver', 'base.decoder_text_to_image_refiner.')]:
                updates[label] = sum(float((p-initial[n]).square().sum()) for n, p in delta.items() if n.startswith(prefix))
            for label, enabled in zip(('sender', 'receiver'), VARIANTS[variant]):
                if enabled and (gradient_steps[label] == 0 or updates[label] <= 0):
                    raise RuntimeError('Enabled {} guide received no gradient/update in {}; keep failed run evidence'.format(label, variant))
            payload = dict(version=VERSION, variant=variant, delta=delta,
                           base_checkpoint_sha256=identity['checkpoint_sha256'], split_sha256=identity['split_sha256'],
                           config=config, steps=steps, optimizer_steps=optimizer_steps)
            delta_paths[variant] = arm/'last_delta.pth'
            atomic_checkpoint(delta_paths[variant], payload)
            timing = dict(steps=steps, optimizer_steps=optimizer_steps, wall_seconds=time.monotonic()-arm_start,
                          feature_delivery_rate=delivered/steps, parameters=sum(p.numel() for _, p in params),
                          guide_nonzero_gradient_steps=gradient_steps, guide_parameter_update_squared_l2=updates)
            timings[variant] = timing
            atomic_json(arm/'COMPLETE.json', timing)
            del optimizer, delta, payload

        model.requires_grad_(False)
        model.eval()
        image_dir, constellation_dir = directory/'images', directory/'constellations'
        image_dir.mkdir()
        constellation_dir.mkdir()

        def record(out, item, cond, variant, diagnostic, context, index, donor_id, donor_ok):
            frame = model.last_frame
            metrics = batch_image_metrics(out['image'], item['images'].to(device))
            caption = model.base.decode_text_logits(out['text_logits'])[0]
            correct, count = token_accuracy(out['text_logits'], out['text_input_ids'], out['text_attention_mask'])
            row = dict(frame.last_stats, variant=variant, diagnostic=diagnostic,
                       image_id=item['image_id'], channel_kind=cond[0], profile=cond[1], nominal_snr_db=cond[2],
                       noise_seed=context['noise_seed'], field_seed=context['field_seed'],
                       psnr=float(metrics['psnr'][0]), ssim=float(metrics['ssim'][0]),
                       token_acc=correct/max(count, 1), token_correct=correct, token_count=count,
                       bleu2=sentence_bleu2(item['text'], caption),
                       source_caption_evaluation_only=item['text'], received_neural_caption=caption,
                       donor_image_id=donor_id, donor_crc_accepted=bool(donor_ok),
                       source_rgb8_bytes=item['images'].numel(), source_utf8_bytes=len(item['text'].encode()),
                       tx_guide_gate=float(out['text_image_gate_mean']), rx_guide_gate=float(out['decoder_text_image_gate_mean']),
                       **text_metrics(item['text'], caption))
            suffix = '{}__{}__{}__{}__{}__snr{}'.format(item['image_id'], variant, diagnostic, cond[0], cond[1], cond[2])
            if index < args.save_images:
                source_path = image_dir/('{}__source.png'.format(item['image_id']))
                if not source_path.exists():
                    save_image(item['images'][0], source_path)
                row['image_file'] = 'images/'+suffix+'.png'
                save_image(out['image'][0], directory/row['image_file'])
                if frame.transport is not None and diagnostic == 'matched':
                    constellation_svg(frame.transport.constellations, constellation_dir/(suffix+'.svg'))
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
            if frame.transport is not None:
                for event in frame.transport.events:
                    packets.write(json.dumps(dict(event, variant=variant, diagnostic=diagnostic,
                                                  image_id=item['image_id'], profile=cond[1], nominal_snr_db=cond[2]), allow_nan=False)+'\n')
                packets.flush()
            rows.append(row)
            if len(rows) == 1 or len(rows) % 20 == 0:
                print('EVAL rows={}/{} {} {} {} elapsed={:.1f}s'.format(len(rows), expected_rows(args.val_samples, args.smoke), variant, diagnostic, cond, time.monotonic()-started), flush=True)

        with (directory/'samples.jsonl').open('x', encoding='utf-8') as stream, (directory/'packets.jsonl').open('x', encoding='utf-8') as packets:
            evaluation_start = time.monotonic()
            for variant in ('OLD', 'A', 'B', 'C', 'D'):
                model.load_state_dict(initial, strict=True)
                if variant != 'OLD':
                    trained = load_checkpoint(delta_paths[variant])
                    if trained['base_checkpoint_sha256'] != identity['checkpoint_sha256']:
                        raise ValueError('Delta has wrong base checkpoint')
                    wanted = {n for n, _ in guide.trainable(variant)}
                    if set(trained['delta']) != wanted:
                        raise ValueError('Missing/unexpected delta parameters')
                    model.load_state_dict(dict(initial, **trained['delta']), strict=True)
                    del trained
                model.requires_grad_(False)
                model.eval()
                guide.set(variant)
                memories = {}
                with torch.no_grad():
                    for cond in conditions(args.smoke):
                        for index, item in enumerate(validation):
                            guide.set(variant)
                            out, context = forward(item, cond, 'validation')
                            memories[(item['image_id'], cond)] = (guide.memory.clone(), bool(model.last_frame.header_ok[0]))
                            record(out, item, cond, variant, 'matched', context, index, None, True)
                        if variant == 'D':
                            for diagnostic in DIAGNOSTICS:
                                for index, item in enumerate(validation):
                                    donor = items_by_id[donors[item['image_id']]]
                                    memory, donor_ok = memories[(donor['image_id'], cond)]
                                    sender = None
                                    if diagnostic == 'tx_wrong':
                                        encoded = model.base.text_encoder([donor['text']], device)
                                        sender = (encoded['tokens'], encoded['attention_mask'])
                                    guide.set('D', diagnostic, sender=sender,
                                              receiver=(memory if diagnostic == 'rx_wrong' else None), receiver_ok=donor_ok)
                                    out, context = forward(item, cond, 'validation')
                                    record(out, item, cond, 'D', diagnostic, context, index, donor['image_id'], donor_ok)
            sync()
            evaluation_seconds = time.monotonic()-evaluation_start

        summary = analyze(rows, design)
        write_report(directory, summary)
        total = time.monotonic()-started
        atomic_json(directory/'timing.json', dict(training=timings, evaluation_seconds=evaluation_seconds,
                    total_wall_seconds=total,
                    quick_estimate_from_smoke_seconds=(sum(t['wall_seconds']/t['steps']*1024 for t in timings.values())+
                                                     evaluation_seconds/len(rows)*expected_rows(32)+120 if args.smoke else None),
                    estimate_note='rough extrapolation only; includes physical CPU decoding, not hardware optical time'))
        identity['status'] = 'complete_smoke' if args.smoke else 'complete_quick_screen'
        atomic_json(directory/'run_identity.json', identity)
        atomic_json(directory/'COMPLETE.json', dict(version=VERSION, rows=len(rows), expected_rows=expected_rows(args.val_samples, args.smoke),
                    four_trainings_complete=True, wall_seconds=total, smoke_only=args.smoke,
                    experiment_complete_not_proof_of_effectiveness=True))
        print('COMPLETE: '+str(directory)+'; '+summary['direction_status'], flush=True)
    except BaseException as exc:
        identity.update(status='failed_or_interrupted', error=str(exc), exception=type(exc).__name__)
        atomic_json(directory/'run_identity.json', identity)
        raise


if __name__ == '__main__':
    main(parser().parse_args())
