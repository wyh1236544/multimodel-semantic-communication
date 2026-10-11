"""Bounded three-arm training plus frozen A, paired proxy and V3 waveform validation."""
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
import numpy as np
EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT))
from fso_jscc_v1 import VERSION, ARMS
from fso_jscc_v1.core import load, digest, seed_for, train_condition, grid, TEMPLATE


def main():
    import torch
    import torch.nn.functional as F
    import v10.model as module
    from v10.runtime import add_shared_args, preflight, load_config
    from v10.io import begin_run, atomic_json, atomic_checkpoint
    from v10.data import build_loader, sha256
    from v10.model import build_model
    from utils.checkpoint import load_checkpoint
    from utils.experiment import seed_everything
    from utils.metrics import batch_image_metrics, token_accuracy, ssim_per_image, sentence_bleu2
    from token_guidance.model import Guidance
    from final_compare_v1.metrics import caption_metrics
    from optical_link_v1.config import load as optical_load
    from optical_realism_v3.spec import load_plan, resources, effective_conditions
    from optical_realism_v3.transport import RealisticTransport
    from optical_realism_v3.constellation import constellation_svg
    from eval_v10 import save_image
    from fso_jscc_v1.model import ContinuousChannel, Frame, parameters, stage, gradient_groups, ENCODER
    from fso_jscc_v1.report import verify, write_report

    p = add_shared_args(argparse.ArgumentParser(description=__doc__, allow_abbrev=False))
    p.add_argument('--a-delta', required=True)
    p.add_argument('--experiment-config', default=str(TEMPLATE))
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    cfg = load(a.experiment_config)
    if a.smoke:
        cfg.update(train_samples=8, epochs=1, val_samples=2, draws=1, save_images=2,
                   physical_val_samples=2, physical_draws=1, warmup_optimizer_steps=1)
    total_steps = cfg['epochs']*math.ceil(cfg['train_samples']/cfg['grad_accum'])
    if cfg['warmup_optimizer_steps'] >= total_steps:
        raise ValueError('Warmup must end before final optimizer step; joint encoder must actually train')
    a.batch_size, a.num_workers, a.no_amp, a.fixed_mode, a.csi, a.blocks, a.seed = 1, 0, True, 0, 'pilot', 1, cfg['seed']
    if not math.isfinite(a.max_hours) or not 0 < a.max_hours <= 24:
        p.error('Overall wall-time budget must be >0 and <=24h')
    device = preflight(a)
    config = load_config(a)
    base, delta = load_checkpoint(a.checkpoint), load_checkpoint(a.a_delta)
    base_hash, delta_hash, split_hash = sha256(a.checkpoint), sha256(a.a_delta), sha256(a.split_manifest)
    if (not str(base.get('version', '')).startswith('V10-') or base.get('split_sha256') != split_hash or
            delta.get('version') != 'V10-token-guidance-quick-v1' or delta.get('variant') != 'A' or
            delta.get('base_checkpoint_sha256') != base_hash or delta.get('split_sha256') != split_hash):
        raise ValueError('Exact original V10 base + A delta + immutable matching split required')
    config['v10'] = copy.deepcopy(base['config']['v10'])
    config['v10'].update(blocks=1, csi='pilot', fixed_mode=0)
    # --no-amp alone does NOT disable a VAE loaded explicitly in FP16.
    config['model']['vae_use_fp16'] = False
    del base
    plan, _ = load_plan(repeats=1)
    _, physical_cases, _ = optical_load()
    known = {x['id']: x for x in effective_conditions(physical_cases, plan)}
    if any(c not in known for c in cfg['physical_conditions']):
        raise ValueError('Unknown V3 physical condition ID')
    physical_conditions = [known[c] for c in cfg['physical_conditions']]
    resource = resources(physical_cases[physical_conditions[0]['case']], plan)
    for optical in physical_cases.values():
        optical['waveform']['policy'] = plan['digital_policy']
    directory, identity = begin_run(a.output_dir, a, config, EXT)
    identity.update(version=VERSION, a_delta_sha256=delta_hash, experiment_config_sha256=digest(a.experiment_config),
                    new_source_sha256={f.name: digest(f) for f in sorted(Path(__file__).parent.iterdir()) if f.suffix in ('.py', '.json', '.md')},
                    physical_source_sha256={str(f.relative_to(EXT)): digest(f) for name in ('optical_realism_v3', 'optical_link_v1', 'final_compare_v1', 'token_guidance')
                        for f in sorted((EXT/name).iterdir()) if f.suffix in ('.py', '.json')},
                    gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None)
    atomic_json(directory/'run_identity.json', identity)
    started = time.monotonic()
    def budget():
        if time.monotonic()-started > a.max_hours*3600:
            raise TimeoutError('Total time budget reached; partial records retained, no COMPLETE')
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    try:
        print('Loading exact A; frozen pretrained VAE/CLIP; R=1 continuous payload; no original-text RX branch.', flush=True)
        model, report = build_model(config, device, a.checkpoint)
        if report != {'v10_exact': True}:
            raise ValueError('Exact base load required')
        model.float()
        guide = Guidance(model)
        if set(delta['delta']) != {n for n, _ in guide.trainable('A')}:
            raise ValueError('A delta keys mismatch')
        model.load_state_dict(dict(model.state_dict(), **delta['delta']), strict=True)
        del delta
        guide.set('A')
        model.base.channel = ContinuousChannel(model.base.channel, cfg).to(device)
        named = parameters(model)
        initial = {n: q.detach().cpu().clone() for n, q in named}
        atomic_json(directory/'checkpoint_load.json', dict(base_exact=True, a_delta_exact=True,
            base_sha256=base_hash, a_delta_sha256=delta_hash, mapping_initialization='zero residual identity',
            new_trainable={n: q.numel() for n, q in named}, total_parameters=sum(q.numel() for q in model.parameters())))

        def cache(split, limit):
            loader = build_loader(config, a.split_manifest, split, limit, cfg['seed'])
            loader.dataset.caption_mode = 'first'
            items = []
            with torch.no_grad():
                for batch in loader:
                    budget()
                    image = batch['images']
                    latent = model.base.vae_wrapper.encode_images(image.to(device)).detach().cpu()
                    items.append(dict(image_id=int(batch['image_ids'][0]), text=batch['texts'][0], images=image, latent=latent,
                        source_sha256=hashlib.sha256(image.numpy().tobytes()+batch['texts'][0].encode()).hexdigest()))
            if len(items) != limit or len({x['image_id'] for x in items}) != limit:
                raise ValueError('Incomplete/duplicate cached image IDs')
            return items
        model.eval()
        train, validation = cache('train', cfg['train_samples']), cache('validation', cfg['val_samples'])
        conditions = [('clean', 40.), ('awgn', 5.), ('fso_moderate', 0.), ('fso_strong', -5.)] if a.smoke else grid(cfg)
        design = dict(version=VERSION, config=cfg, smoke=a.smoke, split='validation', arms=list(ARMS),
            train_ids=[x['image_id'] for x in train], validation_ids=[x['image_id'] for x in validation],
            validation_source_sha256={str(x['image_id']): x['source_sha256'] for x in validation},
            proxy_conditions=[dict(id='{}_snr{:g}'.format(pro, snr), profile=pro, snr=snr) for pro, snr in conditions],
            physical_conditions=physical_conditions, physical_plan=plan, physical_cases=physical_cases,
            physical_resource=resource, candidate_parameters={n: q.numel() for n, q in named},
            optimizer_steps_per_trained_arm=total_steps, train_forward_steps=cfg['epochs']*cfg['train_samples'],
            checkpoint_selection='last fixed optimizer step; no validation/test selection',
            seed_replicates=1, channel_draws_are_independent_R1_trials_not_combined_repetitions=True,
            inherited_A_was_previously_noisy_trained=True, all_arms_same_architecture=True,
            loss='1.5 latentMSE+.75 pixelMSE+.35(1-SSIM)+.5 maskedTextCE+samePAPRpenalty',
            primary='M3-M2 PSNR, FSO weak/moderate/strong at nominal -5/0/5 dB, image-cluster bootstrap; exploratory single training replicate',
            no_original_caption_at_receiver=True, full_text_metrics_and_CLIP77_capacity_reported=True)
        atomic_json(directory/'design.json', design)
        (directory/'images').mkdir(); (directory/'constellations').mkdir()

        def forward(item, context):
            budget()
            with patch.object(module, 'LinkFrame', lambda *args, **kw: Frame(*args, context=context, config=cfg, **kw)):
                # Public profile argument is only an old interface placeholder.
                # Actual stochastic channel lives in Frame, not in model's SNR input.
                return model(item['images'].to(device), [item['text']], 'awgn', 0.,
                             vae_latents_override=item['latent'].to(device))

        with (directory/'samples.jsonl').open('x', encoding='utf-8') as samples, (directory/'events.jsonl').open('x', encoding='utf-8') as events:
            rows_done = 0
            for arm in ARMS:
                budget()
                with torch.no_grad():
                    for n, q in named:
                        q.copy_(initial[n].to(device))
                arm_dir = directory/('train_'+arm)
                arm_dir.mkdir()
                train_start, steps, updates = time.monotonic(), 0, 0
                update_squared = None
                counts = dict(source_encoder=0, tx_mapping=0, rx_mapping=0, decoder=0)
                if arm != ARMS[0]:
                    optimizer = torch.optim.AdamW([
                        dict(params=[q for n, q in named if n.startswith(ENCODER)], lr=cfg['encoder_lr']),
                        dict(params=[q for n, q in named if n.startswith('base.channel.')], lr=cfg['mapping_lr']),
                        dict(params=[q for n, q in named if not n.startswith(ENCODER+('base.channel.',))], lr=cfg['decoder_lr'])], weight_decay=1e-5)
                    optimizer.zero_grad(set_to_none=True)
                    with (arm_dir/'history.jsonl').open('x', encoding='utf-8') as history:
                        for epoch in range(cfg['epochs']):
                            order = list(range(len(train))); random.Random(seed_for(cfg['seed'], epoch, 'order')).shuffle(order)
                            for pos, ix in enumerate(order):
                                budget()
                                if time.monotonic()-train_start > cfg['arm_max_hours']*3600:
                                    raise TimeoutError('Per-arm training cap reached: '+arm)
                                stage(model, warm=updates < cfg['warmup_optimizer_steps'])
                                seed_everything(seed_for(cfg['seed'], epoch, pos, 'network') % (2**31-1))
                                pro, snr = train_condition(cfg, epoch, pos)
                                profile = 'clean' if arm == ARMS[1] else 'awgn' if arm == ARMS[2] else pro
                                item = train[ix]
                                context = dict(profile=profile, snr=snr,
                                    field_seed=seed_for(cfg['seed'], item['image_id'], epoch, pos, 'trainfield'),
                                    noise_seed=seed_for(cfg['seed'], item['image_id'], epoch, pos, 'trainnoise'))
                                out = forward(item, context)
                                pixel = F.mse_loss(out['image'].float(), item['images'].to(device).float())
                                latent = F.mse_loss(out['recovered_vae_latents'].float(), item['latent'].to(device).float())
                                structure = 1-ssim_per_image(out['image'], item['images'].to(device)).mean()
                                logits, ids = out['text_logits'].float(), out['text_input_ids']
                                length = min(logits.shape[1], ids.shape[1])
                                mask = out['text_attention_mask'][:, :length].float()
                                ce = F.cross_entropy(logits[:, :length].reshape(-1, logits.shape[-1]), ids[:, :length].reshape(-1), reduction='none').reshape_as(mask)
                                text = (ce*mask).sum()/mask.sum().clamp_min(1)
                                loss = 1.5*latent+.75*pixel+.35*structure+.5*text+cfg['papr_weight']*model.base.channel.papr_penalty
                                if not torch.isfinite(loss):
                                    raise FloatingPointError('Nonfinite loss: '+arm)
                                divisor = min(cfg['grad_accum'], len(order)-(pos//cfg['grad_accum'])*cfg['grad_accum'])
                                (loss/divisor).backward()
                                steps += 1
                                if (pos+1) % cfg['grad_accum'] == 0 or pos+1 == len(order):
                                    norm = torch.nn.utils.clip_grad_norm_([q for _, q in named], 1.)
                                    if not torch.isfinite(norm):
                                        raise FloatingPointError('Nonfinite gradient: '+arm+'; no silently skipped steps')
                                    grad = gradient_groups(named)
                                    for key in counts:
                                        counts[key] += int(grad[key] > 0)
                                    optimizer.step(); optimizer.zero_grad(set_to_none=True); updates += 1
                                record = dict(model.last_frame.last_stats, arm=arm, epoch=epoch+1, forward_step=steps, optimizer_step=updates,
                                    image_id=item['image_id'], profile=profile, nominal_snr_db=snr, loss=float(loss.detach()),
                                    latent_mse=float(latent.detach()), pixel_mse=float(pixel.detach()), text_ce=float(text.detach()),
                                    elapsed_seconds=time.monotonic()-train_start)
                                history.write(json.dumps(record, allow_nan=False)+'\n'); history.flush()
                                if steps == 1 or steps % 20 == 0:
                                    print('TRAIN {} epoch={}/{} forward={}/{} opt={}/{} loss={:.4f} elapsed={:.1f}s'.format(
                                        arm, epoch+1, cfg['epochs'], steps, design['train_forward_steps'], updates, total_steps, record['loss'], record['elapsed_seconds']), flush=True)
                                    atomic_json(directory/'progress.json', dict(record, stage='training'))
                                if updates in (100, 200) and (pos+1) % cfg['grad_accum'] == 0:
                                    sync()
                                    elapsed = time.monotonic()-train_start
                                    atomic_json(arm_dir/'timing_projection_{}.json'.format(updates), dict(
                                        optimizer_steps=updates, elapsed_seconds=elapsed,
                                        projected_training_seconds=elapsed*total_steps/updates,
                                        estimate_not_promise=True, excludes_validation=True))
                                del out, loss, logits, ce, pixel, latent, structure, text
                    if steps != design['train_forward_steps'] or updates != total_steps or min(counts.values()) <= 0:
                        raise RuntimeError('Incomplete steps or absent encoder/TX/RX/decoder gradient: '+str(counts))
                    changed = {n: float((q.detach().cpu()-initial[n]).square().sum()) for n, q in named}
                    for prefixes in [('base.multimodal_encoder.',), ('base.channel.tx_map.',), ('base.channel.rx_map.',), ('base.joint_decoder.',)]:
                        if sum(v for n, v in changed.items() if n.startswith(prefixes)) <= 0:
                            raise RuntimeError('Required module received no parameter update: '+str(prefixes))
                    update_squared = dict(source_encoder=sum(v for n, v in changed.items() if n.startswith('base.multimodal_encoder.')),
                        tx_mapping=sum(v for n, v in changed.items() if n.startswith('base.channel.tx_map.')),
                        rx_mapping=sum(v for n, v in changed.items() if n.startswith('base.channel.rx_map.')),
                        decoder=sum(v for n, v in changed.items() if n.startswith('base.joint_decoder.')))
                    payload = dict(version=VERSION, arm=arm, config=cfg,
                        delta={n: q.detach().cpu().clone() for n, q in named}, optimizer=optimizer.state_dict(),
                        base_checkpoint_sha256=base_hash, a_delta_sha256=delta_hash, split_sha256=split_hash,
                        steps=steps, optimizer_steps=updates, nonzero_gradient_steps=counts,
                        parameter_update_squared_l2_by_group=update_squared)
                    atomic_checkpoint(arm_dir/'last_delta.pth', payload)
                    del optimizer, payload
                sync()
                atomic_json(arm_dir/'COMPLETE.json', dict(arm=arm, steps=steps, optimizer_steps=updates,
                    wall_seconds=time.monotonic()-train_start, nonzero_gradient_steps=counts,
                    trained=arm != ARMS[0], parameter_update_squared_l2_by_group=update_squared,
                    checkpoint_sha256=digest(arm_dir/'last_delta.pth') if arm != ARMS[0] else None))
                model.requires_grad_(False); model.eval()

                def emit(item, engine, condition, draw, context, index):
                    nonlocal rows_done
                    output = forward(item, context)
                    metric = batch_image_metrics(output['image'], item['images'].to(device))
                    caption = model.base.decode_text_logits(output['text_logits'])[0]
                    correct, count = token_accuracy(output['text_logits'], output['text_input_ids'], output['text_attention_mask'])
                    stats = model.last_frame.last_stats
                    row = dict(stats, arm=arm, engine=engine, condition=condition, draw=draw,
                        image_id=item['image_id'], source_sha256=item['source_sha256'],
                        field_seed=context['field_seed'], noise_seed=context['noise_seed'],
                        psnr=float(metric['psnr'][0]), ssim=float(metric['ssim'][0]), token_acc=correct/max(count, 1),
                        token_count=count, bleu2=sentence_bleu2(item['text'], caption),
                        source_caption_evaluation_only=item['text'], received_caption=caption,
                        neural_input_capacity_tokens=int(output['text_input_ids'].shape[1]),
                        source_utf8_bytes=len(item['text'].encode()), source_rgb8_bytes=item['images'].numel(),
                        raw_utf8_wire_bytes=0, analog_repeats=1, pre_fec_ber=None, post_fec_ber=None,
                        **caption_metrics(item['text'], caption))
                    row['receiver_available'] = bool(model.last_frame.header_ok[0])
                    target = output['common_text_semantic_target'].float()
                    row['common_to_source_text_cosine'] = float(F.cosine_similarity(
                        output['common_receiver_semantic'].float(), target, dim=-1)[0])
                    row['receiver_text_to_source_text_cosine'] = float(F.cosine_similarity(
                        output['receiver_text_semantic'].float(), target, dim=-1)[0])
                    row['semantic_metric_note'] = 'inherited CLIP-family latent alignment, NOT independent object/relationship correctness'
                    if index < cfg['save_images']:
                        source = directory/'images'/('{}__source.png'.format(item['image_id']))
                        if not source.exists():
                            save_image(item['images'][0], source)
                        row['source_image_file'] = 'images/'+source.name
                        row['image_file'] = 'images/{}__{}__{}__{}__d{}.png'.format(item['image_id'], arm, engine, condition, draw)
                        save_image(output['image'][0], directory/row['image_file'])
                        for key in ('image_file', 'source_image_file'):
                            row[key+'_sha256'] = digest(directory/row[key])
                    transport = model.last_frame.transport
                    if transport is not None:
                        row['event_count'] = len(transport.events)
                        for e in transport.events:
                            events.write(json.dumps(dict(e, arm=arm, engine=engine, condition=condition, image_id=item['image_id'], draw=draw), allow_nan=False)+'\n')
                        events.flush()
                        if index < cfg['save_images'] and transport.constellations:
                            row['constellation_file'] = 'constellations/{}__{}__{}__d{}.svg'.format(item['image_id'], arm, condition, draw)
                            constellation_svg(transport.constellations, directory/row['constellation_file'])
                            row['constellation_file_sha256'] = digest(directory/row['constellation_file'])
                    samples.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); samples.flush()
                    rows_done += 1
                    if rows_done == 1 or rows_done % 20 == 0:
                        print('EVAL {} rows={} {} {} PSNR={:.2f} WER={:.3f}'.format(arm, rows_done, engine, condition, row['psnr'], row['normalized_wer']), flush=True)
                        atomic_json(directory/'progress.json', dict(stage='evaluation', arm=arm, rows=rows_done, elapsed_seconds=time.monotonic()-started))
                    del output

                with torch.no_grad():
                    for index, item in enumerate(validation):
                        for pro, snr in conditions:
                            condition = '{}_snr{:g}'.format(pro, snr)
                            for draw in range(1 if pro == 'clean' else cfg['draws']):
                                context = dict(profile=pro, snr=snr,
                                    field_seed=seed_for(cfg['seed'], item['image_id'], pro, draw, 'valfield'),
                                    noise_seed=seed_for(cfg['seed'], item['image_id'], condition, draw, 'valnoise'))
                                emit(item, 'proxy', condition, draw, context, index)
                        if index < cfg['physical_val_samples']:
                            for c in physical_conditions:
                                for draw in range(cfg['physical_draws']):
                                    field = seed_for(cfg['seed'], item['image_id'], c['case'], draw, 'physicalfield')
                                    noise = seed_for(cfg['seed'], item['image_id'], c['case'], draw, 'physicalnoise')
                                    transport = RealisticTransport(physical_cases[c['case']], field, noise, c['launch_signal_dbm'],
                                        resource['slot_samples'], plan['laser_max_dbm'], plan['hardware'])
                                    emit(item, 'waveform_v3', c['id'], draw, dict(field_seed=field, noise_seed=noise, transport=transport), index)
                budget()
        verify(directory)
        write_report(directory)
        identity['status'] = 'complete'; atomic_json(directory/'run_identity.json', identity)
        result_files = [f for f in directory.rglob('*') if f.is_file() and f.name != 'COMPLETE.json' and f.suffix != '.pth']
        # Include arm COMPLETE records and checkpoint hashes in the final manifest.
        result_files += [directory/('train_'+arm)/'COMPLETE.json' for arm in ARMS]
        checkpoint_hashes = {str(f.relative_to(directory)): digest(f) for f in directory.glob('train_*/last_delta.pth')}
        atomic_json(directory/'COMPLETE.json', dict(version=VERSION, rows=rows_done,
            wall_seconds=time.monotonic()-started,
            result_sha256={str(f.relative_to(directory)): digest(f) for f in sorted(set(result_files))},
            checkpoint_sha256=checkpoint_hashes))
        verify(directory, complete=True)
        print('COMPLETE: '+str(directory/'report.html')+' | collect records, NOT weights.', flush=True)
    except BaseException as exc:
        identity.update(status='failed', error=repr(exc), elapsed_seconds=time.monotonic()-started)
        atomic_json(directory/'run_identity.json', identity)
        atomic_json(directory/'FAILED.json', dict(error=repr(exc), partial_artifacts_preserved=True))
        raise


if __name__ == '__main__':
    main()
