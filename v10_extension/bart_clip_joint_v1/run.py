"""Real offline GPU1 diagnostics; gated matched joint/separate FSO screen."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
EXT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXT))
from bart_clip_joint_v1 import VERSION, CANDIDATES, ARMS
from bart_clip_joint_v1.core import load, TEMPLATE, digest, quality_gate, make_row, tensor_signature, seed_for


def main():
    import torch
    import numpy as np
    from v10.runtime import add_shared_args, preflight
    from v10.io import begin_run, atomic_json, atomic_checkpoint
    from text_jscc_v1_fix1.legacy import load_reference
    from bart_clip_joint_v1.backbones import load_bart, load_clip
    from bart_clip_joint_v1.model import JointBridge
    from bart_clip_joint_v1.data import cache, source_wire
    from bart_clip_joint_v1.channel import clean, fidelity
    from bart_clip_joint_v1.workflow import train, memory_controls, frozen_signatures, verify_frozen
    from bart_clip_joint_v1.report import report, seal, verify
    from fso_fluctuating_v2.calibrate import read_bank
    from fso_fluctuating_v2.core import paths, transport, environment
    from optical_realism_v3.constellation import constellation_svg
    from utils.metrics import batch_image_metrics
    from utils.checkpoint import load_checkpoint
    from eval_v10 import save_image

    p = add_shared_args(argparse.ArgumentParser(description=__doc__, allow_abbrev=False))
    p.add_argument('--source-run', required=True); p.add_argument('--legacy-run', required=True)
    p.add_argument('--a-delta', required=True); p.add_argument('--bart-path', required=True)
    p.add_argument('--calibration', required=True); p.add_argument('--experiment-config', default=str(TEMPLATE))
    p.add_argument('--stage', choices=('diagnostic', 'screen'), default='diagnostic')
    p.add_argument('--smoke', action='store_true'); a = p.parse_args(); cfg = load(a.experiment_config)
    if a.smoke:
        cfg.update(train_samples=8, val_samples=2, diagnostic_samples=2, diagnostic_updates=8,
                   feature_warmup_updates=2, clean_epochs=1, continue_epochs=1, save_images=2)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1': raise ValueError('Use launcher: physical GPU1 only')
    if not math.isfinite(a.max_hours) or not 0 < a.max_hours <= 24: raise ValueError('Invalid stop budget')
    a.batch_size, a.num_workers, a.no_amp, a.blocks, a.fixed_mode, a.csi, a.seed = 1, 0, True, 1, 0, 'pilot', cfg['seed']
    a.min_free_gb, a.min_free_disk_gb = cfg['min_free_gib'], 12.
    device = preflight(a)
    if device.type != 'cuda': raise ValueError('Actual pretrained experiment requires GPU; toys are separate')
    folder, identity = begin_run(a.output_dir, a, cfg, EXT); started = time.monotonic()
    identity.update(version=VERSION, status='running',
        new_source_sha256={f.name: digest(f) for f in Path(__file__).parent.iterdir() if f.suffix in ('.py', '.json', '.md')})
    atomic_json(folder/'run_identity.json', identity)
    def budget():
        if time.monotonic()-started > a.max_hours*3600: raise TimeoutError('Finite run cap reached; preserve outputs')
        free, _ = torch.cuda.mem_get_info(device)
        if free/2**30 < cfg['reserve_free_gib']: raise MemoryError('GPU free reserve breached; no unrelated process action')
    try:
        old, reference, cc, hashes = load_reference(a, device)
        # Historical VAE image decoder stays frozen, but input-gradient path stays enabled.
        vae = old.base.vae_wrapper.vae.eval().requires_grad_(False)
        scaling = float(old.base.vae_wrapper.scaling_factor)
        if abs(float(vae.config.scaling_factor)-scaling) > 1e-8: raise ValueError('VAE scaling mismatch')
        bart, tokenizer, bart_record = load_bart(a.bart_path, device)
        clip, processor, clip_record = load_clip(a.text_model_path, device)
        model = JointBridge(bart, reference, cfg, clip.config.projection_dim).to(device).float().eval()
        initial = model.trainable_state(); frozen = frozen_signatures(model)
        frozen_other = {n: tensor_signature(v) for n, v in old.state_dict().items()}
        clip_before = {n: tensor_signature(v) for n, v in clip.state_dict().items()}
        identity.update(**hashes, bart_snapshot=bart_record, clip_snapshot=clip_record,
                        initial_bridge_sha256={n: tensor_signature(v) for n, v in initial.items()})
        atomic_json(folder/'run_identity.json', identity)
        atomic_json(folder/'checkpoint_load.json', dict(bart=bart_record, clip=clip_record, exact_historical_reference=True,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            frozen_bart_parameters=sum(p.numel() for p in bart.parameters()),
            frozen_image_reference_parameters=sum(p.numel() for p in reference.parameters()),
            trainable_groups={n: p.numel() for n, p in model.named_parameters() if p.requires_grad},
            decoder_unfrozen=False, clip_unfrozen=False, vae_unfrozen=False))
        train_items = cache(a, cfg, old, reference, bart, tokenizer, clip, processor, 'train',
                            cfg['diagnostic_samples'] if a.stage == 'diagnostic' else cfg['train_samples'], device, budget)
        val = [] if a.stage == 'diagnostic' else cache(a, cfg, old, reference, bart, tokenizer, clip, processor,
                                                   'validation', cfg['val_samples'], device, budget)
        if {r['image_id'] for r in train_items} & {r['image_id'] for r in val}: raise ValueError('Source split overlap')
        if a.smoke: cc = dict(cc, conditions=cc['conditions'][:2], primary_conditions=[c['id'] for c in cc['conditions'][:2]])
        design = dict(version=VERSION, config=cfg, channel_config=cc, stage=a.stage, smoke=a.smoke,
            candidates=list(CANDIDATES), arms=list(ARMS), train_ids=[r['image_id'] for r in train_items],
            validation_ids=[r['image_id'] for r in val],
            source_sha256={str(r['image_id']): r['source_sha256'] for r in train_items+val},
            source_coverage={str(r['image_id']): dict(source_tokens=r['source_tokens'], covered_tokens=r['covered_tokens'], truncated=r['truncated']) for r in train_items+val},
            payload_real_values=4608, payload_complex_symbols=2304, analog_repeats=1, raw_utf8_wire_bytes=0,
            common_slots=32, image_slots=224, text_slots=32, fs_hz=64e9, sps=4, symbol_rate_baud=16e9, rolloff=.1,
            primary='J1 vs J2 same phase, paired all-frame image PSNR and covered WER over declared primary conditions',
            clip_ablation='J1 vs J0 only CLIP source input removed; SAME CLIP TRAIN teacher retained for alignment loss',
            noisy_ablation='Each candidate clean_continue vs FSO from SAME own clean weights with same order/update budget',
            source_bart_frozen=True, decoder_bart_frozen=True, clip_frozen=True, vae_frozen=True,
            image_route='frozen B4 separate image-only code + new trainable image TX/RX and common residual',
            reconstructed_text_to_image_branch='DEFERRED; images use received C/I, not decoded/source text',
            resource_scope='same symbol/peak limits, not identical measured energy; no independent digital comparison',
            receiver_inputs='wire, pilot reliability, pilot SNR, lock; fixed128 memory mask; NO source length/mask/image/text',
            training_proxy='existing full2304-complex five-tap widely-linear stochastic proxy, subject to fresh payload gate',
            evaluation='full optical_realism_v3 waveform; failed frames count with empty text/zero image',
            diagnostic='T32 only clean text memorization; C/I disabled, TRAIN ONLY, not generalization',
            selection='last fixed step; no validation checkpoint selection or automatic enlarging budgets',
            limitations='one trained replicate, fixed weather, reused validation, not physical hardware or final paper evidence',
            environment=environment(cc))
        atomic_json(folder/'design.json', design)
        diag = train_items[:cfg['diagnostic_samples']]
        # Real saved B4 + full pretrained BART + frozen VAE input-gradient gate.
        budget(); item = diag[0]; x = source_wire(model, [item], device)
        native_out = model.receive(*clean(x))
        native_pixel = (vae.decode(native_out['latent']/scaling).sample+1)/2
        native_loss = model.teacher_loss(item['input_ids'].to(device), item['mask'].to(device), native_out['text_memory'])
        native_loss = native_loss+(native_pixel-item['image'].to(device)).square().mean()+native_out['common_alignment'].square().mean()*.001
        native_loss.backward()
        from bart_clip_joint_v1.workflow import required_groups
        native_groups = {g: any(p.grad is not None and torch.isfinite(p.grad).all() and bool(p.grad.abs().sum()>0)
            for n, p in model.named_parameters() if n.startswith(prefixes)) for g, prefixes in required_groups(False).items()}
        if not all(native_groups.values()) or any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise RuntimeError('Actual saved-backbone bridge gradient gate failed: '+str(native_groups))
        atomic_json(folder/'real_backbone_gradient_gate.json', dict(passed=True, groups=native_groups,
            frozen_decoder_input_gradients_verified=True, frozen_vae_input_gradients_verified=True))
        model.zero_grad(set_to_none=True); del x, native_out, native_pixel, native_loss
        # A diagnostic of the complete pretrained encoder-decoder, without lossy compression.
        native = []
        with torch.no_grad():
            for item in diag:
                budget(); result = model.generate(item['bart_hidden'].to(device), tokenizer)
                native.append(make_row(item, 'pretrained_uncompressed_public_memory', 'clean', result['text'],
                    source_split='train', diagnostic_only=True, stopped_by_capacity=result['stopped_by_capacity']))
        with (folder/'pretrained_uncompressed.jsonl').open('x', encoding='utf-8') as f:
            for r in native: f.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n')
        train(model, diag, cfg, device, folder/'text_private_overfit', 'diagnostic', budget, atomic_json, atomic_checkpoint, diagnostic=True)
        groups = memory_controls(model, diag, tokenizer, device, budget, folder/'overfit_controls.jsonl', atomic_json, text_private_only=True)
        gate = quality_gate(groups, cfg); gate['smoke_quality_bypass'] = a.smoke
        atomic_json(folder/'overfit_gate.json', gate)
        def finish(status):
            verify_frozen(model, frozen)
            if frozen_other != {n: tensor_signature(v) for n, v in old.state_dict().items()} or clip_before != {n: tensor_signature(v) for n, v in clip.state_dict().items()}:
                raise RuntimeError('Historical/CLIP frozen weights changed')
            if any(p.grad is not None for obj in (old, clip) for p in obj.parameters()):
                raise RuntimeError('Historical/CLIP frozen gradients leaked')
            atomic_json(folder/'frozen_verification.json', dict(bart_and_reference_unchanged=True, old_unchanged=True,
                clip_unchanged=True, no_test_source_used=True, gradients_verified=True))
            identity.update(status=status, wall_seconds=time.monotonic()-started)
            atomic_json(folder/'run_identity.json', identity); report(folder); seal(folder); verify(folder)
            print('STATUS {} wall={:.1f}s'.format(status, identity['wall_seconds']), flush=True)
        print('TEXT PRIVATE GATE passed={} WER={:.4f}'.format(gate['passed'], gate['matched_wer']), flush=True)
        if not gate['passed'] and not a.smoke: finish('overfit_stop'); return
        if a.stage == 'diagnostic': finish('smoke_complete' if a.smoke else 'diagnostic_ready'); return
        model.load_trainable(initial)
        clean_states = {}
        for candidate in CANDIDATES:
            budget(); model.configure(candidate); model.load_trainable(initial)
            state = train(model, train_items, cfg, device, folder/(candidate+'_clean'), 'clean', budget,
                          atomic_json, atomic_checkpoint, vae=vae, cc=cc)
            groups = memory_controls(model, val, tokenizer, device, budget, folder/(candidate+'_clean_controls.jsonl'), atomic_json)
            gate = quality_gate(groups, cfg, diagnostic=False); gate['smoke_quality_bypass'] = a.smoke
            atomic_json(folder/(candidate+'_clean_gate.json'), gate)
            if not gate['passed'] and not a.smoke: finish('clean_validation_stop'); return
            clean_states[candidate] = state
        bank = read_bank(a.calibration)
        # Exact full-frame physics contract. Bank never comes from text-only384-symbol calibration.
        if bank['config'] != json.loads((Path(a.source_run)/'design.json').read_text())['channel_config']:
            raise ValueError('Bank physics differs from completed B4 source physics')
        for candidate in CANDIDATES:
            for phase in ('clean_continue', 'fso'):
                arm = candidate+'_'+phase; model.configure(candidate); model.load_trainable(clean_states[candidate])
                proof = fidelity(model, train_items, bank, cc, cfg, arm+'_before', device, budget)
                atomic_json(folder/(arm+'_fidelity_before.json'), proof)
                if not proof['passed']: raise RuntimeError('Learned payload proxy gate failed; no noisy training')
                train(model, train_items, cfg, device, folder/arm, phase, budget, atomic_json, atomic_checkpoint,
                      vae=vae, cc=cc, bank=bank if phase == 'fso' else None)
                proof = fidelity(model, train_items, bank, cc, cfg, arm+'_after', device, budget)
                atomic_json(folder/(arm+'_fidelity_after.json'), proof)
                if not proof['passed']: raise RuntimeError('Post-training learned payload proxy gate failed')
        (folder/'images').mkdir(); (folder/'constellations').mkdir()
        with (folder/'samples.jsonl').open('x', encoding='utf-8') as f, (folder/'events.jsonl').open('x') as events:
            count = 0
            for arm in ARMS:
                candidate = next(c for c in CANDIDATES if arm.startswith(c+'_'))
                model.configure(candidate); checkpoint = load_checkpoint(folder/arm/'last_bridge.pth')
                if checkpoint['version'] != VERSION or checkpoint['candidate'] != candidate or checkpoint['config'] != cfg:
                    raise ValueError('Nonexact trained arm')
                model.load_trainable(checkpoint['weights']); del checkpoint
                for condition in [None]+cc['conditions']:
                    cid = 'clean' if condition is None else condition['id']
                    weather = None if condition is None else paths(cc, condition['weather'], len(val), ('bart-validation', condition['weather']))
                    for ix, item in enumerate(val):
                        budget(); ns = seed_for(cfg['seed'], cid, item['image_id'], 'bart-eval'); t = None
                        with torch.no_grad():
                            x = source_wire(model, [item], device)
                            if condition is None:
                                observed = clean(x)
                                stats = dict(receiver_available=True, clean_control_no_physical_transmission=True,
                                    physical_symbol_snr_db=None, pilot_snr_db=None, scheduled_seconds=None,
                                    launch_signal_energy_j=None, pre_fec_ber=None, post_fec_ber=None)
                            else:
                                t = transport(cc, condition, weather[ix], ns); pilot = float(t.probe())
                                yy, reliability, stats = t.continuous(x[0].cpu().numpy(), 1, 'lmmse')
                                available = bool(stats['analog_receiver_available'])
                                observed = (torch.as_tensor(yy[None], device=device, dtype=x.dtype), x.new_tensor([reliability]),
                                            x.new_tensor([pilot]), torch.tensor([available], device=device))
                                stats.update(receiver_available=available, pilot_snr_db=pilot, clean_control_no_physical_transmission=False,
                                             pre_fec_ber=None, post_fec_ber=None)
                            out = model.receive(*observed)
                            ok = bool(observed[3][0]); result = model.generate(out['text_memory'], tokenizer, available=ok)
                            image = ((vae.decode(out['latent']/scaling).sample+1)/2).clamp(0, 1) if ok else torch.zeros_like(item['image'].to(device))
                            im = batch_image_metrics(image, item['image'].to(device))
                        key = '{}__{}__{}'.format(arm, cid, item['image_id'])
                        row = make_row(item, arm, cid, result['text'], stopped_by_capacity=result['stopped_by_capacity'],
                            source_split='validation', noise_seed=ns, frame_key=key, psnr=float(im['psnr'][0]), ssim=float(im['ssim'][0]),
                            stats=stats, payload_sha256=tensor_signature(x), received_sha256=tensor_signature(observed[0]),
                            payload_complex_symbols=2304, analog_repeats=1, raw_utf8_wire_bytes=0)
                        if ix < cfg['save_images']:
                            if arm == ARMS[0] and cid == 'clean': save_image(item['image'][0], folder/'images'/('source_'+str(item['image_id'])+'.png'))
                            name = 'images/'+key+'.png'; save_image(image[0].cpu(), folder/name); row['image_file'] = name
                            if t is not None:
                                name = 'constellations/'+key+'.svg'; constellation_svg(t.constellations, folder/name); row['constellation_file'] = name
                        if t is not None:
                            for event in t.events: events.write(json.dumps(dict(event, frame_key=key), allow_nan=False)+'\n')
                        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); f.flush(); events.flush(); count += 1
                    atomic_json(folder/'progress.json', dict(stage='full_waveform_evaluation', arm=arm, condition=cid,
                        rows=count, expected_rows=len(ARMS)*len(val)*(len(cc['conditions'])+1)))
                    print('EVAL {} {} rows={}'.format(arm, cid, count), flush=True)
        finish('smoke_complete' if a.smoke else 'complete')
    except BaseException as exc:
        atomic_json(folder/'FAILED.json', dict(version=VERSION, type=type(exc).__name__, message=str(exc), wall_seconds=time.monotonic()-started))
        raise


if __name__ == '__main__': main()
