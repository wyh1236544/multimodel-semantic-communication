"""Offline GPU1 decodable text-latent screen, optional own-gated expansion."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
EXT = Path(__file__).resolve().parents[1]; PROJECT = EXT.parent; sys.path.insert(0, str(EXT))
from bart_latent_codec_v4 import VERSION, ARMS
from bart_latent_codec_v4.core import load, TEMPLATE, digest, row, summarize, gate, geometry


def evaluate(model, items, a, c, folder, tokenizer, device, budget, arm, stage, atomic_json):
    import torch
    from bart_latent_codec_v4.model import generate
    from bart_latent_codec_v4.channel import proxy_numpy, resource, send
    from fso_fluctuating_v2.core import paths, seed_for
    from optical_realism_v3.constellation import constellation_svg
    folder.mkdir(exist_ok=False); (folder/'constellations').mkdir()
    source = Path(a.channel_source_run)/'design.json'; source_sha = digest(source)
    parent = json.loads(source.read_text())
    if parent.get('smoke') is not False or not (source.parent/'COMPLETE.json').is_file():
        raise ValueError('Completed non-smoke optical source required')
    cc = parent['channel_config']; resources = resource(cc, model.width)
    conditions = [('awgn', s) for s in c['proxy_snr_db']]+[(p, s) for p in sorted(c['proxy_profiles']) for s in c['proxy_snr_db']]
    atomic_json(folder/'eval_design.json', dict(validation_ids=[v['image_id'] for v in items],
        proxy_conditions=['{}@{}'.format(p, s) for p, s in conditions],
        waveform_conditions=[v['id'] for v in cc['conditions']], waveform_config=cc,
        channel_source=str(source), channel_source_sha256=source_sha, resources=resources,
        training_proxy_not_waveform_equivalent=True, one_channel_draw_not_retransmissions=True))
    with torch.no_grad(), (folder/'proxy.jsonl').open('x', encoding='utf-8') as writer:
        for profile, snr in conditions:
            for item in items:
                budget(); x = model.transmit(item['bart_hidden'].to(device), item['mask'].to(device))
                y, _, _, stats = proxy_numpy(x[0].cpu().numpy(), c, profile, snr,
                    seed_for(c['seed'], 'v4-eval-field', item['image_id'], profile, snr),
                    seed_for(c['seed'], 'v4-eval-noise', item['image_id'], profile, snr))
                output = model.receive_text(torch.as_tensor(y[None], device=device))
                r = row(item, arm, '{}@{}'.format(profile, snr), generate(model.bart, output, tokenizer), model.width, stage=stage, stats=stats)
                writer.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n'); writer.flush()
    with torch.no_grad(), (folder/'waveform.jsonl').open('x', encoding='utf-8') as writer, (folder/'events.jsonl').open('x') as events:
        for condition in cc['conditions']:
            trace = paths(cc, condition['weather'], len(items), ('v4-waveform-eval', condition['weather']))
            for i, item in enumerate(items):
                budget(); x = model.transmit(item['bart_hidden'].to(device), item['mask'].to(device))
                seed = seed_for(c['seed'], 'v4-waveform-noise', item['image_id'], condition['id'])
                y, stats, transport = send(cc, condition, trace[i], seed, model.width, x[0].cpu().numpy())
                output = model.receive_text(torch.as_tensor(y[None], device=device), torch.tensor([stats['receiver_available']], device=device))
                r = row(item, arm, condition['id'], generate(model.bart, output, tokenizer), model.width,
                    stage=stage, stats=stats, actual_payload_single_shot=True)
                if i < 4:
                    name = 'constellations/{}__{}.svg'.format(condition['id'], item['image_id'])
                    constellation_svg(transport.constellations, folder/name); r['constellation_file'] = name
                writer.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n'); writer.flush()
                for event in transport.events:
                    events.write(json.dumps(dict(event, image_id=item['image_id'], condition=condition['id']), allow_nan=False)+'\n')
                events.flush()
            print('WAVEFORM {} {} {} finished'.format(arm, stage, condition['id']), flush=True)
    if digest(source) != source_sha: raise RuntimeError('Parent physical configuration changed')


def main():
    import torch
    from v10.paths import write_path
    from v10.io import atomic_json, atomic_checkpoint
    from bart_clip_joint_v1.backbones import load_bart
    from bart_clip_joint_v1.core import tensor_signature
    from text_recovery_v3.model import consistency
    from bart_latent_codec_v4.model import LatentCodec, generate
    from bart_latent_codec_v4.data import cache
    from bart_latent_codec_v4.workflow import signature, frozen_check, train, controls
    from bart_latent_codec_v4.report import finish
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ('output-dir', 'data-root', 'split-manifest', 'bart-path'): p.add_argument('--'+key, required=True)
    p.add_argument('--channel-source-run'); p.add_argument('--experiment-config', default=str(TEMPLATE))
    p.add_argument('--width', type=int, choices=(16, 32, 64), default=64)
    p.add_argument('--max-hours', type=float, default=2.)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--expand', action='store_true')
    a = p.parse_args(); c = load(a.experiment_config)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1': raise ValueError('Use launcher on physical GPU1')
    if not math.isfinite(a.max_hours) or not 0 < a.max_hours <= 24: raise ValueError('Finite budget required')
    if a.expand and not a.channel_source_run: raise ValueError('Expansion requires optical config source')
    root = write_path(a.output_dir, PROJECT)
    if root.exists(): raise FileExistsError('Use NEW output; preserve previous evidence')
    if root.relative_to(PROJECT).parts[0] != 'experiment_results' or not root.name.startswith('v10_'):
        raise ValueError('Project-local experiment_results/v10_* only')
    ancestor = root.parent
    while not ancestor.exists(): ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < 8*2**30: raise RuntimeError('8GiB disk reserve required')
    if not torch.cuda.is_available(): raise RuntimeError('Actual pretrained GPU required')
    device = torch.device('cuda:0'); free, total = torch.cuda.mem_get_info(device)
    if free/2**30 < c['min_free_gib']: raise MemoryError('Insufficient GPU reserve; no other-process action')
    if a.smoke: c.update(diagnostic_samples=2, diagnostic_updates=8, batch_size=2, train_samples=8,
        val_samples=2, clean_updates=8, continue_updates=8)
    root.mkdir(parents=True, exist_ok=False); start = time.monotonic()
    source_files = [f for f in Path(__file__).parent.iterdir() if f.suffix in ('.py', '.md', '.json')]
    (root/'source_snapshot').mkdir()
    for f in source_files:
        with (root/'source_snapshot'/f.name).open('xb') as writer: writer.write(f.read_bytes())
    identity = dict(version=VERSION, status='running', python=sys.executable, python_version=sys.version,
        torch=str(torch.__version__), cuda=torch.version.cuda, transformers=importlib.metadata.version('transformers'),
        pid=os.getpid(), argv=sys.argv, config=c, width=a.width, smoke=a.smoke, expand=a.expand,
        split_sha256=digest(a.split_manifest), source_sha256={f.name: digest(f) for f in source_files},
        startup_free_gib=free/2**30, gpu_total_gib=total/2**30)
    atomic_json(root/'run_identity.json', identity)
    def budget():
        if time.monotonic()-start > a.max_hours*3600: raise TimeoutError('Finite run budget reached')
        left, _ = torch.cuda.mem_get_info(device)
        if left/2**30 < c['reserve_free_gib']: raise MemoryError('GPU reserve breached; preserve evidence')
    try:
        budget(); bart, tokenizer, br = load_bart(a.bart_path, device)
        original = signature(bart); atomic_json(root/'checkpoint_load.json', br)
        diagnostic, receipt = cache(a, c, bart, tokenizer, 'train', c['diagnostic_samples'], device, budget)
        check = consistency(bart, diagnostic[0], device)
        if max(check['incremental_max_logit_error'], check['cached_source_max_logit_error']) > c['forward_tolerance']:
            raise RuntimeError('Actual BART cache/interface mismatch')
        check['passed'] = True; atomic_json(root/'bart_interface_audit.json', check)
        native_rows = []
        for item in diagnostic:
            budget(); output = dict(memory=item['bart_hidden'].to(device), receiver_mask=item['mask'].to(device), available=torch.ones(1, device=device, dtype=torch.bool))
            native_rows.append(row(item, 'native', 'matched', generate(bart, output, tokenizer), a.width,
                source_memory_oracle_train_only=True, not_a_transmission_baseline=True))
        with (root/'native.jsonl').open('x', encoding='utf-8') as writer:
            for r in native_rows: writer.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n')
        ns = summarize(native_rows); native_ok = ns['covered_wer'] <= c['overfit_wer_gate'] and ns['capacity_stop_fraction'] == 0
        atomic_json(root/'native_gate.json', dict(passed=bool(native_ok), groups=ns, smoke_quality_bypass=a.smoke))
        design = dict(version=VERSION, config=c, width=a.width, arms=list(ARMS), diagnostic_view=receipt,
            text_only=True, C_I_T_historical_architecture_preserved=True, CLIP_VAE_images_loaded=False,
            decoder_frozen=True, protocol=geometry(a.width), fixed_last_checkpoint=True,
            comparable='P1-P0 same initial codec, data, updates, payload; KD adds teacher compute',
            no_original_position_MSE=True, no_source_length_or_feature_receiver_side_channel=True,
            surrogate_not_waveform_equivalent=True, no_hardware_or_joint_gain_claim=True)
        atomic_json(root/'design.json', design); statuses = {}
        if not native_ok and not a.smoke:
            statuses = {arm: dict(status='native_stop', diagnostic_passed=False, expanded=False) for arm in ARMS}
        else:
            train_items = val = None
            for arm in ARMS:
                budget(); torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
                model = LatentCodec(bart, c, a.width).to(device).float().eval(); initial = model.state()
                if arm == ARMS[0]: initial_hashes = {n: tensor_signature(v) for n, v in initial.items()}
                elif initial_hashes != {n: tensor_signature(v) for n, v in initial.items()}:
                    raise RuntimeError('Matched arms initial tensors differ')
                folder = root/arm; folder.mkdir()
                atomic_json(folder/'initial_state_hashes.json', initial_hashes)
                train(model, diagnostic, c, arm, device, folder/'diagnostic', tokenizer, budget,
                    atomic_json, atomic_checkpoint, c['diagnostic_updates'], original)
                dg = gate(controls(model, diagnostic, tokenizer, device, budget, folder/'diagnostic_controls.jsonl', arm), c)
                atomic_json(folder/'diagnostic_gate.json', dg)
                statuses[arm] = dict(status='diagnostic_passed' if dg['passed'] else 'overfit_stop', diagnostic_passed=dg['passed'], expanded=False)
                if a.expand and dg['passed'] and not a.smoke:
                    model.restore(initial)  # Never initialize production from TRAIN8-fitted weights.
                    if train_items is None:
                        train_items, tr = cache(a, c, bart, tokenizer, 'train', c['train_samples'], device, budget)
                        val, vr = cache(a, c, bart, tokenizer, 'validation', c['val_samples'], device, budget)
                        if set(tr['image_ids']) & set(vr['image_ids']): raise ValueError('Train/validation overlap')
                        design.update(main_train_view=tr, validation_view=vr); atomic_json(root/'design.json', design)
                    clean_state = train(model, train_items, c, arm, device, folder/'clean', tokenizer, budget,
                        atomic_json, atomic_checkpoint, c['clean_updates'], original, stage='clean')
                    vg = gate(controls(model, val, tokenizer, device, budget, folder/'validation_controls.jsonl', arm), c, False)
                    atomic_json(folder/'validation_gate.json', vg)
                    statuses[arm].update(status='clean_validation_stop', expanded=True, validation_passed=vg['passed'])
                    if vg['passed']:
                        evaluate(model, val, a, c, folder/'eval_clean', tokenizer, device, budget, arm, 'clean', atomic_json)
                        teacher = LatentCodec(bart, c, a.width).to(device).float().eval()
                        teacher.restore(clean_state); teacher.requires_grad_(False)
                        for stage in ('clean_continue', 'fso_continue'):
                            model.restore(clean_state)
                            train(model, train_items, c, arm, device, folder/stage, tokenizer, budget,
                                atomic_json, atomic_checkpoint, c['continue_updates'], original, stage=stage, clean_teacher=teacher)
                            controls(model, val, tokenizer, device, budget, folder/(stage+'_controls.jsonl'), arm)
                            evaluate(model, val, a, c, folder/('eval_'+stage), tokenizer, device, budget, arm, stage, atomic_json)
                        del teacher; statuses[arm]['status'] = 'waveform_evaluated'
                atomic_json(root/'arm_status.json', statuses)
                del initial, model; torch.cuda.empty_cache()
        atomic_json(root/'arm_status.json', statuses); frozen_check(bart, original)
        if digest(a.split_manifest) != identity['split_sha256']: raise RuntimeError('Original split changed')
        for view in [receipt]+[design[k] for k in ('main_train_view', 'validation_view') if k in design]:
            if digest(view['annotation_path']) != view['annotation_sha256']: raise RuntimeError('Original captions changed')
        if any(digest(f) != identity['source_sha256'][f.name] for f in source_files): raise RuntimeError('Runtime source changed')
        atomic_json(root/'frozen_verification.json', dict(passed=True, BART_unchanged=True,
            original_data_split_unchanged=True, old_C_I_T_weights_results_untouched=True))
        identity.update(status='smoke_complete' if a.smoke else 'screen_complete', wall_seconds=time.monotonic()-start)
        atomic_json(root/'run_identity.json', identity); finish(root)
        print('Screen finished; quality failures retained. Send small records; weights stay server.', flush=True)
    except BaseException as exc:
        atomic_json(root/'FAILED.json', dict(type=type(exc).__name__, message=str(exc), wall_seconds=time.monotonic()-start)); raise


if __name__ == '__main__': main()
