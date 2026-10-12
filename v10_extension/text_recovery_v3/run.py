"""Offline GPU1 text repair screen; no image/full-backbone retraining."""
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
from text_recovery_v3 import VERSION, ARMS, OPTIONAL_ARMS
from text_recovery_v3.core import load, TEMPLATE, digest, tensor_signature, summarize, row, gate, geometry, spec


def main():
    import torch
    from v10.paths import write_path
    from v10.io import atomic_json, atomic_checkpoint
    from bart_clip_joint_v1.backbones import load_bart, load_clip
    from text_recovery_v3.model import TextBridge, install_lora, adapters, lora_state, reset_lora, generate, consistency
    from text_recovery_v3.data import cache, transmit
    from text_recovery_v3.workflow import train, controls
    from text_recovery_v3.report import seal, verify, render
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ('output-dir', 'data-root', 'split-manifest', 'bart-path', 'text-model-path', 'channel-source-run'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--experiment-config', default=str(TEMPLATE)); p.add_argument('--max-hours', type=float, default=2.)
    p.add_argument('--expand', action='store_true'); p.add_argument('--smoke', action='store_true')
    p.add_argument('--include-common', action='store_true'); a = p.parse_args(); c = load(a.experiment_config)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1': raise ValueError('Use launcher on physical GPU1')
    if not math.isfinite(a.max_hours) or not 0 < a.max_hours <= 24: raise ValueError('Finite cap required')
    root = write_path(a.output_dir, PROJECT)
    if root.exists(): raise FileExistsError('Existing result preserved')
    ancestor = root.parent
    while not ancestor.exists(): ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free/2**30 < 4: raise RuntimeError('Insufficient disk reserve')
    if not torch.cuda.is_available(): raise RuntimeError('Actual pretrained GPU required')
    device = torch.device('cuda:0'); free, total = torch.cuda.mem_get_info(device)
    if free/2**30 < c['min_free_gib']: raise MemoryError('GPU startup reserve; no other-process action')
    torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
    if a.smoke: c.update(diagnostic_samples=2, diagnostic_steps=8, native_steps=8, warmup_steps=2,
        ramp_steps=2, train_samples=8, val_samples=2, clean_epochs=1)
    root.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    arms = list(ARMS)+(list(OPTIONAL_ARMS) if a.include_common else [])
    identity = dict(version=VERSION, status='running', python=sys.executable, python_version=sys.version,
        torch=str(torch.__version__), cuda=torch.version.cuda, pid=os.getpid(), argv=sys.argv,
        transformers=importlib.metadata.version('transformers'), split_sha256=digest(a.split_manifest),
        source_sha256={f.name:digest(f) for f in Path(__file__).parent.iterdir() if f.suffix in ('.py', '.json', '.md')},
        config=c, smoke=a.smoke, expand=a.expand, arms=arms, startup_free_gib=free/2**30, gpu_total_gib=total/2**30)
    atomic_json(root/'run_identity.json', identity)
    def budget():
        if time.monotonic()-started > a.max_hours*3600: raise TimeoutError('Finite total run budget')
        left, _ = torch.cuda.mem_get_info(device)
        if left/2**30 < c['reserve_free_gib']: raise MemoryError('GPU reserve breached, preserve output')
    try:
        budget(); bart, tokenizer, br = load_bart(a.bart_path, device); clip, processor, cr = load_clip(a.text_model_path, device)
        modules = install_lora(bart, c); lora_initial = lora_state(bart)
        original_bart = {n:tensor_signature(v) for n,v in bart.named_parameters() if not n.endswith(('.a', '.b'))}
        original_clip = {n:tensor_signature(v) for n,v in clip.named_parameters()}
        atomic_json(root/'checkpoint_load.json', dict(bart=br, clip=cr,
            lora_parameters=sum(m.a.numel()+m.b.numel() for m in modules), VAE_image_reference_loaded=False,
            source_encoder_frozen=True, original_pretrained_weights_frozen=True))
        items, tr = cache(a, c, bart, tokenizer, clip, processor, 'train', c['diagnostic_samples'], device, budget)
        native = []
        # Original mask here is explicitly diagnostic oracle, never a deployed receive input.
        for control in ('native_masked', 'native_all128', 'native_pad_zero'):
            for item in items:
                budget(); memory = item['bart_hidden'].to(device); mask = item['mask'].to(device)
                if control == 'native_pad_zero': memory = memory*mask.unsqueeze(-1)
                if control == 'native_all128': mask = torch.ones_like(mask)
                result = generate(bart, memory, mask, tokenizer)
                native.append(row(item, 'native', control, result, source_feature_oracle_diagnostic_only=True,
                    not_a_transmission_baseline=True))
        with (root/'native_controls.jsonl').open('x', encoding='utf-8') as f:
            for v in native: f.write(json.dumps(v, ensure_ascii=False, allow_nan=False)+'\n')
        check = consistency(bart, items[0], device); check['tolerance'] = c['forward_tolerance']
        check['passed'] = max(check['incremental_max_logit_error'], check['cached_source_max_logit_error']) <= c['forward_tolerance']
        atomic_json(root/'bart_interface_audit.json', check)
        if not check['passed']: raise RuntimeError('Native teacher/incremental/source-cache inconsistency; no compression training')
        summaries = {k:summarize([v for v in native if v['control']==k]) for k in ('native_masked', 'native_all128', 'native_pad_zero')}
        good = summaries['native_masked']['covered_wer'] <= c['overfit_wer_gate'] and summaries['native_masked']['capacity_stop_fraction']==0
        if not good or a.smoke:
            adapters(bart, True, True); bart.experiment_arm = 'A3_lora'; bart.s = spec('A3_lora')
            train(bart, items, c, device, root/'native_lora_overfit', tokenizer, budget, atomic_json, atomic_checkpoint, native=True)
            fitted = []
            for item in items:
                budget(); result = generate(bart, item['bart_hidden'].to(device), item['mask'].to(device), tokenizer)
                fitted.append(row(item, 'native_lora', 'matched', result, source_feature_oracle_diagnostic_only=True))
            with (root/'native_lora_controls.jsonl').open('x', encoding='utf-8') as f:
                for v in fitted: f.write(json.dumps(v, ensure_ascii=False, allow_nan=False)+'\n')
            summaries['native_lora'] = summarize(fitted)
            good = summaries['native_lora']['covered_wer'] <= c['overfit_wer_gate'] and summaries['native_lora']['capacity_stop_fraction']==0
            reset_lora(bart, lora_initial)  # Fitted TRAIN diagnostic NEVER initializes production candidates.
        atomic_json(root/'native_gate.json', dict(passed=bool(good), groups=summaries, quality_limit=c['overfit_wer_gate'],
            smoke_quality_bypass=a.smoke, normalized_exact_target_note='8/8 is desired, WER gate unchanged; raw casing separate'))
        design = dict(version=VERSION, config=c, diagnostic_caption_view=tr, arms=arms, expand=a.expand,
            geometry={arm:geometry(arm) for arm in arms}, text_only=True, image_training_executed=False,
            source_mask_rx_policy='A0 all128; other arms decode a charged 16-real length header from observed same-channel symbols',
            native_oracle_scope='TRAIN diagnosis only; not a communication baseline',
            losses='A0/A1 legacy; A2+ stronger memory supervision, CE ramp and lower bridge LR; A3+ decoder LoRA',
            comparisons='A1-A0 interface+charged-header; A2-A1 optimization bundle; A3-A2 same resource LoRA; A4 more symbols capacity diagnostic; A5 more C symbols not joint evidence',
            selection='fixed-last-step, all diagnostics reported; only predeclared A2/A3 may expand if their own gate passes',
            noisy_training_executed=False, waveform_eval='only after own clean validation gate; no proxy/sliced old bank',
            limitations='one training replicate; small reused validation; no full image-text joint gain/hardware/publication claim')
        atomic_json(root/'design.json', design); statuses = {}
        if good or a.smoke:
            for arm in arms:
                budget(); reset_lora(bart, lora_initial); torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
                model = TextBridge(bart, c, arm, clip.config.projection_dim).to(device).float().eval()
                initial = model.state(); folder = root/arm; folder.mkdir()
                train(model, items, c, device, folder/'diagnostic', tokenizer, budget, atomic_json, atomic_checkpoint)
                groups = controls(model, items, tokenizer, device, budget, folder/'diagnostic_controls.jsonl')
                decision = gate(groups, c); decision['groups'] = groups; decision['smoke_quality_bypass'] = a.smoke
                atomic_json(folder/'diagnostic_gate.json', decision)
                statuses[arm] = dict(status='diagnostic_passed' if decision['passed'] else 'overfit_stop', diagnostic_passed=decision['passed'], expanded=False)
                if a.expand and arm in ('A2_ramp', 'A3_lora') and (decision['passed'] or a.smoke):
                    # Reset all TRAIN-fitted parameters before independent main training.
                    model.restore(initial)
                    train_items, main_tr = cache(a, c, bart, tokenizer, clip, processor, 'train', c['train_samples'], device, budget)
                    val, vr = cache(a, c, bart, tokenizer, clip, processor, 'validation', c['val_samples'], device, budget)
                    if set(main_tr['image_ids']) & set(vr['image_ids']): raise ValueError('Split overlap')
                    design.setdefault('main_caption_views', {})[arm] = dict(train=main_tr, validation=vr)
                    atomic_json(root/'design.json', design)
                    train(model, train_items, c, device, folder/'clean', tokenizer, budget, atomic_json, atomic_checkpoint, diagnostic=False)
                    vg = controls(model, val, tokenizer, device, budget, folder/'validation_controls.jsonl')
                    vd = gate(vg, c, diagnostic=False); vd['groups'] = vg; vd['smoke_quality_bypass'] = a.smoke
                    atomic_json(folder/'validation_gate.json', vd)
                    statuses[arm].update(expanded=True, validation_passed=vd['passed'], status='clean_validation_stop')
                    if vd['passed'] or a.smoke:
                        evaluate(model, val, a, c, root, folder, tokenizer, device, budget, atomic_json)
                        statuses[arm]['status'] = 'waveform_evaluated'
                    del train_items, val
                atomic_json(root/'arm_status.json', statuses); del model
                adapters(bart); torch.cuda.empty_cache()
        else:
            statuses = {arm:dict(status='native_stop', expanded=False, diagnostic_passed=False) for arm in arms}
            atomic_json(root/'arm_status.json', statuses)
        adapters(bart)
        if any(p.grad is not None or tensor_signature(p)!=original_bart[n] for n,p in bart.named_parameters() if n in original_bart):
            raise RuntimeError('Original BART parameter changed')
        if any(p.grad is not None or tensor_signature(p)!=original_clip[n] for n,p in clip.named_parameters()): raise RuntimeError('CLIP changed')
        if digest(a.split_manifest) != identity['split_sha256']: raise RuntimeError('Original split changed')
        receipts = [tr]+[view for arm in design.get('main_caption_views', {}).values() for view in arm.values()]
        if any(digest(v['annotation_path']) != v['annotation_sha256'] for v in receipts): raise RuntimeError('Official captions changed')
        atomic_json(root/'frozen_verification.json', dict(passed=True, BART_original_CLIP_unchanged=True, old_images_VAE_weights_untouched=True))
        identity.update(status='smoke_complete' if a.smoke else 'screen_complete', wall_seconds=time.monotonic()-started)
        atomic_json(root/'run_identity.json', identity)
        render(root); seal(root); verify(root)
        print('Text screen finished. Negative gates retained. Send records, not large weights.', flush=True)
    except BaseException as exc:
        atomic_json(root/'FAILED.json', dict(version=VERSION, type=type(exc).__name__, message=str(exc), wall_seconds=time.monotonic()-started)); raise


def evaluate(model, items, args, c, root, folder, tokenizer, device, budget, atomic_json):
    import torch
    from text_recovery_v3.channel import send, resource
    from fso_fluctuating_v2.core import paths, seed_for
    from optical_realism_v3.constellation import constellation_svg
    from text_recovery_v3.model import generate
    from text_recovery_v3.data import transmit
    parent = Path(args.channel_source_run)
    if not (parent/'COMPLETE.json').is_file(): raise ValueError('Completed non-smoke optical configuration source required')
    source = parent/'design.json'; parent_design = json.loads(source.read_text())
    if parent_design.get('smoke') is not False: raise ValueError('Non-smoke source required')
    cc = parent_design['channel_config']; cc_hash = digest(source)
    atomic_json(folder/'waveform_design.json', dict(channel_source=str(source), source_sha256=cc_hash,
        config=cc, resources=resource(cc, model.experiment_arm), validation_ids=[v['image_id'] for v in items],
        no_noisy_retraining=True, single_shot=True))
    (folder/'constellations').mkdir()
    with torch.no_grad(), (folder/'waveform.jsonl').open('x', encoding='utf-8') as writer, (folder/'events.jsonl').open('x') as events:
        for condition in cc['conditions']:
            trajectory = paths(cc, condition['weather'], len(items), ('text-recovery-v3-validation', condition['weather']))
            for i, item in enumerate(items):
                budget(); x = transmit(model, item, device)
                seed = seed_for(c['seed'], item['image_id'], condition['id'], 'text-recovery-v3-eval')
                received, stats, t = send(cc, condition, trajectory[i], seed, model.experiment_arm, x[0].cpu().numpy())
                y = torch.as_tensor(received[None], device=device)
                out = model.receive_text(y, torch.tensor([stats['receiver_available']], device=device),
                    y.new_tensor([stats['pilot_snr_db']]), y.new_tensor([stats['analog_reliability']]))
                result = generate(model.bart, out['memory'], out['receiver_mask'], tokenizer, bool(out['available'][0]))
                entry = row(item, model.experiment_arm, condition['id'], result, received_length=out['received_length'][0],
                    received_length_correct=out['received_length'][0]==item['covered_tokens'],
                    header_correctness_evaluation_only=True, stats=stats, payload_sha256=tensor_signature(x))
                if i < 4:
                    name = 'constellations/{}__{}.svg'.format(condition['id'], item['image_id'])
                    constellation_svg(t.constellations, folder/name); entry['constellation_file'] = name
                writer.write(json.dumps(entry, ensure_ascii=False, allow_nan=False)+'\n'); writer.flush()
                for event in t.events: events.write(json.dumps(dict(event, image_id=item['image_id'], condition=condition['id']), allow_nan=False)+'\n')
                events.flush()
            atomic_json(root/'progress.json', dict(stage='full_waveform', arm=model.experiment_arm, condition=condition['id']))
    if digest(source) != cc_hash: raise RuntimeError('Parent optical source changed')


if __name__ == '__main__': main()
