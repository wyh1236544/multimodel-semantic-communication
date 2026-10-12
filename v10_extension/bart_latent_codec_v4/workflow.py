"""Fixed-last-step CE/KD, frozen audits, disk reload and real generation controls."""
import json
import time
import torch
import torch.nn.functional as F
from bart_clip_joint_v1.core import tensor_signature
from bart_latent_codec_v4 import VERSION
from bart_latent_codec_v4.core import row, summarize, batch_indices, digest
from bart_latent_codec_v4.data import batch
from bart_latent_codec_v4.model import prediction, distillation, generate
from bart_latent_codec_v4.channel import proxy_tensor


def signature(module):
    return {n: tensor_signature(p) for n, p in module.named_parameters()}


def frozen_check(module, expected):
    if any(p.requires_grad or p.grad is not None or tensor_signature(p) != expected[n] for n, p in module.named_parameters()):
        raise RuntimeError('Frozen BART changed/received gradients')


def task_loss(model, b, output, c, arm):
    student, labels = prediction(model.bart, b['input_ids'], b['mask'], output['memory'], output['receiver_mask'])
    ce = F.cross_entropy(student.reshape(-1, student.shape[-1]), labels.reshape(-1), ignore_index=-100)
    kd = ce.new_zeros(())
    if arm == 'P1_kd':
        with torch.no_grad(): teacher, _ = prediction(model.bart, b['input_ids'], b['mask'], b['bart_hidden'], b['mask'])
        kd = distillation(student, teacher, labels, c['kd_temperature'])
    return ce+c['kd_weight']*kd, ce, kd


@torch.no_grad()
def controls(model, items, tokenizer, device, budget, path, arm):
    outputs = [model.receive_text(model.transmit(v['bart_hidden'].to(device), v['mask'].to(device))) for v in items]
    entries = []
    with path.open('x', encoding='utf-8') as f:
        for control in ('matched', 'zero', 'shuffled', 'matched_beam4'):
            for i, item in enumerate(items):
                budget(); donor = (i+1)%len(items)
                output = dict(outputs[donor] if control == 'shuffled' else outputs[i])
                if control == 'zero': output['memory'] = torch.zeros_like(output['memory'])
                result = generate(model.bart, output, tokenizer, control == 'matched_beam4')
                r = row(item, arm, control, result, model.width,
                    donor_image_id=items[donor]['image_id'] if control == 'shuffled' else None,
                    memory_sha256=tensor_signature(output['memory']), original_length_provided_to_receiver=False,
                    decode_policy='beam4/no-repeat3/penalty1.2; secondary only' if control == 'matched_beam4' else 'greedy/no repetition restrictions')
                entries.append(r); f.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n'); f.flush()
    return {k: summarize([v for v in entries if v['control'] == k]) for k in ('matched', 'zero', 'shuffled')}


def train(model, items, c, arm, device, folder, tokenizer, budget, atomic_json, atomic_checkpoint,
          updates, frozen_bart, stage='diagnostic', clean_teacher=None):
    folder.mkdir(exist_ok=False); active = {n: p for n, p in model.named_parameters() if p.requires_grad}
    initial = model.state(); observed_gradients = set(); start = time.monotonic()
    base_lr = c['continue_lr'] if stage in ('clean_continue', 'fso_continue') else c['learning_rate']
    optimizer = torch.optim.AdamW(list(active.values()), lr=base_lr, weight_decay=c['weight_decay'])
    b = batch(items, list(range(min(c['batch_size'], len(items)))), device)
    x = model.transmit(b['bart_hidden'], b['mask']); output = model.receive_text(x)
    with torch.no_grad(): xx = model.transmit(b['bart_hidden'], b['mask']); yy = model.receive_text(xx)
    errors = dict(wire=float((x.detach()-xx).abs().max()), memory=float((output['memory'].detach()-yy['memory']).abs().max()))
    atomic_json(folder/'forward_consistency.json', dict(errors=errors, tolerance=c['forward_tolerance']))
    if max(errors.values()) > c['forward_tolerance']: raise RuntimeError('Grad/no-grad interface mismatch')
    del b, x, xx, output, yy
    atomic_json(folder/'trainability.json', dict(active_parameters=sum(p.numel() for p in active.values()),
        frozen_BART_parameters=sum(p.numel() for p in model.bart.parameters()), updates=updates,
        batch_size=c['batch_size'], example_presentations=updates*c['batch_size'], stage=stage,
        teacher_forcing_train_only=True, original_position_MSE=False, decoder_unfrozen=False))
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    with (folder/'history.jsonl').open('x') as hist, (folder/'snapshots.jsonl').open('x', encoding='utf-8') as snapshots:
        for update in range(updates):
            budget(); indices = batch_indices(len(items), c['batch_size'], update, c['seed'])
            b = batch(items, indices, device); optimizer.zero_grad(set_to_none=True)
            warmup = min(c['lr_warmup_updates'], max(1, updates//4))
            fraction = min(1., (update+1)/max(1, warmup))*max(.1, 1-update/updates)
            for group in optimizer.param_groups: group['lr'] = base_lr*fraction
            wire = model.transmit(b['bart_hidden'], b['mask']); clean = model.receive_text(wire)
            clean_loss, ce, kd = task_loss(model, b, clean, c, arm)
            loss = clean_loss; consistency = clean_loss.new_zeros(()); noise_stats = []
            if clean_teacher is not None:
                with torch.no_grad():
                    clean_target = clean_teacher.receive_text(clean_teacher.transmit(b['bart_hidden'], b['mask']))['memory']
                if stage == 'fso_continue':
                    noisy_wire, noise_stats = proxy_tensor(wire, c, update, [items[i]['image_id'] for i in indices])
                    noisy = model.receive_text(noisy_wire)
                    noisy_loss, ce, kd = task_loss(model, b, noisy, c, arm)
                    consistency = F.mse_loss(noisy['memory'], clean_target)
                    loss = (1-c['clean_anchor_weight'])*noisy_loss+c['clean_anchor_weight']*clean_loss
                else:
                    consistency = F.mse_loss(clean['memory'], clean_target)
            loss = loss+c['memory_consistency_weight']*consistency
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
            loss.backward()
            for n, p in active.items():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all(): raise FloatingPointError('Nonfinite gradient '+n)
                    if bool(p.grad.abs().sum()>0): observed_gradients.add(n)
            torch.nn.utils.clip_grad_norm_(list(active.values()), 1., error_if_nonfinite=True)
            optimizer.step()
            elapsed = time.monotonic()-start
            r = dict(update=update+1, total_updates=updates, batch_image_ids=[items[i]['image_id'] for i in indices],
                ce=float(ce.detach()), kd=float(kd.detach()), memory_consistency=float(consistency.detach()),
                loss=float(loss.detach()), lr=optimizer.param_groups[0]['lr'], elapsed_seconds=elapsed,
                estimated_remaining_train_seconds=elapsed/(update+1)*(updates-update-1), proxy=noise_stats,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30 if device.type=='cuda' else None)
            hist.write(json.dumps(r, allow_nan=False)+'\n'); hist.flush()
            if (update+1)%c['snapshot_every'] == 0 or update+1 == updates:
                with torch.no_grad():
                    for sample in items[:2]:
                        budget(); rx = model.receive_text(model.transmit(sample['bart_hidden'].to(device), sample['mask'].to(device)))
                        result = generate(model.bart, rx, tokenizer)
                        snapshots.write(json.dumps(row(sample, arm, 'snapshot', result, model.width, update=update+1), ensure_ascii=False)+'\n'); snapshots.flush()
            if update == 0 or (update+1)%20 == 0:
                atomic_json(folder.parents[1]/'progress.json', dict(arm=arm, stage=stage, **r))
                print('CODEC {} {} {}/{} CE={:.4f} KD={:.4f} elapsed={:.1f}s ETAtrain={:.1f}s'.format(
                    arm, stage, update+1, updates, r['ce'], r['kd'], elapsed, r['estimated_remaining_train_seconds']), flush=True)
    audit = {}
    for key, prefixes in model.groups().items():
        names = [n for n in active if n.startswith(prefixes)]
        audit[key] = dict(tensors=len(names), nonzero_gradient=any(n in observed_gradients for n in names),
            changed=any(not torch.equal(initial[n], active[n].detach().cpu()) for n in names))
    frozen_check(model.bart, frozen_bart)
    passed = all(v['tensors'] and v['nonzero_gradient'] and v['changed'] for v in audit.values())
    atomic_json(folder/'learning_audit.json', dict(passed=bool(passed), groups=audit, frozen_unchanged=True,
        observed_gradient_parameters=sum(active[n].numel() for n in observed_gradients)))
    saved = model.state()
    atomic_checkpoint(folder/'last.pth', dict(version=VERSION, arm=arm, width=model.width, config=c,
        stage=stage, updates=updates, state=saved, precise_resume_supported=False))
    loaded = torch.load(str(folder/'last.pth'), map_location='cpu')
    if loaded['version'] != VERSION or loaded['config'] != c or loaded['width'] != model.width: raise ValueError('Saved identity differs')
    model.restore(loaded['state'])
    restored = model.state()
    if any(not torch.equal(v, restored[n]) for n, v in saved.items()): raise RuntimeError('Actual disk reload mismatch')
    atomic_json(folder/'reload_audit.json', dict(passed=True, exact_disk_reload=True, tensors=len(saved)))
    if not passed: raise RuntimeError('Required codec groups did not learn; preserve audit')
    atomic_json(folder/'TRAIN_COMPLETE.json', dict(updates=updates, wall_seconds=time.monotonic()-start,
        checkpoint_sha256=digest(folder/'last.pth'), quality_not_implied=True))
    return saved
