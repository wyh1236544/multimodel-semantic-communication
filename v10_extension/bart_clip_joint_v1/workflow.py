"""Actual greedy recovery controls, gradient audit, staged bounded training."""
import json
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from bart_clip_joint_v1 import VERSION
from bart_clip_joint_v1.core import make_row, summarize, tensor_signature, seed_for
from bart_clip_joint_v1.data import source_wire
from bart_clip_joint_v1.channel import clean, noisy, papr


def frozen_signatures(model):
    result = {n: tensor_signature(p) for n, p in model.named_parameters() if not p.requires_grad}
    result.update({'BUFFER:'+n: tensor_signature(v) for n, v in model.named_buffers()})
    return result


def verify_frozen(model, before):
    after = frozen_signatures(model)
    if before != after or any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise RuntimeError('Frozen pretrained/reference parameters changed or acquired gradients')


@torch.no_grad()
def memory_controls(model, items, tokenizer, device, budget, path, atomic_json, text_private_only=False):
    memories = []
    for item in items:
        budget(); x = source_wire(model, [item], device)
        if text_private_only: x = torch.cat((torch.zeros_like(x[:, :256]), x[:, 256:]), 1)
        memories.append(model.receive(*clean(x), text_only=True)['text_memory'])
    if len(items) < 2: raise ValueError('Shuffle diagnostic needs two distinct texts')
    rows = []
    with path.open('x', encoding='utf-8') as f:
        for k in ('matched', 'zero', 'shuffled'):
            for ix, item in enumerate(items):
                budget(); memory = memories[ix] if k == 'matched' else torch.zeros_like(memories[ix]) if k == 'zero' else memories[(ix+1) % len(items)]
                result = model.generate(memory, tokenizer)
                r = make_row(item, k, 'clean', result['text'], stopped_by_capacity=result['stopped_by_capacity'],
                    source_split=item['split'], memory_sha256=tensor_signature(memory))
                rows.append(r); f.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n'); f.flush()
    groups = {k: summarize([r for r in rows if r['arm'] == k]) for k in ('matched', 'zero', 'shuffled')}
    atomic_json(path.with_suffix('.summary.json'), groups)
    return groups


def required_groups(diagnostic):
    groups = dict(text_compression=('text_fold.', 'text_source.'), text_tx=('text_tx.',),
                  text_rx=('text_rx.', 'text_receiver.', 'text_expand.'),
                  pretrained_memory_adapter=('bart_memory_norm.', 'bart_memory_adapter.'))
    if not diagnostic:
        groups.update(common_source=('common_attention.', 'common_queries', 'common_tx.'),
            common_receiver=('common_rx.',), image_tx=('image_tx.',), image_rx=('image_rx.', 'image_residual.'),
            cross_modal_text_receiver=('text_cross.', 'text_gate.'))
    return groups


def train(model, items, cfg, device, directory, label, budget, atomic_json, atomic_checkpoint,
          vae=None, cc=None, bank=None, diagnostic=False):
    directory.mkdir(exist_ok=False); model.eval()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                           lr=cfg['learning_rate'] if diagnostic or label == 'clean' else cfg['fine_learning_rate'],
                           weight_decay=cfg['weight_decay'])
    before = model.trainable_state(); frozen = frozen_signatures(model)
    gradients = set(); start = time.monotonic(); step = updates = usable = 0
    epochs = math.ceil(cfg['diagnostic_updates']/len(items)) if diagnostic else cfg['clean_epochs'] if label == 'clean' else cfg['continue_epochs']
    total = cfg['diagnostic_updates'] if diagnostic else epochs*len(items)
    groups = required_groups(diagnostic); torch.cuda.reset_peak_memory_stats(device) if device.type == 'cuda' else None
    opt.zero_grad(set_to_none=True)
    with (directory/'history.jsonl').open('x', encoding='utf-8') as log:
        for epoch in range(epochs):
            order = np.random.default_rng(seed_for(cfg['seed'], epoch, 'bart-order')).permutation(len(items))
            for pos, ix in enumerate(order):
                if step >= total: break
                budget(); item = items[int(ix)]; x = source_wire(model, [item], device)
                if diagnostic: x = torch.cat((torch.zeros_like(x[:, :256]), x[:, 256:]), 1)
                observed = noisy(x, bank, cc, epoch, pos) if bank is not None else clean(x)
                y, rel, snr, ok = observed; valid = bool(ok[0]); usable += int(valid)
                out = model.receive(y, rel, snr, ok, text_only=diagnostic)
                ids, mask = item['input_ids'].to(device), item['mask'].to(device)
                memory = out['text_memory']; target = item['bart_hidden'].to(device)
                feature = ((memory-target).square().mean(-1)*mask).sum()/mask.sum().clamp_min(1)
                warmup = diagnostic and step < cfg['feature_warmup_updates']
                ce = feature.new_zeros(()) if warmup or not valid else model.teacher_loss(ids, mask, memory)
                # Decoder gradients flow to bridge despite all decoder weights frozen.
                loss = (cfg['feature_weight']*feature+ce)*int(valid)
                image_loss = feature.new_zeros(()); alignment = feature.new_zeros(())
                if not diagnostic and valid:
                    latent = out['latent']; image_loss = F.mse_loss(latent, item['latent'].to(device))
                    pixel = vae.decode(latent/vae.config.scaling_factor).sample
                    pixel = (pixel+1)/2
                    loss = loss+cfg['image_latent_weight']*image_loss+cfg['image_pixel_weight']*F.mse_loss(pixel, item['image'].to(device))
                    teacher = F.normalize(item['clip_image'].to(device)+item['clip_text'].to(device), dim=-1)
                    alignment = (1-F.cosine_similarity(out['common_alignment'], teacher)).mean()
                    loss = loss+cfg['common_alignment_weight']*alignment
                if not diagnostic: loss = loss+cfg['papr_weight']*papr(x, cc)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss; no skipped update')
                # A final partial accumulation is scaled by its actual group size.
                block_start = (step//cfg['grad_accum'])*cfg['grad_accum']
                divisor = min(cfg['grad_accum'], total-block_start)
                (loss/divisor).backward(); step += 1
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        if not torch.isfinite(p.grad).all(): raise FloatingPointError('Nonfinite gradient: '+n)
                        if bool(p.grad.abs().sum() > 0): gradients.add(n)
                if step % cfg['grad_accum'] == 0 or step == total:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
                    opt.step(); opt.zero_grad(set_to_none=True); updates += 1
                elapsed = time.monotonic()-start
                r = dict(step=step, total_steps=total, optimizer_updates=updates, phase='feature_warmup' if warmup else label,
                    loss=float(loss.detach()), ce=float(ce.detach()), feature_mse=float(feature.detach()),
                    image_latent_mse=float(image_loss.detach()), common_alignment=float(alignment.detach()),
                    receiver_available=valid, elapsed_seconds=elapsed,
                    estimated_remaining_train_seconds=elapsed/step*(total-step),
                    cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else None)
                r['cuda_peak_reserved_gib'] = torch.cuda.max_memory_reserved(device)/2**30 if device.type == 'cuda' else None
                log.write(json.dumps(r, allow_nan=False)+'\n'); log.flush()
                if step == 1 or step % 20 == 0 or step == total:
                    atomic_json(directory.parent/'progress.json', dict(arm=model.arm, stage=directory.name, **r))
                    print('BART {} {} {}/{} CE={:.4f} feature={:.4f} elapsed={:.1f}s ETAtrain={:.1f}s'.format(
                        model.arm, label, step, total, r['ce'], r['feature_mse'], elapsed, r['estimated_remaining_train_seconds']), flush=True)
    after = model.trainable_state()
    audit = {}
    for group, prefixes in groups.items():
        names = [n for n in before if n.startswith(prefixes)]
        audit[group] = dict(parameters=len(names), nonzero_gradient=any(n in gradients for n in names),
                            changed=any(not torch.equal(before[n], after[n]) for n in names))
    verify_frozen(model, frozen)
    passed = usable > 0 and all(v['parameters'] and v['nonzero_gradient'] and v['changed'] for v in audit.values())
    atomic_json(directory/'learning_audit.json', dict(passed=passed, groups=audit, usable_frames=usable, frozen_unchanged=True))
    # Save evidence before raising, including in failed diagnostic stages.
    atomic_checkpoint(directory/'last_bridge.pth', dict(version=VERSION, candidate=model.arm,
        stage=label, config=cfg, weights=after, steps=step, optimizer_updates=updates,
        precise_resume_supported=False))
    atomic_json(directory/'TRAIN_COMPLETE.json', dict(steps=step, optimizer_updates=updates,
        learning_audit_passed=passed, wall_seconds=time.monotonic()-start))
    if not passed: raise RuntimeError('Required bridge groups did not receive gradients/update; inspect learning_audit.json')
    return after
