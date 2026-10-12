"""Matched phased joint learning; source labels are loss-only, not receiver inputs."""
import json
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from clip_bart_joint_v5 import VERSION
from clip_bart_joint_v5.core import digest, metrics, summarize
from clip_bart_joint_v5.data import source, clean_args
from bart_latent_codec_v4.model import prediction, generate
from bart_latent_codec_v4.workflow import signature, frozen_check
from bart_latent_codec_v4.channel import proxy_tensor
from bart_clip_joint_v1.core import tensor_signature


@torch.no_grad()
def text_controls(model, items, tok, device, budget, path):
    outputs = []
    for item in items:
        budget(); outputs.append(model.receive(*clean_args(source(model, item, device)))['text'])
    rows = []
    with path.open('x', encoding='utf-8') as f:
        for control in ('matched', 'zero', 'shuffled'):
            for i, item in enumerate(items):
                budget(); donor = (i+1)%len(items); output = dict(outputs[donor] if control == 'shuffled' else outputs[i])
                if control == 'zero': output['memory'] = torch.zeros_like(output['memory'])
                result = generate(model.text.bart, output, tok)
                r = dict(image_id=item['image_id'], source_text=item['text'], covered_text=item['covered_text'],
                    received_caption=result['text'], control=control, source_split=item['split'], input_truncated=item['truncated'],
                    stopped_by_capacity=result['stopped_by_capacity'], receiver_failure=result['receiver_failure'],
                    donor_image_id=items[donor]['image_id'] if control == 'shuffled' else None,
                    **metrics(item['text'], item['covered_text'], result['text']))
                rows.append(r); f.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n'); f.flush()
    return {k: summarize([r for r in rows if r['control'] == k]) for k in ('matched', 'zero', 'shuffled')}


def train(model, items, c, tc, vae, scaling, device, folder, phase, tok, budget, frozen, atomic_json, atomic_checkpoint):
    folder.mkdir(exist_ok=False); total = len(items)*(c['clean_epochs'] if phase == 'clean' else c['continue_epochs'])
    model.configure(model.arm); initial = model.bridge_state()
    named = {n: p for n, p in model.named_parameters() if n in initial}
    lr = c['joint_lr'] if phase == 'clean' else c['continue_lr']
    groups = [dict(params=[p for n, p in named.items() if n.startswith('text.')], lr=c['text_lr']),
        dict(params=[p for n, p in named.items() if not n.startswith('text.')], lr=lr)]
    optimizer = torch.optim.AdamW(groups, weight_decay=c['weight_decay']); gradients = set()
    updates = 0; start = time.monotonic(); optimizer.zero_grad(set_to_none=True)
    item = items[0]; x = source(model, item, device); out = model.receive(*clean_args(x))
    with torch.no_grad(): xx = source(model, item, device); yy = model.receive(*clean_args(xx))
    errors = dict(wire=float((x.detach()-xx).abs().max()), text=float((out['text']['memory'].detach()-yy['text']['memory']).abs().max()),
        image=float((out['latent'].detach()-yy['latent']).abs().max()))
    atomic_json(folder/'forward_consistency.json', dict(errors=errors, tolerance=c['forward_tolerance']))
    if max(errors.values()) > c['forward_tolerance']: raise RuntimeError('Actual grad/no-grad interface mismatch')
    del x, xx, out, yy
    atomic_json(folder/'trainability.json', dict(registered_bridge_parameters=sum(p.numel() for p in named.values()),
        phase=phase, total_steps=total, grad_accum=c['grad_accum'], text_codec_lr=c['text_lr'], other_lr=lr,
        text_warm_frozen_steps=c['joint_warm_steps'] if phase == 'clean' else 0,
        full_backbones_frozen=True, one_seed=c['seed']))
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    with (folder/'history.jsonl').open('x') as log, (folder/'snapshots.jsonl').open('x', encoding='utf-8') as snapshots:
        for step in range(total):
            budget(); epoch, pos = divmod(step, len(items))
            if pos == 0: order = np.random.default_rng(c['seed']+epoch).permutation(len(items))
            item = items[int(order[pos])]; model.configure(model.arm, phase == 'clean' and step < c['joint_warm_steps'])
            wire = source(model, item, device); noisy = phase == 'fso' and step%c['clean_anchor_period'] != 0
            stats = []
            if noisy:
                observed, stats = proxy_tensor(wire, tc, step, [item['image_id']])
                args = clean_args(observed)
                pilot = wire.new_tensor([stats[0]['pilot_snr_db']])
                args = (args[0], args[1], pilot, torch.sigmoid(pilot.clamp(-40,40)*math.log(10)/10))
            else: args = clean_args(wire)
            out = model.receive(*args)
            logits, labels = prediction(model.text.bart, item['input_ids'].to(device), item['mask'].to(device),
                out['text']['memory'], out['text']['receiver_mask'])
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
            latent = F.mse_loss(out['latent'], item['latent'].to(device))
            pixel = (vae.decode(out['latent']/scaling).sample+1)/2
            pix = F.mse_loss(pixel, item['image'].to(device))
            alignment = ((1-F.cosine_similarity(out['alignment_image'], item['clip_image'].to(device))).mean()
                +(1-F.cosine_similarity(out['alignment_text'], item['clip_text'].to(device))).mean())/2
            coarse = F.mse_loss(out['coarse_latent'], F.adaptive_avg_pool2d(item['latent'].to(device), (4, 4)))
            anchor = F.mse_loss(out['text']['memory'], item['text_anchor'].to(device))
            loss = ce+c['image_latent_weight']*latent+c['image_pixel_weight']*pix+c['common_alignment_weight']*alignment
            loss = loss+c['common_coarse_weight']*coarse+c['text_anchor_weight']*anchor
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite joint loss')
            divisor = min(c['grad_accum'], total-(step//c['grad_accum'])*c['grad_accum'])
            (loss/divisor).backward()
            for n, p in named.items():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all(): raise FloatingPointError('Nonfinite gradient '+n)
                    if bool(p.grad.abs().sum()>0): gradients.add(n)
            if (step+1)%c['grad_accum'] == 0 or step+1 == total:
                torch.nn.utils.clip_grad_norm_(list(named.values()), 1., error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); updates += 1
            elapsed = time.monotonic()-start
            r = dict(step=step+1, total_steps=total, optimizer_updates=updates, image_id=item['image_id'],
                ce=float(ce.detach()), image_latent_mse=float(latent.detach()), image_pixel_mse=float(pix.detach()),
                common_alignment=float(alignment.detach()), common_coarse=float(coarse.detach()), text_anchor=float(anchor.detach()),
                loss=float(loss.detach()), noisy=noisy, proxy=stats, elapsed_seconds=elapsed,
                estimated_remaining_train_seconds=elapsed/(step+1)*(total-step-1),
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30 if device.type=='cuda' else None)
            log.write(json.dumps(r, allow_nan=False)+'\n'); log.flush()
            if (step+1)%c['snapshot_every'] == 0 or step+1 == total:
                with torch.no_grad():
                    sample = items[0]; clean = model.receive(*clean_args(source(model, sample, device)))
                    result = generate(model.text.bart, clean['text'], tok)
                    snapshots.write(json.dumps(dict(step=step+1, source_text=sample['text'], result=result), ensure_ascii=False)+'\n'); snapshots.flush()
            if step == 0 or (step+1)%20 == 0:
                atomic_json(folder.parents[1]/'progress.json', dict(arm=model.arm, phase=phase, **r))
                print('JOINT {} {} {}/{} CE={:.4f} image={:.4f} elapsed={:.1f}s ETAtrain={:.1f}s'.format(
                    model.arm, phase, step+1, total, r['ce'], r['image_latent_mse'], elapsed, r['estimated_remaining_train_seconds']), flush=True)
    model.configure(model.arm); saved = model.bridge_state(); audit = {}
    for name, prefixes in model.groups().items():
        ns = [n for n in initial if n.startswith(prefixes)]
        audit[name] = dict(tensors=len(ns), nonzero_gradient=any(n in gradients for n in ns),
            changed=any(not torch.equal(initial[n], saved[n]) for n in ns))
    for module, signatures in frozen: frozen_check(module, signatures)
    passed = all(v['tensors'] and v['nonzero_gradient'] and v['changed'] for v in audit.values())
    atomic_json(folder/'learning_audit.json', dict(passed=bool(passed), groups=audit, frozen_unchanged=True,
        registered_bridge_parameters=sum(p.numel() for p in named.values()), observed_gradient_parameters=sum(named[n].numel() for n in gradients)))
    atomic_checkpoint(folder/'last.pth', dict(version=VERSION, arm=model.arm, phase=phase, config=c, weights=saved, precise_resume_supported=False))
    loaded = torch.load(str(folder/'last.pth'), map_location='cpu')
    if loaded['version'] != VERSION or loaded['config'] != c or loaded['arm'] != model.arm: raise RuntimeError('Checkpoint identity mismatch')
    model.restore(loaded['weights']); restored = model.bridge_state()
    if any(not torch.equal(v, restored[n]) for n, v in saved.items()): raise RuntimeError('Disk reload tensor mismatch')
    atomic_json(folder/'reload_audit.json', dict(passed=True, actual_disk_reload=True))
    if not passed: raise RuntimeError('Required joint groups did not learn; preserve audit')
    atomic_json(folder/'TRAIN_COMPLETE.json', dict(steps=total, optimizer_updates=updates, checkpoint_sha256=digest(folder/'last.pth'),
        wall_seconds=time.monotonic()-start, quality_not_implied=True))
    return saved
