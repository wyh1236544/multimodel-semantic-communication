"""Fixed-last-step audits and actual greedy controls, not teacher-forcing success."""
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from text_recovery_v3 import VERSION
from text_recovery_v3.core import row, summarize, schedule, tensor_signature, geometry
from text_recovery_v3.model import logits, generate, adapters
from text_recovery_v3.data import transmit


def frozen_signature(model):
    return {n: tensor_signature(v) for n, v in model.named_parameters() if not v.requires_grad}


def verify_frozen(model, before):
    named = dict(model.named_parameters())
    if any(named[n].grad is not None or tensor_signature(named[n]) != h for n, h in before.items()):
        raise RuntimeError('Frozen weights changed/received gradients')


def loss_terms(bart, item, output, device, weights):
    target = item['bart_hidden'].to(device); mask = item['mask'].to(device)
    weights_mask = mask.to(target)+(~mask).to(target)*weights['pad_weight']
    feature = ((output['memory']-target).square().mean(-1)*weights_mask).sum()/weights_mask.sum().clamp_min(1)
    ce = feature.new_zeros(())
    if weights['ce_weight']:
        prediction, labels = logits(bart, item['input_ids'].to(device), mask, output['memory'], output['receiver_mask'])
        ce = F.cross_entropy(prediction.reshape(-1, prediction.shape[-1]), labels.reshape(-1), ignore_index=-100)
    return feature, ce


@torch.no_grad()
def controls(model, items, tokenizer, device, budget, path):
    received = []
    for item in items:
        budget(); received.append(model.receive_text(transmit(model, item, device)))
    rows = []
    with path.open('x', encoding='utf-8') as f:
        for control in ('matched', 'zero', 'shuffled'):
            for i, item in enumerate(items):
                budget(); donor = (i+1)%len(items)
                out = received[donor] if control == 'shuffled' else received[i]
                memory = torch.zeros_like(out['memory']) if control == 'zero' else out['memory']
                # Zero-memory diagnostic retains that frame's RECEIVED header, never source length.
                result = generate(model.bart, memory, out['receiver_mask'], tokenizer, bool(out['available'][0]))
                entry = row(item, model.experiment_arm, control, result,
                    received_length=out['received_length'][0], memory_sha256=tensor_signature(memory),
                    donor_image_id=items[donor]['image_id'] if control == 'shuffled' else None,
                    intervention='fixed decoder-memory intervention, not new physical transmission', geometry=geometry(model.experiment_arm))
                rows.append(entry); f.write(json.dumps(entry, ensure_ascii=False, allow_nan=False)+'\n'); f.flush()
    return {k: summarize([r for r in rows if r['control'] == k]) for k in ('matched', 'zero', 'shuffled')}


def train(model, items, c, device, folder, tokenizer, budget, atomic_json, atomic_checkpoint, diagnostic=True, native=False):
    folder.mkdir(exist_ok=False); active = {n:p for n,p in model.named_parameters() if p.requires_grad}
    if not active: raise ValueError('No trainable parameters')
    initial = {n:p.detach().cpu().clone() for n,p in active.items()}; frozen = frozen_signature(model)
    arm = model.experiment_arm
    if native:
        groups = {'decoder_lora': ('model.decoder.',)}
    else:
        groups = model.groups()
        item = items[0]; x = transmit(model, item, device); out = model.receive_text(x)
        with torch.no_grad():
            xx = transmit(model, item, device); yy = model.receive_text(xx)
        errors = dict(source=float((x.detach()-xx).abs().max()), receiver=float((out['memory'].detach()-yy['memory']).abs().max()))
        atomic_json(folder/'forward_consistency.json', dict(errors=errors, tolerance=c['forward_tolerance'],
            passed=all(v <= c['forward_tolerance'] for v in errors.values())))
        if any(v > c['forward_tolerance'] for v in errors.values()): raise RuntimeError('Grad/no-grad mismatch')
        del x, out, xx, yy
    total = c['native_steps'] if native else c['diagnostic_steps'] if diagnostic else len(items)*c['clean_epochs']
    rates = []
    for is_lora in (False, True):
        selected = [p for n,p in active.items() if n.endswith(('.a', '.b')) == is_lora]
        if selected: rates.append(dict(params=selected, lr=c['lora_lr'] if is_lora else c['bridge_lr'] if model.s['ramp'] else c['legacy_lr']))
    opt = torch.optim.AdamW(rates, weight_decay=c['weight_decay']); opt.zero_grad(set_to_none=True)
    atomic_json(folder/'trainability.json', dict(active_parameters=sum(p.numel() for p in active.values()),
        lora_parameters=sum(p.numel() for n,p in active.items() if n.endswith(('.a', '.b'))), active_names=list(active),
        total_steps=total, optimizer_updates=(total+c['grad_accum']-1)//c['grad_accum'], native_diagnostic=native,
        registered_parameters=sum(p.numel() for p in model.parameters()), clean_only=True))
    start = time.monotonic(); gradients = set(); updates = 0; order = []
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    with (folder/'history.jsonl').open('x') as history, (folder/'generation_snapshots.jsonl').open('x', encoding='utf-8') as samples:
        for step in range(total):
            budget()
            if step%len(items) == 0: order = np.random.default_rng(c['seed']+step//len(items)).permutation(len(items)).tolist()
            item = items[order[step%len(items)]]
            if native:
                output = dict(memory=item['bart_hidden'].to(device), receiver_mask=item['mask'].to(device))
                weights = dict(ce_weight=1., feature_weight=0., pad_weight=0.)
            else:
                output = model.receive_text(transmit(model, item, device)); weights = schedule(step, total, c, arm, diagnostic)
            feature, ce = loss_terms(model if native else model.bart, item, output, device, weights)
            loss = weights['ce_weight']*ce+weights['feature_weight']*feature
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
            divisor = min(c['grad_accum'], total-(step//c['grad_accum'])*c['grad_accum'])
            (loss/divisor).backward()
            for n,p in active.items():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all(): raise FloatingPointError('Nonfinite gradient '+n)
                    if bool(p.grad.abs().sum()>0): gradients.add(n)
            if (step+1)%c['grad_accum'] == 0 or step+1 == total:
                torch.nn.utils.clip_grad_norm_(list(active.values()), 1., error_if_nonfinite=True)
                opt.step(); opt.zero_grad(set_to_none=True); updates += 1
            elapsed = time.monotonic()-start
            record = dict(step=step+1, total_steps=total, optimizer_updates=updates,
                feature_mse=float(feature.detach()), ce=float(ce.detach()), loss=float(loss.detach()), **weights,
                elapsed_seconds=elapsed, estimated_remaining_train_seconds=elapsed/(step+1)*(total-step-1),
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30 if device.type=='cuda' else None)
            history.write(json.dumps(record, allow_nan=False)+'\n'); history.flush()
            if (step+1)%c['snapshot_every'] == 0 or step+1 == total:
                with torch.no_grad():
                    for sample in items[:2]:
                        budget()
                        rx = dict(memory=sample['bart_hidden'].to(device), receiver_mask=sample['mask'].to(device)) if native else model.receive_text(transmit(model, sample, device))
                        result = generate(model if native else model.bart, rx['memory'], rx['receiver_mask'], tokenizer)
                        entry = row(sample, 'native_lora' if native else arm, 'snapshot', result, step=step+1)
                        samples.write(json.dumps(entry, ensure_ascii=False, allow_nan=False)+'\n'); samples.flush()
            if (step+1)%20 == 0 or step == 0:
                atomic_json(folder.parent/'progress.json', dict(stage=folder.name, **record))
                print('TEXT {} {}/{} CE={:.4f} MSE={:.4f} CEweight={:.3f} elapsed={:.1f}s ETAtrain={:.1f}s'.format(
                    folder.name, step+1, total, record['ce'], record['feature_mse'], weights['ce_weight'], elapsed, record['estimated_remaining_train_seconds']), flush=True)
    audit = {}
    for key, prefixes in groups.items():
        names = [n for n in active if n.startswith(prefixes)]
        audit[key] = dict(tensors=len(names), nonzero_gradient=any(n in gradients for n in names),
            changed=any(not torch.equal(initial[n], active[n].detach().cpu()) for n in names))
    verify_frozen(model, frozen)
    passed = all(v['tensors'] and v['nonzero_gradient'] and v['changed'] for v in audit.values())
    atomic_json(folder/'learning_audit.json', dict(passed=bool(passed), groups=audit, frozen_unchanged=True,
        eligible_parameters=sum(p.numel() for p in active.values()),
        observed_gradient_parameters=sum(active[n].numel() for n in gradients)))
    saved = {n:p.detach().cpu().clone() for n,p in active.items()}
    atomic_checkpoint(folder/'last.pth', dict(version=VERSION, arm='native_lora' if native else arm,
        config=c, diagnostic=diagnostic, weights=saved, precise_resume_supported=False))
    # Real disk reload, not in-memory serialization only.
    payload = torch.load(str(folder/'last.pth'), map_location='cpu')
    if payload['version'] != VERSION or payload['config'] != c or set(payload['weights']) != set(saved): raise RuntimeError('Reload identity mismatch')
    with torch.no_grad():
        for n,p in active.items(): p.copy_(payload['weights'][n].to(p))
    if any(not torch.equal(p.detach().cpu(), saved[n]) for n,p in active.items()): raise RuntimeError('Reload tensor mismatch')
    atomic_json(folder/'reload_audit.json', dict(passed=True, tensor_count=len(saved), exact_disk_reload=True))
    if not passed: raise RuntimeError('Required training groups did not learn; preserve audit')
    atomic_json(folder/'TRAIN_COMPLETE.json', dict(steps=total, optimizer_updates=updates, wall_seconds=time.monotonic()-start,
        quality_not_implied=True, checkpoint_sha256=__import__('hashlib').sha256((folder/'last.pth').read_bytes()).hexdigest()))
    return saved
