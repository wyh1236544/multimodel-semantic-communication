"""Fixed-final, token-weighted CE and unchanged pixel-first I2 objectives."""
import json
import time
import torch
import torch.nn.functional as F
from docci_baseline_v14.core import atomic, sha, batches, updates, parameter_hash
from docci_baseline_v14.data import tensors
from docci_baseline_v14.model import loss as text_loss


def image_forward(model, src):
    wire = model.encode(src)
    b = wire.shape[0]
    # 40 is the historical clean conditioning sentinel, NEVER a measured SNR.
    out = model.receive(wire, torch.ones(b, device=wire.device, dtype=torch.bool),
                        wire.new_full((b,), 40.), wire.new_ones(b))
    return out['latent']


def image_loss(model, items, ids, backbone, device):
    src = tensors(items, ids, 'image', device)
    latent = image_forward(model, src)
    decoded = (backbone.decode(latent/float(backbone.config.scaling_factor)).sample+1)/2
    terms = []
    for j, i in enumerate(ids):
        item = items[i]
        l, t, r, b = map(int, item['rectangle'])
        target = torch.as_tensor(item['rgb'].copy(), device=device).permute(2, 0, 1).float()/255.
        terms.append(F.mse_loss(decoded[j, :, t:b, l:r], target[:, t:b, l:r]))
    pixel = torch.stack(terms).mean()
    aux = F.mse_loss(latent, src['latent'].float())
    return pixel+.01*aux


def fit(model, items, task, c, folder, device, budget, backbone=None,
        virtual_count=None, smoke=False):
    folder.mkdir(exist_ok=False)
    if any(e['official_split'] != 'train' for e in items.entries):
        raise ValueError('Optimizer may only access official TRAIN')
    model.eval()
    active = {n: p for n, p in model.named_parameters() if p.requires_grad}
    initial = {n: p.detach().cpu().clone() for n, p in active.items()}
    grad_seen, seen = set(), set()
    micro = c['text_microbatch'] if task == 'text' else 1
    accum = c['text_accumulation'] if task == 'text' else c['image_accumulation']
    effective = micro*accum
    count = virtual_count or len(items)
    epochs = c[task+'_epochs']
    total = 2 if smoke else updates(count, effective, epochs)
    optimizer = torch.optim.AdamW(list(active.values()), lr=c[task+'_learning_rate'],
                                  weight_decay=c['weight_decay'])
    atomic(folder/'trainability.json', dict(task=task, active_parameters=sum(p.numel() for p in active.values()),
        active_names=sorted(active), microbatch=micro, accumulation=accum,
        effective_descriptions_or_images=effective, total_updates=total, epochs=epochs,
        unique_training_ids=[e['example_id'] for e in items.entries],
        virtual_schedule_count=count, init_hashes=parameter_hash(model),
        new_optimizer=True, exact_optimizer_resume=False, checkpoint_policy='fixed final',
        smoke=smoke, teacher_or_KD=False,
        loss='valid-token-weighted CE' if task == 'text' else 'pixel_MSE + 0.01*latent_MSE'))
    started = time.monotonic()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    stream = batches(count, effective, epochs, c['seed'])
    if smoke:
        stream = iter([(0, list(range(min(effective, len(items))))) for _ in range(2)])
    with (folder/'history.jsonl').open('x', encoding='utf-8') as log:
        for step, (epoch, ix) in enumerate(stream):
            budget()
            ix = [i % len(items) for i in ix]
            optimizer.zero_grad(set_to_none=True)
            warm = min(c['warmup_updates'], max(1, total//10))
            fraction = min(1., (step+1)/warm)*(c['min_lr_fraction']+
                (1-c['min_lr_fraction'])*max(0., 1-step/max(1, total)))
            for g in optimizer.param_groups:
                g['lr'] = c[task+'_learning_rate']*fraction
            if task == 'text':
                # All public frames, including empty ones. Weight by actual
                # target count across the entire update, not equal chunk means.
                denominator = sum(int((items[i]['chunk_ids'] != 1).sum()) for i in ix)
            else:
                denominator = len(ix)
            value = 0.
            token_count = 0
            ids = [items.entries[i]['example_id'] for i in ix]
            seen.update(ids)
            for start in range(0, len(ix), micro):
                budget()
                part = ix[start:start+micro]
                if task == 'text':
                    src = tensors(items, part, task, device)
                    summed, n = text_loss(model, src['chunk_ids'], src['chunk_hidden'])
                    objective = summed/denominator
                    token_count += n
                else:
                    objective = image_loss(model, items, part, backbone, device)*len(part)/denominator
                if not torch.isfinite(objective):
                    raise FloatingPointError('Nonfinite loss; no skipped update')
                objective.backward()
                value += float(objective.detach())
            if task == 'text' and token_count != denominator:
                raise ValueError('Accumulated valid-token denominator differs')
            for name, p in active.items():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        raise FloatingPointError('Nonfinite gradient '+name)
                    if bool(p.grad.abs().sum() > 0):
                        grad_seen.add(name)
            norm = torch.nn.utils.clip_grad_norm_(list(active.values()), 1., error_if_nonfinite=True)
            optimizer.step()
            elapsed = time.monotonic()-started
            row = dict(update=step+1, total_updates=total, epoch=epoch+1,
                       batch_example_ids=ids, valid_target_tokens=token_count,
                       unique_examples_seen=len(seen), loss=value,
                       lr=optimizer.param_groups[0]['lr'], gradient_norm=float(norm),
                       elapsed_seconds=elapsed, eta_train_seconds=elapsed/(step+1)*(total-step-1),
                       peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None,
                       peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30 if device.type=='cuda' else None)
            log.write(json.dumps(row, allow_nan=False)+'\n')
            log.flush()
            if step == 0 or (step+1) % c['log_every'] == 0 or step+1 == total:
                atomic(folder.parent/'progress.json', dict(stage='training', task=task, arm=model.arm, **row))
                print('V14 {} {} update={}/{} unique={} loss={:.4f} elapsed={:.1f}s ETAtrain={:.1f}s'.format(
                    task, model.arm, step+1, total, len(seen), value, elapsed, row['eta_train_seconds']), flush=True)
    audits = {}
    for name, prefixes in model.groups().items():
        names = [n for n in active if n.startswith(prefixes)]
        audits[name] = dict(parameters=sum(active[n].numel() for n in names),
            nonzero_gradient=any(n in grad_seen for n in names),
            changed=any(not torch.equal(initial[n], active[n].detach().cpu()) for n in names))
    passed = all(v['parameters'] and v['nonzero_gradient'] and v['changed'] for v in audits.values())
    atomic(folder/'learning_audit.json', dict(passed=passed, groups=audits))
    if not passed:
        raise RuntimeError('Required codec group did not learn; preserve diagnostics')
    saved = model.saved()
    checkpoint = folder/'last.pth'
    with checkpoint.open('xb') as f:
        torch.save(dict(schema='v14-checkpoint', task=task, arm=model.arm, state=saved,
                        config=c, updates=total, optimizer=optimizer.state_dict(),
                        torch_rng=torch.get_rng_state()), f)
    actual = torch.load(str(checkpoint), map_location='cpu')
    model.restore(actual['state'])
    if any(not torch.equal(v, model.saved()[k]) for k, v in saved.items()):
        raise RuntimeError('Actual disk checkpoint roundtrip failed')
    atomic(folder/'TRAIN_COMPLETE.json', dict(updates=total, checkpoint_sha256=sha(checkpoint),
        actual_disk_roundtrip=True, unique_examples_seen=len(seen), smoke=smoke,
        fixed_final=True, training_seconds=time.monotonic()-started))
