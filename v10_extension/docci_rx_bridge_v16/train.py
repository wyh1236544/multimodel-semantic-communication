"""TRAIN-only alignment followed by token-weighted CE; fixed last checkpoint."""
import json
import math
import time
import torch
from docci_longclip_joint_v8.backbones import signature
from docci_rx_bridge_v16.core import atomic, sha, batches
from docci_rx_bridge_v16.cache import batch
from docci_rx_bridge_v16.model import Bridge, ce


def fit(bart, items, c, arm, root, device, budget, smoke=False):
    if any(r['official_split'] != 'train' for r in items.entries):
        raise ValueError('Optimizer cannot read validation/test')
    torch.manual_seed(c['seed'])
    bridge = Bridge(bart, c).to(device).eval()
    before = signature(bart)
    initial = {n: p.detach().cpu().clone() for n, p in bridge.named_parameters()}
    seen_grad, seen_ids = set(), set()
    active = dict(bridge.named_parameters())
    total = 3 if smoke else math.ceil(len(items)/c['ce_accumulation'])*c['ce_epochs']
    align_total = math.ceil(len(items)/c['alignment_batch'])*c['alignment_epochs']
    atomic(root/'trainability.json', dict(arm=arm, trainable_parameters=sum(p.numel() for p in active.values()),
           trainable_names=sorted(active), alignment_batch=c['alignment_batch'], ce_microbatch=1,
           ce_accumulation=c['ce_accumulation'], ce_updates=total, alignment_updates=align_total,
           training_ids=[r['example_id'] for r in items.entries], frozen_BART=True,
           checkpoint_policy='fixed last, no validation selection',
           auxiliary_source=arm, target_projection='fixed random buffer seed160042',
           contrastive_batch_is_real_batch=True, CE_gradient_accumulation_is_NOT_contrastive_batch=True,
           original_text_memory_replaced=False))
    started = time.monotonic()
    try:
        optimizer = torch.optim.AdamW(bridge.resampler.parameters(), lr=c['alignment_lr'], weight_decay=c['weight_decay'])
        with (root/'alignment.jsonl').open('x', encoding='utf-8') as log:
            for step, (epoch, ix) in enumerate(batches(len(items), c['alignment_batch'], c['alignment_epochs'], c['seed'])):
                budget()
                memory, aux, gate, _ = batch(items, ix, arm, device)
                optimizer.zero_grad(set_to_none=True)
                objective = bridge.alignment(aux, memory, gate, c['temperature'])
                if not torch.isfinite(objective):
                    raise FloatingPointError('Nonfinite alignment loss')
                objective.backward()
                torch.nn.utils.clip_grad_norm_(bridge.resampler.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                row = dict(update=step+1, batch_ids=[items.entries[i]['example_id'] for i in ix], loss=float(objective.detach()))
                log.write(json.dumps(row, allow_nan=False)+'\n')
                log.flush()
                if step == 0 or (step+1) % c['log_every'] == 0 or step+1 == align_total:
                    print('ALIGN {} {}/{} loss={:.4f}'.format(arm, step+1, align_total, row['loss']), flush=True)
        optimizer = torch.optim.AdamW(list(active.values()), lr=c['ce_lr'], weight_decay=c['weight_decay'])
        ce_started = time.monotonic()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        with (root/'history.jsonl').open('x', encoding='utf-8') as log:
            for step, (epoch, ix) in enumerate(batches(len(items), c['ce_accumulation'], c['ce_epochs'], c['seed'])):
                if step >= total:
                    break
                budget()
                optimizer.zero_grad(set_to_none=True)
                denom = sum(int((items[i]['ids'] != 1).sum()) for i in ix)
                value, counted = 0., 0
                for i in ix:
                    budget()
                    memory, aux, gate, ids = batch(items, [i], arm, device)
                    summed, n = ce(bart, bridge, memory, aux, gate, ids)
                    loss = summed/denom
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite CE')
                    loss.backward()
                    value += float(loss.detach())
                    counted += n
                if counted != denom:
                    raise ValueError('Token-weighted accumulation denominator differs')
                for name, p in active.items():
                    if p.grad is not None:
                        if not torch.isfinite(p.grad).all():
                            raise FloatingPointError('Nonfinite bridge gradient: '+name)
                        if bool(p.grad.abs().sum()):
                            seen_grad.add(name)
                norm = torch.nn.utils.clip_grad_norm_(list(active.values()), 1., error_if_nonfinite=True)
                # Same predeclared schedule and batch order for every arm.
                lr = c['ce_lr']*min(1., (step+1)/min(100, max(1, total//10)))*(.1+.9*(1-step/max(1, total)))
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.step()
                seen_ids.update(items.entries[i]['example_id'] for i in ix)
                elapsed = time.monotonic()-ce_started
                row = dict(update=step+1, total_updates=total, epoch=epoch+1,
                           batch_ids=[items.entries[i]['example_id'] for i in ix], loss=value,
                           valid_target_tokens=counted, lr=lr, gradient_norm=float(norm),
                           elapsed_seconds=elapsed, eta_train_seconds=elapsed/(step+1)*(total-step-1),
                           peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else None,
                           peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30 if device.type == 'cuda' else None)
                log.write(json.dumps(row, allow_nan=False)+'\n')
                log.flush()
                if step == 0 or (step+1) % c['log_every'] == 0 or step+1 == total:
                    atomic(root/'progress.json', dict(stage='CE', arm=arm, **row))
                    print('V16 {} CE {}/{} loss={:.4f} elapsed={:.1f}s ETAtrain={:.1f}s'.format(
                          arm, step+1, total, value, elapsed, row['eta_train_seconds']), flush=True)
        audit = {}
        for group, prefixes in bridge.groups().items():
            names = [n for n in active if n.startswith(prefixes)]
            audit[group] = dict(parameters=sum(active[n].numel() for n in names),
                   nonzero_gradient=any(n in seen_grad for n in names),
                   changed=any(not torch.equal(initial[n], active[n].detach().cpu()) for n in names))
        atomic(root/'learning_audit.json', dict(groups=audit, passed=all(x['nonzero_gradient'] and x['changed'] for x in audit.values())))
        if not all(x['nonzero_gradient'] and x['changed'] for x in audit.values()):
            raise RuntimeError('Required bridge group did not learn')
        if signature(bart) != before or any(p.requires_grad or p.grad is not None for p in bart.parameters()):
            raise RuntimeError('Original BART changed or acquired parameter gradients')
        atomic(root/'frozen_audit.json', dict(before=before, after=signature(bart), unchanged=True))
        state = {n: t.detach().cpu().clone() for n, t in bridge.state_dict().items()}
        path = root/'last.pth'
        with path.open('xb') as f:
            torch.save(dict(schema='v16-bridge', arm=arm, config=c, state=state, updates=total,
                            optimizer=optimizer.state_dict()), f)
        saved = torch.load(str(path), map_location='cpu')
        bridge.load_state_dict(saved['state'], strict=True)
        if any(not torch.equal(t, bridge.state_dict()[n].detach().cpu()) for n, t in state.items()):
            raise RuntimeError('Actual disk checkpoint roundtrip differs')
        atomic(root/'TRAIN_COMPLETE.json', dict(updates=total, checkpoint_sha256=sha(path),
               actual_disk_roundtrip=True, unique_training_ids=len(seen_ids),
               alignment_updates=align_total, wall_seconds=time.monotonic()-started,
               smoke=smoke, fixed_final=True))
        return bridge
    except BaseException:
        bridge.close()
        raise
