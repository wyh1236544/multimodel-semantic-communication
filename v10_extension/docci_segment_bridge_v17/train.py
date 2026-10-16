"""Equal four-arm schedules; no loss maximization on wrong-image generations."""
import json
import math
import time
import torch
from docci_segment_bridge_v17.core import (atomic, sha, batches, condition, seed, expected_parameters,
                                         kind, noise_seed)
from docci_segment_bridge_v17.data import Receiver, materialize
from docci_segment_bridge_v17.model import SegmentBridge, ce, canonical
from docci_longclip_joint_v8.backbones import signature


def learning_rate(c, step, total):
    return c['ce_lr']*min(1., (step+1)/min(100, max(1, total//10)))*(.1+.9*(1-step/max(1, total)))


def fit(model, prepared, arm, root, device, budget, smoke=False):
    c, items = prepared.c, prepared.train
    if any(r['official_split'] != 'train' for r in items.entries):
        raise ValueError('TRAIN only optimizer')
    torch.manual_seed(c['seed'])
    bridge = SegmentBridge(model.bart.config.d_model, c).to(device).eval()
    count = sum(p.numel() for p in bridge.parameters())
    if count != expected_parameters(model.bart.config.d_model, c['width'], c['slots']):
        raise ValueError('Independent bridge parameter count differs')
    original = signature(model)
    initial = {n: p.detach().cpu().clone() for n, p in bridge.named_parameters()}
    seen, receiver = set(), Receiver(model)
    total = 5 if smoke else math.ceil(len(items)/c['ce_accumulation'])*c['ce_epochs']
    atomic(root/'trainability.json', dict(trainable_parameters=count, frozen_backbones=True,
           alignment_updates=math.ceil(len(items)/c['alignment_batch'])*c['alignment_epochs'], ce_updates=total,
           TRAIN_ids=[r['example_id'] for r in items.entries], ce_microbatch=1, ce_accumulation=c['ce_accumulation'],
           source=kind(arm), native_teacher_TRAIN_only=True, fixed_final=True, exact_resume=False))
    started = time.monotonic()
    def grad_seen():
        for name, p in bridge.named_parameters():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    raise FloatingPointError('Nonfinite gradient: '+name)
                if bool(p.grad.abs().sum()):
                    seen.add(name)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=c['alignment_lr'], weight_decay=c['weight_decay'])
    with (root/'alignment.jsonl').open('x', encoding='utf-8') as log:
        for step, (epoch, ix) in enumerate(batches(len(items), c['alignment_batch'], c['alignment_epochs'], c['seed'])):
            budget()
            channels = [condition(arm, 'align', epoch, items.entries[i]['example_id']) for i in ix]
            memory, aux, gate, ids, stats = materialize(items, ix, receiver, arm, channels, 'align', epoch, device)
            teacher = torch.as_tensor(prepared.teacher[ix], device=device)
            counts = torch.as_tensor(prepared.counts[ix], device=device)
            # Nonidentical in-batch negatives. Actual batch, not accumulation.
            wrong = aux.roll(1, 0)
            optimizer.zero_grad(set_to_none=True)
            itc, segment, match = bridge.auxiliary(memory, aux, gate, teacher, counts, wrong, True, c['temperature'])
            loss = itc+.5*segment+.2*match
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite alignment loss')
            loss.backward()
            grad_seen()
            torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            row = dict(update=step+1, epoch=epoch+1, batch_ids=[items.entries[i]['example_id'] for i in ix],
                       conditions=channels, observation_hashes=[s['memory_sha256'] for s in stats],
                       loss=float(loss.detach()), contrast=float(itc.detach()), segment=float(segment.detach()), matching=float(match.detach()))
            log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush()
            if step == 0 or (step+1) % c['log_every'] == 0:
                print('ALIGN {} {}/{} loss={:.4f}'.format(arm, step+1,
                      math.ceil(len(items)/c['alignment_batch'])*c['alignment_epochs'], row['loss']), flush=True)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=c['ce_lr'], weight_decay=c['weight_decay'])
    ce_start = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    unique = set()
    with (root/'history.jsonl').open('x', encoding='utf-8') as log:
        schedule = list(batches(len(items), c['ce_accumulation'], c['ce_epochs'], c['seed']))[:total]
        for step, (epoch, ix) in enumerate(schedule):
            budget()
            optimizer.zero_grad(set_to_none=True)
            denominator = sum(int((items[i]['ids'] != 1).sum()) for i in ix)
            value = 0.; ce_value = 0.; observations = []; channels = []; donors = []
            for i in ix:
                budget()
                eid = items.entries[i]['example_id']
                channel = condition(arm, 'ce', epoch, eid)
                channels.append(channel)
                memory, aux, gate, ids, stats = materialize(items, [i], receiver, arm, [channel], 'ce', epoch, device)
                donor = (i+1+seed('ce-donor', epoch, eid) % (len(items)-1)) % len(items)
                donors.append(items.entries[donor]['example_id'])
                donor_channel = condition(arm, 'ce', epoch, items.entries[donor]['example_id'])
                if kind(arm) == 'self':
                    _, wrong, _, _, _ = materialize(items, [donor], receiver, arm, [donor_channel], 'ce', epoch, device)
                else:
                    visual = torch.as_tensor(items[donor]['oracle_clip'][None], device=device)
                    wrong = canonical(memory, visual, 'oracle_clip')
                summed, n, features = ce(model.bart, bridge, memory, aux, gate, ids, return_features=True)
                teacher = torch.as_tensor(prepared.teacher[i:i+1], device=device)
                counts = torch.as_tensor(prepared.counts[i:i+1], device=device)
                _, segment, match = bridge.auxiliary(memory, aux, gate, teacher, counts, wrong, features=features)
                objective = summed/denominator+(c['matching_weight']*match+c['segment_weight']*segment)*n/denominator
                if not torch.isfinite(objective):
                    raise FloatingPointError('Nonfinite CE/auxiliary loss')
                objective.backward()
                value += float(objective.detach()); ce_value += float(summed.detach())/denominator
                unique.add(eid); observations.append(stats[0])
            grad_seen()
            norm = torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1., error_if_nonfinite=True)
            lr = learning_rate(c, step, total)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()
            elapsed = time.monotonic()-ce_start
            row = dict(update=step+1, total_updates=total, epoch=epoch+1,
                       batch_ids=[items.entries[i]['example_id'] for i in ix], donor_ids=donors,
                       conditions=channels, observations=observations, valid_target_tokens=denominator,
                       loss=value, CE=ce_value, lr=lr, gradient_norm=float(norm), elapsed_seconds=elapsed,
                       eta_train_seconds=elapsed/(step+1)*(total-step-1),
                       peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                       peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
            log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush()
            if step == 0 or (step+1) % c['log_every'] == 0 or step+1 == total:
                atomic(root/'progress.json', dict(stage='CE', arm=arm, **{k: row[k] for k in ('update', 'total_updates', 'CE', 'elapsed_seconds', 'eta_train_seconds')}))
                print('V17 {} CE {}/{} CE={:.4f} elapsed={:.1f}s ETA_this_arm_CE={:.1f}s'.format(
                      arm, step+1, total, ce_value, elapsed, row['eta_train_seconds']), flush=True)
            if step+1 == total or (step+1 < len(schedule) and schedule[step+1][0] != epoch):
                with (root/('epoch_'+str(epoch+1)+'.pth')).open('xb') as f:
                    torch.save(dict(schema='v17-epoch-partial-not-final', arm=arm, state=bridge.state_dict(), update=step+1), f)
    groups = {}
    for group, prefixes in bridge.groups().items():
        names = [n for n, _ in bridge.named_parameters() if n.startswith(prefixes)]
        groups[group] = dict(parameters=sum(dict(bridge.named_parameters())[n].numel() for n in names),
                             nonzero_gradient=any(n in seen for n in names),
                             changed=any(not torch.equal(initial[n], dict(bridge.named_parameters())[n].detach().cpu()) for n in names))
    if not all(x['nonzero_gradient'] and x['changed'] for x in groups.values()):
        raise RuntimeError('A required repair/alignment/matching group did not learn')
    if signature(model) != original or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError('Frozen original pipeline changed')
    state = {n: v.detach().cpu().clone() for n, v in bridge.state_dict().items()}
    with (root/'last.pth').open('xb') as f:
        torch.save(dict(schema='v17-segment-bridge', arm=arm, state=state, config=c,
                        updates=total, optimizer=optimizer.state_dict()), f)
    disk = torch.load(str(root/'last.pth'), map_location='cpu')
    bridge.load_state_dict(disk['state'], strict=True)
    if any(not torch.equal(v, bridge.state_dict()[n].detach().cpu()) for n, v in state.items()):
        raise RuntimeError('Saved checkpoint roundtrip differs')
    atomic(root/'learning_audit.json', dict(groups=groups, passed=True))
    atomic(root/'frozen_audit.json', dict(before=original, after=signature(model), unchanged=True))
    atomic(root/'TRAIN_COMPLETE.json', dict(updates=total, alignment_updates=step_count(c, len(items)),
           unique_training_ids=len(unique), wall_seconds=time.monotonic()-started,
           checkpoint_sha256=sha(root/'last.pth'), actual_disk_roundtrip=True, fixed_final=True, smoke=smoke))
    return bridge


def step_count(c, n):
    return math.ceil(n/c['alignment_batch'])*c['alignment_epochs']
