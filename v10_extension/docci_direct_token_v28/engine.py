"""Independent modalities, paired-compute two-view training, fixed final weights."""
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image
from docci_direct_token_v28.core import (atomic, read, rows, sha, seed, schedule,
    training_snr, trials, geometry, PARENTS, WIDTH)
from docci_direct_token_v28 import backend
from docci_direct_token_v28.channel import transmit, array_sha
from docci_baseline_v14.core import parameter_hash
from docci_longclip_joint_v8.backbones import signature


@torch.no_grad()
def replay(model, backbone, tok, items, task, receipt, device, budget):
    """Actual saved parent, same source, historical clean outputs before adapting."""
    from docci_baseline_v14.model import decode
    from utils.metrics import batch_image_metrics
    old = {r['example_id']: r for r in rows(receipt['reference'])
           if r.get('control') == 'matched' and r.get('snr_db') is None
           and r.get('strategy', 'greedy') in ('greedy', 'image')}
    results = []
    for i in range(min(2, len(items))):
        budget(); item = items[i]; eid = item['example_id']
        if eid not in old:
            raise ValueError('Historical replay source missing: '+eid)
        wire = backend.encode(model, backend.source(item, task, device), task)
        observed, stats = transmit(wire, task, None, 0)
        rx = backend.receive(model, observed, task, stats)
        if task == 'text':
            result = decode(backbone, rx, tok, 'greedy')
            if result['chunk_generated_token_ids'] != old[eid]['chunk_generated_token_ids']:
                raise ValueError('Actual V18 parent clean token replay differs')
        else:
            target = torch.as_tensor(item['rgb'].copy(), device=device).permute(2, 0, 1)[None].float()/255.
            rgb = backend.pixels(backbone, rx['latent']).clamp(0, 1)
            l, t, r, b = map(int, item['rectangle'])
            score = batch_image_metrics(rgb[:, :, t:b, l:r], target[:, :, t:b, l:r])
            for name in ('psnr', 'ssim'):
                if not math.isclose(float(score[name][0]), old[eid][name], rel_tol=3e-6, abs_tol=3e-5):
                    raise ValueError('Actual V14 parent clean metric replay differs: '+name)
        results.append(eid)
    return dict(passed=True, example_ids=results, reference_sha256=receipt['reference_sha256'],
                real_saved_checkpoint=True, no_training=True)


def fit(args, c, task, arm, train, validation, inputs, root, device, budget):
    root.mkdir(parents=True, exist_ok=False)
    model, backbone, tok, br = backend.make(args, task, device)
    atomic(root/'REPLAY.json', replay(model, backbone, tok, validation, task, read(inputs/'INPUTS.json'), device, budget))
    frozen = signature(backbone); initial_hash = parameter_hash(model)
    active = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not active or any(r['official_split'] != 'train' for r in train.entries):
        raise ValueError('TRAIN-only optimizer contract failed')
    initial = {n: p.detach().cpu().clone() for n, p in active.items()}
    seen_grad = set(); seen = set(); stream = schedule(c, task); total = len(stream)
    optimizer = torch.optim.AdamW(list(active.values()), lr=c[task+'_lr'], weight_decay=c['weight_decay'])
    atomic(root/'identity.json', dict(task=task, arm=arm, config=c,
        parent_sha256=PARENTS[task]['sha256'], initial_hashes=initial_hash,
        trainable_parameters=sum(p.numel() for p in active.values()),
        groups={k: list(v) for k, v in model.groups().items()},
        train_ids=[r['example_id'] for r in train.entries], two_views=True,
        native_source_cache_only=True, checkpoint_policy='fixed final', exact_resume=False,
        backbone_frozen=True, common=False, cross_modal=False, R=1, geometry=geometry(task)))
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats(device)
    with (root/'history.jsonl').open('x', encoding='utf-8') as log:
        for step, (epoch, ix) in enumerate(stream):
            budget(); optimizer.zero_grad(set_to_none=True)
            warmup = min(100, max(1, total//10))
            factor = min(1., (step+1)/warmup)*(.1+.9*max(0., 1-step/total))
            for group in optimizer.param_groups:
                group['lr'] = c[task+'_lr']*factor
            denominator = sum(train.entries[i]['source_tokens']+16 for i in ix) if task == 'text' else len(ix)
            clean_loss = 0.; noisy_loss = 0.; observations = []
            for i in ix:
                item = train[i]; eid = item['example_id']; seen.add(eid)
                noisy_snr = training_snr(c, arm, epoch, eid)
                views = []
                for view, snr, weight in [('clean', None, c['clean_weight']),
                                          ('noisy', noisy_snr, 1-c['clean_weight'])]:
                    budget()
                    src = backend.source(item, task, device)
                    wire = backend.encode(model, src, task)
                    observed, stats = transmit(wire, task, snr, seed(c['seed'], 'train', task, epoch, eid, view))
                    rx = backend.receive(model, observed, task, stats)
                    if task == 'text':
                        value, n = backend.text_objective(model, src['chunk_ids'], rx)
                        if n != item['source_tokens']+16:
                            raise ValueError('Public full-frame valid-token count differs')
                    else:
                        value = backend.image_objective(backbone, rx['latent'], src, item, device)
                    value = value/denominator
                    if not torch.isfinite(value):
                        raise FloatingPointError('Nonfinite objective; no skipped updates')
                    (value*weight).backward()
                    if view == 'clean': clean_loss += float(value.detach())
                    else: noisy_loss += float(value.detach())
                    views.append(dict(view=view, weight=weight, **stats))
                    del value, rx, wire, observed, src
                observations.append(dict(example_id=eid, views=views))
            for name, p in active.items():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        raise FloatingPointError('Nonfinite gradient: '+name)
                    if bool(p.grad.abs().sum() > 0): seen_grad.add(name)
            norm = torch.nn.utils.clip_grad_norm_(list(active.values()), 1., error_if_nonfinite=True)
            optimizer.step()
            elapsed = time.monotonic()-started
            row = dict(update=step+1, total_updates=total, epoch=epoch,
                batch_ids=[train.entries[i]['example_id'] for i in ix], denominator=denominator,
                clean_loss=clean_loss, noisy_loss=noisy_loss,
                loss=c['clean_weight']*clean_loss+(1-c['clean_weight'])*noisy_loss,
                lr=optimizer.param_groups[0]['lr'], gradient_norm=float(norm), observations=observations,
                elapsed_seconds=elapsed, eta_train_seconds=elapsed/(step+1)*(total-step-1),
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
            log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush()
            if step == 0 or (step+1) % c['log_every'] == 0 or step+1 == total:
                atomic(root.parent.parent/'progress.json', dict(stage='train', task=task, arm=arm,
                    **{k: v for k, v in row.items() if k != 'observations'}))
                print('V28 {} {} {}/{} loss={:.4f} ETAtrain={:.1f}s'.format(
                    task, arm, step+1, total, row['loss'], row['eta_train_seconds']), flush=True)
    audits = {}
    for group, prefixes in model.groups().items():
        names = [n for n in active if n.startswith(prefixes)]
        audits[group] = dict(parameters=sum(active[n].numel() for n in names),
            gradient=any(n in seen_grad for n in names),
            changed=any(not torch.equal(initial[n], active[n].detach().cpu()) for n in names))
    if sum(v['parameters'] for v in audits.values()) != sum(p.numel() for p in active.values()):
        raise ValueError('Trainable group coverage is not exhaustive')
    if not all(v['parameters'] and v['gradient'] and v['changed'] for v in audits.values()):
        raise RuntimeError('Required TX/RX group failed to learn')
    if signature(backbone) != frozen or any(p.grad is not None for p in backbone.parameters()):
        raise RuntimeError('Frozen backbone mutated/received parameter gradients')
    saved = model.saved(); checkpoint = root/'last.pth'
    with checkpoint.open('xb') as f:
        torch.save(dict(schema='v28-checkpoint', task=task, arm=arm, config=c,
            parent_sha256=PARENTS[task]['sha256'], state=saved, optimizer=optimizer.state_dict(),
            updates=total, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device)), f)
    model.restore(torch.load(str(checkpoint), map_location='cpu')['state'])
    restored = model.saved()
    if any(not torch.equal(v, restored[k]) for k, v in saved.items()):
        raise ValueError('Actual disk checkpoint roundtrip differs')
    atomic(root/'learning_audit.json', dict(groups=audits, backbone_before=frozen,
        backbone_after=signature(backbone), final_hashes=parameter_hash(model), frozen_unchanged=True))
    atomic(root/'TRAIN_COMPLETE.json', dict(checkpoint_sha256=sha(checkpoint), updates=total,
        unique_seen=len(seen), actual_disk_roundtrip=True, fixed_final=True,
        seconds=time.monotonic()-started, quality_success=False))


@torch.no_grad()
def evaluate(args, c, task, arm, items, inputs, root, device, budget, checkpoint=None):
    from docci_baseline_v14.model import decode
    from docci_text_continue_v11.metrics import metrics
    from utils.metrics import batch_image_metrics
    root.mkdir(parents=True, exist_ok=False); (root/'frames').mkdir(); (root/'images').mkdir()
    model, backbone, tok, br = backend.make(args, task, device, checkpoint)
    if arm == 'frozen':
        atomic(root/'REPLAY.json', replay(model, backbone, tok, items, task, read(inputs/'INPUTS.json'), device, budget))
    model.requires_grad_(False).eval(); before = parameter_hash(model); frozen = signature(backbone)
    metric = None; metric_info = {'status': 'NOT_MEASURED' if task == 'image' else 'NOT_APPLICABLE'}
    if task == 'image' and args.lpips_assets:
        from docci_text_guided_image_v20.perceptual import Perceptual, ASSETS
        metric = Perceptual(args.lpips_assets).to(device).eval()
        metric_info = dict(status='MEASURED_OFFLINE', assets_sha256=ASSETS, input_range='0..1')
    matrix = trials(c); wires = []
    for i in range(len(items)):
        budget(); wire = backend.encode(model, backend.source(items[i], task, device), task)
        x = wire[0].cpu().numpy().copy(); wires.append(x)
        with (root/'frames'/(items.entries[i]['example_id']+'.npz')).open('xb') as f:
            np.savez_compressed(f, tx=x)
    atomic(root/'identity.json', dict(task=task, arm=arm, config=c, trials=[list(x) for x in matrix],
        ids=[r['example_id'] for r in items.entries], geometry=geometry(task),
        checkpoint_sha256=sha(checkpoint) if checkpoint else PARENTS[task]['sha256'],
        trainable_parameters=sum(p.numel() for n, p in model.named_parameters() if 'bart.' not in n),
        LPIPS=metric_info, evaluation_split='qual_dev', qual_dev_reused=True,
        independent_final_test=False, semantic_facts='NOT_MEASURED', digital='NOT_RUN', FSO='NOT_RUN'))
    count = 0
    with (root/'rows.jsonl').open('x', encoding='utf-8') as log:
        for i in range(len(items)):
            item = items[i]; eid = item['example_id']
            for snr, repeat, control in matrix:
                budget(); donor = (i+1) % len(items) if control == 'shuffled' else i
                wire = torch.as_tensor(wires[donor][None], device=device)
                observed, stats = transmit(wire, task, snr,
                    seed(c['seed'], 'eval', task, items.entries[donor]['example_id'], snr, repeat))
                if control == 'zero': observed = torch.zeros_like(observed)
                rx = backend.receive(model, observed, task, stats)
                row = dict(example_id=eid, official_split='qual_dev', task=task, arm=arm, snr_db=snr,
                    repeat=repeat, control=control, donor_id=items.entries[donor]['example_id'],
                    stats=stats, receiver_wire_sha256=array_sha(observed.cpu().numpy()),
                    geometry=geometry(task), input_truncated=False, native=False)
                if task == 'text':
                    result = decode(backbone, rx, tok, 'greedy')
                    row.update(source_text=item['text'], source_text_sha256=item['text_sha256'],
                        source_tokens=item['source_tokens'], **result,
                        **metrics(item['text'], result['received_caption']))
                else:
                    target = torch.as_tensor(item['rgb'].copy(), device=device).permute(2, 0, 1)[None].float()/255.
                    rgb = backend.pixels(backbone, rx['latent']).clamp(0, 1)
                    l, t, r, b = map(int, item['rectangle'])
                    score = batch_image_metrics(rgb[:, :, t:b, l:r], target[:, :, t:b, l:r])
                    row.update(psnr=float(score['psnr'][0]), ssim=float(score['ssim'][0]),
                        LPIPS=None if metric is None else float(metric(rgb[:, :, t:b, l:r],
                                                                     target[:, :, t:b, l:r]).mean()),
                        evaluation_rectangle=[l, t, r, b], source_image_sha256=item['image_sha256'])
                    if i < c['save_images'] and repeat == 0:
                        label = 'clean' if snr is None else str(snr)
                        image = root/'images'/('{}__{}__{}.png'.format(eid, label, control))
                        Image.fromarray((rgb[0].permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)).save(image)
                        srcfile = root/'images'/(eid+'__source.png')
                        if not srcfile.exists(): Image.fromarray(item['rgb']).save(srcfile)
                        row.update(image_file=image.relative_to(root).as_posix(), image_sha256=sha(image),
                            source_file=srcfile.relative_to(root).as_posix(), source_view_sha256=sha(srcfile))
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); log.flush(); count += 1
            atomic(root.parent.parent/'progress.json', dict(stage='eval', task=task, arm=arm,
                examples_finished=i+1, total_examples=len(items), rows=count))
            print('V28 EVAL {} {} {}/{} ({} trials each)'.format(task, arm, i+1, len(items), len(matrix)), flush=True)
    if parameter_hash(model) != before or signature(backbone) != frozen:
        raise RuntimeError('Evaluation mutated model/backbone')
    atomic(root/'EVAL_COMPLETE.json', dict(rows=count, rows_sha256=sha(root/'rows.jsonl'),
        identity_sha256=sha(root/'identity.json'), unchanged=True))
