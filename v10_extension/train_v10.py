from __future__ import annotations
import argparse
import json
import math
import random
import time
from pathlib import Path
import torch
from v10 import VERSION
from v10.channels import PROFILES
from v10.data import build_loader
from v10.io import atomic_checkpoint, atomic_json, begin_run
from v10.model import build_model
from v10.runtime import add_shared_args, common_loss, evaluate, load_config, preflight, capture_rng, restore_rng

def build_parser():
    p = add_shared_args(argparse.ArgumentParser(description=VERSION, allow_abbrev=False))
    p.add_argument('--regime', choices=['mixed']+list(PROFILES), default='mixed')
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--train-samples', type=int, default=10000)
    p.add_argument('--val-samples', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-6)
    p.add_argument('--adapter-lr', type=float, default=2e-5)
    p.add_argument('--grad-accum', type=int, default=2)
    p.add_argument('--smoke-test', action='store_true')
    p.add_argument('--no-common-probe', action='store_true')
    p.add_argument('--no-anchor', action='store_true')
    return p

def main(args):
    from losses import MultimodalCommunicationLoss
    if min(args.epochs, args.train_samples, args.val_samples, args.grad_accum, args.lr, args.adapter_lr) <= 0:
        raise ValueError('Training arguments must be positive')
    if args.smoke_test:
        args.epochs, args.train_samples, args.val_samples = 1, 8, 4
    device = preflight(args)
    config = load_config(args)
    root = Path(__file__).resolve().parent
    directory, identity = begin_run(args.output_dir, args, config, root)
    started = time.monotonic()
    deadline = started+args.max_hours*3600
    history, optimizer_steps = [], 0
    try:
        print('Loading frozen backbones and exact initial checkpoint...', flush=True)
        from utils.checkpoint import load_checkpoint
        initial_metadata = load_checkpoint(args.checkpoint)
        if str(initial_metadata.get('version', '')).startswith('V10-'):
            if initial_metadata.get('split_sha256') != identity['split_sha256']:
                raise ValueError('Cannot continue a V10 checkpoint with a different split manifest')
        del initial_metadata
        model, report = build_model(config, device, args.checkpoint)
        atomic_json(directory/'checkpoint_load.json', report)
        train = build_loader(config, args.split_manifest, 'train', args.train_samples, args.seed)
        validation = build_loader(config, args.split_manifest, 'validation', args.val_samples, args.seed)
        base_params, adapter_params = [], []
        for name, p in model.named_parameters():
            if p.requires_grad:
                (adapter_params if name.startswith('base.channel.') else base_params).append(p)
        print('Parameters: total={} initially_trainable={}; decoder channel adapter={}'.format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            sum(p.numel() for p in model.base.channel.adapter.parameters())), flush=True)
        optimizer = torch.optim.AdamW([{'params': base_params, 'lr': args.lr},
                                      {'params': adapter_params, 'lr': args.adapter_lr}], weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda' and not args.no_amp)
        criterion = MultimodalCommunicationLoss(config['loss'])
        profiles = config['v10']['train_profiles']
        ema = {p: 1. for p in profiles}
        group_weights = {p: 1. for p in profiles}
        best = -math.inf
        for epoch in range(args.epochs):
            from utils.experiment import seed_everything
            seed_everything(args.seed+epoch*100003)
            stage = 'adapter' if epoch < config['v10']['adapter_epochs'] else 'joint'
            model.configure_stage(stage)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss, seen = 0., 0
            order = list(profiles)
            random.shuffle(order)
            for step, batch in enumerate(train):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Training budget exhausted; no COMPLETE marker will be written')
                profile = order[step % len(order)]
                snr = random.choice(config['v10']['train_snr'])
                images = batch['images'].to(device)
                # Normalize the FINAL partial accumulation group correctly.
                group_start = (step//args.grad_accum)*args.grad_accum
                divisor = min(args.grad_accum, len(train)-group_start)
                with torch.cuda.amp.autocast(enabled=device.type == 'cuda' and not args.no_amp):
                    out = model(images, batch['texts'], profile, snr)
                    loss, parts = criterion(out, images)
                    weighted_loss = loss*group_weights[profile]/divisor
                    latent_target = out['vae_latents'].detach()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss in '+profile)
                scaler.scale(weighted_loss).backward()
                value = float(loss.detach())
                del out, weighted_loss, loss, parts
                ema[profile] = .9*ema[profile]+.1*value
                probe_value, anchor_value = None, None
                cfg = config['v10']
                if (step+1) % cfg['common_probe_every'] == 0 and cfg['common_probe_weight'] > 0:
                    with torch.cuda.amp.autocast(enabled=device.type == 'cuda' and not args.no_amp):
                        probe_profile = profiles[((step+1)//cfg['common_probe_every']-1+epoch) % len(profiles)]
                        probe = model(images, batch['texts'], probe_profile, snr, semantic_ablation='common_only',
                                      common_task_only=True, vae_latents_override=latent_target)
                        auxiliary = common_loss(probe)*cfg['common_probe_weight']
                    probe_value = float(auxiliary.detach())
                    scaler.scale(auxiliary/divisor).backward()
                    del probe, auxiliary
                if (step+1) % cfg['anchor_every'] == 0 and cfg['anchor_weight'] > 0:
                    with torch.cuda.amp.autocast(enabled=device.type == 'cuda' and not args.no_amp):
                        anchor = model(images, batch['texts'], 'awgn', 20., vae_latents_override=latent_target)
                        replay, _ = criterion(anchor, images)
                    anchor_value = float(replay.detach())
                    scaler.scale(replay*cfg['anchor_weight']/divisor).backward()
                    del anchor, replay
                if (step+1) % args.grad_accum == 0 or step+1 == len(train):
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    if not torch.isfinite(norm):
                        raise FloatingPointError('Nonfinite gradient; abort, not a silent skipped update')
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                epoch_loss += value*len(images)
                seen += len(images)
                if step == 0 or (step+1) % 20 == 0 or step+1 == len(train):
                    print('epoch={}/{} stage={} step={}/{} channel={} snr={} loss={:.5f} common={} replay={} elapsed={:.1f}s'.format(
                        epoch+1, args.epochs, stage, step+1, len(train), profile, snr, value,
                        probe_value, anchor_value, time.monotonic()-started), flush=True)
            rng = capture_rng(device)
            metrics = evaluate(model, validation, device, cfg['validation_profiles'], cfg['validation_snr'],
                               args.seed, not args.no_amp, deadline)
            restore_rng(rng, device)
            scores = [v['score'] for v in metrics.values()]
            score = .5*sum(scores)/len(scores)+.5*min(scores)
            hardness = torch.tensor([ema[p] for p in profiles])
            q = torch.softmax(hardness-hardness.mean(), dim=0)
            mix = cfg['worst_group_weight']
            group_weights = {p: (1-mix)+mix*len(profiles)*float(q[i]) for i,p in enumerate(profiles)}
            row = {'epoch': epoch+1, 'stage': stage, 'loss': epoch_loss/seen, 'samples': seen,
                   'validation': metrics, 'selection_score': score, 'group_weights_next': group_weights,
                   'elapsed_seconds': time.monotonic()-started, 'optimizer_steps': optimizer_steps}
            history.append(row)
            payload = {'version': VERSION, 'model': model.state_dict(), 'config': config,
                       'epoch': epoch+1, 'split_sha256': identity['split_sha256'], 'metrics': row,
                       'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict()}
            atomic_checkpoint(directory/'last.pth', payload)
            if score > best:
                best = score
                atomic_checkpoint(directory/'best.pth', payload)
            atomic_json(directory/'history.json', history)
            print('Validation score={:.5f}, best={:.5f}; saved epoch {}'.format(score, best, epoch+1), flush=True)
        if len(history) != args.epochs or optimizer_steps <= 0:
            raise RuntimeError('Incomplete training')
        atomic_json(directory/'COMPLETE.json', {'version': VERSION, 'epochs': len(history),
                    'optimizer_steps': optimizer_steps, 'wall_seconds': time.monotonic()-started,
                    'smoke_only': args.smoke_test})
        identity['status'] = 'complete_smoke' if args.smoke_test else 'complete'
    except BaseException as exc:
        identity.update(status='failed_or_interrupted', exception=type(exc).__name__, error=str(exc))
        atomic_json(directory/'run_identity.json', identity)
        raise
    atomic_json(directory/'run_identity.json', identity)

if __name__ == '__main__':
    main(build_parser().parse_args())
