"""Frozen A delta + existing V10 base, validation only. NO retraining or test use."""
import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from unittest.mock import patch
EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT))
from optical_link_v1.config import load, TEMPLATE
from optical_link_v1 import VERSION


def main():
    import torch
    import v10.model as module
    from v10.runtime import add_shared_args, preflight, load_config
    from v10.io import begin_run, atomic_json
    from v10.data import build_loader, sha256
    from v10.model import build_model
    from utils.checkpoint import load_checkpoint
    from utils.metrics import batch_image_metrics, token_accuracy, sentence_bleu2
    from business_link.metrics import text_metrics
    from business_link.engine import identity_seed
    from optical_link_v1.plots import constellation_svg
    from eval_v10 import save_image
    from token_guidance.model import Guidance, TokenFrame
    from optical_link_v1.model import OpticalFrame
    from optical_link_v1.report import write_report

    p = add_shared_args(argparse.ArgumentParser(description=__doc__, allow_abbrev=False))
    p.add_argument('--a-delta', required=True)
    p.add_argument('--optical-config', default=str(TEMPLATE))
    p.add_argument('--val-samples', type=int, default=8)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--save-images', type=int, default=8)
    args = p.parse_args()
    if not 1 <= args.val_samples <= 500 or not 1 <= args.repeats <= 8 or args.save_images < 0:
        raise ValueError('Bounded validation only: 1..500 images, 1..8 channel repeats')
    args.batch_size, args.num_workers, args.no_amp, args.fixed_mode, args.csi, args.blocks = 1, 0, True, 0, 'pilot', 1
    specification, cases, config_hash = load(args.optical_config)
    device = preflight(args)
    config = load_config(args)
    base = load_checkpoint(args.checkpoint)
    delta = load_checkpoint(args.a_delta)
    split_hash, base_hash = sha256(args.split_manifest), sha256(args.checkpoint)
    if (not str(base.get('version','')).startswith('V10-') or base.get('split_sha256') != split_hash
            or delta.get('version') != 'V10-token-guidance-quick-v1' or delta.get('variant') != 'A'
            or delta.get('base_checkpoint_sha256') != base_hash or delta.get('split_sha256') != split_hash):
        raise ValueError('Require exact A delta, its original V10 base and matching frozen split')
    config['v10'] = copy.deepcopy(base['config']['v10'])
    config['v10'].update(blocks=1, csi='pilot', fixed_mode=0)
    del base
    directory, identity = begin_run(args.output_dir, args, config, EXT)
    started, rows = time.monotonic(), []
    identity.update(version=VERSION, a_delta_sha256=sha256(args.a_delta), optical_config_sha256=config_hash,
        optical_config=specification, scope='frozen_A_validation_no_training',
        extension_source_sha256={str(p.relative_to(EXT)):sha256(p) for folder in ('optical_link_v1','token_guidance','business_link','physical_link') for p in sorted((EXT/folder).iterdir()) if p.suffix in ('.py','.json','.md')})
    atomic_json(directory/'run_identity.json', identity)
    def budget():
        if time.monotonic()-started >= args.max_hours*3600:
            raise TimeoutError('Wall-time cap reached; preserve partial results, no COMPLETE')
    try:
        print('Loading frozen A: no training; SI optical link, no preset SNR.', flush=True)
        model, report = build_model(config, device, args.checkpoint)
        if report != {'v10_exact':True}:
            raise ValueError('Exact original base loading required')
        guide = Guidance(model)
        wanted = {name for name, _ in guide.trainable('A')}
        if set(delta['delta']) != wanted:
            raise ValueError('A delta has missing/unexpected parameter names')
        combined = dict(model.state_dict(), **delta['delta'])
        model.load_state_dict(combined, strict=True)
        del combined, delta
        model.requires_grad_(False)
        model.eval()
        guide.set('A')
        atomic_json(directory/'checkpoint_load.json', dict(base_exact=True, a_delta_exact=True,
            base_checkpoint_sha256=base_hash, a_delta_sha256=identity['a_delta_sha256'], retraining=False))
        loader = build_loader(config, args.split_manifest, 'validation', args.val_samples, args.seed)
        loader.dataset.caption_mode = 'first'
        image_dir, constellation_dir = directory/'images', directory/'constellations'
        image_dir.mkdir()
        constellation_dir.mkdir()
        expected = args.val_samples*(1+len(cases)*args.repeats)
        atomic_json(directory/'design.json', dict(version=VERSION, variant='A', split='validation',
            optical_config=specification, effective_cases=cases, expected_rows=expected,
            val_samples=args.val_samples, repeats=args.repeats,
            conditions=['quantized_clean']+list(cases), semantic_reals=4608, raw_utf8_wire_bytes=0,
            repeats_are_channel_draws_not_training_seeds=True,
            nominal_snr_db=None, parameter_status=specification['parameter_status']))
        with (directory/'samples.jsonl').open('x',encoding='utf-8') as stream, (directory/'packets.jsonl').open('x',encoding='utf-8') as packets:
            seen_ids = []
            with torch.no_grad():
                for index, batch in enumerate(loader):
                    budget()
                    image_id, caption = int(batch['image_ids'][0]), batch['texts'][0]
                    seen_ids.append(image_id)
                    images = batch['images'].to(device)
                    latent = model.base.vae_wrapper.encode_images(images)
                    conditions = [('quantized_clean',0,None)]+[(name,repeat,c) for name,c in cases.items() for repeat in range(args.repeats)]
                    for case, repeat, optical in conditions:
                        budget()
                        field_seed = identity_seed(args.seed,image_id,repeat,'optical_field')
                        noise_seed = identity_seed(args.seed,image_id,repeat,'optical_noise')
                        context = dict(config=optical,field_seed=field_seed,noise_seed=noise_seed)
                        if optical is None:
                            context['clean'] = True
                            factory = lambda *a,**kw: TokenFrame(*a,context=context,**kw)
                        else:
                            factory = lambda *a,**kw: OpticalFrame(*a,context=context,**kw)
                        before = time.monotonic()
                        if device.type == 'cuda':
                            torch.cuda.synchronize(device)
                        with patch.object(module,'LinkFrame',factory):
                            out = model(images,[caption],'awgn',40. if optical is None else 0.,vae_latents_override=latent)
                        if device.type == 'cuda':
                            torch.cuda.synchronize(device)
                        wall = time.monotonic()-before
                        frame = model.last_frame
                        metric = batch_image_metrics(out['image'],images)
                        recovered_caption = model.base.decode_text_logits(out['text_logits'])[0]
                        correct,count = token_accuracy(out['text_logits'],out['text_input_ids'],out['text_attention_mask'])
                        row = dict(frame.last_stats, case=case, repeat=repeat, image_id=image_id, variant='A',
                            nominal_snr_db=None, field_seed=field_seed, noise_seed=noise_seed,
                            physical_config_sha256=config_hash, parameter_status=specification['parameter_status'],
                            psnr=float(metric['psnr'][0]), ssim=float(metric['ssim'][0]),
                            token_acc=correct/max(count,1), token_correct=correct, token_count=count,
                            bleu2=sentence_bleu2(caption,recovered_caption),
                            source_caption_evaluation_only=caption, received_neural_caption=recovered_caption,
                            source_rgb8_bytes=batch['images'].numel(), source_utf8_bytes=len(caption.encode()),
                            source_caption_sha256=hashlib.sha256(caption.encode()).hexdigest(),
                            neural_and_cpu_transport_wall_seconds=wall, **text_metrics(caption,recovered_caption))
                        for key in ('psnr','ssim','token_acc','bleu2'):
                            if not math.isfinite(row[key]):
                                raise FloatingPointError('Nonfinite semantic metric: '+key)
                        if index < args.save_images:
                            source = image_dir/('{}__source.png'.format(image_id))
                            if not source.exists():
                                save_image(batch['images'][0],source)
                            suffix = '{}__{}__repeat{}'.format(image_id,case,repeat)
                            row['source_image_file'] = 'images/'+source.name
                            row['image_file'] = 'images/'+suffix+'.png'
                            save_image(out['image'][0],directory/row['image_file'])
                            if frame.transport is not None:
                                row['constellation_file'] = 'constellations/'+suffix+'.svg'
                                constellation_svg(frame.transport.constellations,directory/row['constellation_file'])
                        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                        stream.flush()
                        if frame.transport is not None:
                            for event in frame.transport.events:
                                packets.write(json.dumps(dict(event,case=case,repeat=repeat,image_id=image_id),allow_nan=False)+'\n')
                            packets.flush()
                        rows.append(row)
                        print('EVAL {}/{} id={} {} PSNR={:.2f}dB CRC={} elapsed={:.1f}s'.format(len(rows),expected,image_id,case,row['psnr'],row['feature_object_crc_accepted'],time.monotonic()-started),flush=True)
                        del out
            if len(rows) != expected or len(seen_ids) != args.val_samples or len(set(seen_ids)) != len(seen_ids):
                raise RuntimeError('Incomplete or duplicate validation matrix')
        atomic_json(directory/'validation_ids.json',seen_ids)
        write_report(directory,rows,'冻结A权重的图文validation；无训练、无测试集、无原文旁路')
        identity['status'] = 'complete'
        atomic_json(directory/'run_identity.json',identity)
        atomic_json(directory/'COMPLETE.json',dict(version=VERSION,rows=len(rows),expected_rows=expected,
            wall_seconds=time.monotonic()-started, trained=False, hardware_calibrated=False))
        print('COMPLETE: '+str(directory/'REPORT.md'),flush=True)
    except BaseException as exc:
        identity.update(status='failed_or_interrupted',error=str(exc),exception=type(exc).__name__)
        atomic_json(directory/'run_identity.json',identity)
        raise


if __name__ == '__main__':
    main()
