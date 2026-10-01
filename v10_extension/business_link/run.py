"""Fixed image/text task evaluation; never trains or changes a checkpoint."""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

EXT = Path(__file__).resolve().parents[1]
PROJECT = EXT.parent
sys.path.insert(0, str(EXT))
from business_link.engine import POLICIES, identity_seed
from business_link.metrics import text_metrics


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--checkpoint', default=str(PROJECT/'experiment_results/v10_formal_paper_v10_02_noamp/train_mixed_seed42/best.pth'))
    p.add_argument('--config', default=str(EXT/'configs/v10_multichannel.json'))
    p.add_argument('--split-manifest', default=str(EXT/'manifests/ln_v10_splits.json'))
    p.add_argument('--data-root', default=str(PROJECT/'COCO2017'))
    p.add_argument('--vae-path', default=str(PROJECT/'pretrained/sd-vae-ft-mse'))
    p.add_argument('--text-model-path', default=str(PROJECT/'pretrained/clip-vit-base-patch32'))
    p.add_argument('--samples', type=int, default=16)
    p.add_argument('--business-bytes', type=int, help='Upper bound on RGB8 model-input bytes + UTF8; select whole image/text tasks, never truncate')
    p.add_argument('--reuse-business-manifest', help='Reuse and verify the exact frozen validation tasks from another V3 output')
    p.add_argument('--profiles', nargs='+', choices=('awgn', 'fso_weak', 'fso_moderate', 'fso_strong'), default=['fso_moderate', 'fso_strong'])
    p.add_argument('--snr-list', nargs='+', type=float, default=[0, 10, 20])
    p.add_argument('--policies', nargs='+', choices=POLICIES, default=['analog', '16qam_none', '16qam_conv', '16qam_repeat2'])
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--receivers', nargs='+', choices=('baseline', 'received_text'), default=['baseline', 'received_text'])
    p.add_argument('--fusion-weight', type=float, default=.25)
    p.add_argument('--mode', choices=('reconstruction', 'hybrid', 'generation', 'adaptive'), default='reconstruction')
    p.add_argument('--fs', type=float, default=64e9)
    p.add_argument('--sps', type=int, default=4)
    p.add_argument('--rolloff', type=float, default=.1)
    p.add_argument('--span', type=int, default=32)
    p.add_argument('--qbits', type=int, choices=(4, 6, 8), default=8)
    p.add_argument('--quantizer-limit', type=float, default=4.)
    p.add_argument('--dac-bits', type=int, choices=(8, 10, 12, 16), default=8)
    p.add_argument('--bandwidth', type=float, help='Assumed analog frontend 3 dB bandwidth, NOT measured device response')
    p.add_argument('--packet-bytes', type=int, default=512)
    p.add_argument('--guard-samples', type=int, default=256)
    p.add_argument('--max-attempts', type=int, default=1)
    p.add_argument('--ideal-ack-arq', action='store_true')
    p.add_argument('--feedback-rate', type=float, default=1e6)
    p.add_argument('--distance-m', type=float, default=1000.)
    p.add_argument('--task-psnr-threshold', type=float, default=20., help='Predeclared operational threshold, NOT a universal perceptual criterion')
    p.add_argument('--seed', type=int, default=42, help='One reproducible channel experiment, NOT multiple training seeds')
    p.add_argument('--save-images', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-hours', type=float, default=2.)
    p.add_argument('--min-free-gb', type=float, default=4.)
    p.add_argument('--min-free-disk-gb', type=float, default=1.)
    return p


def freeze_business(dataset, args, split_sha):
    import numpy as np
    wanted, prior = None, None
    if args.reuse_business_manifest:
        prior = json.loads(Path(args.reuse_business_manifest).read_text(encoding='utf-8'))
        if prior.get('schema') != 'fixed-business-v3' or prior['split_sha256'] != split_sha or prior['split'] != 'validation':
            raise ValueError('Frozen task belongs to another split')
        wanted = [r['image_id'] for r in prior['tasks']]
        if len(wanted) != len(set(wanted)) or not wanted:
            raise ValueError('Invalid frozen task IDs')
        by_id = {r['image_id']: i for i, r in enumerate(dataset.records)}
        indices = [by_id[i] for i in wanted]  # reject missing IDs, no substitution
    else:
        indices = range(min(args.samples, len(dataset)))
    items, tasks, total = [], [], 0
    for index in indices:
        item = dataset[index]
        rgb = np.rint(item['image'].numpy()*255).astype(np.uint8)
        text = item['text'].encode('utf-8')
        size = rgb.size+len(text)
        if wanted is None and args.business_bytes is not None and total+size > args.business_bytes:
            break
        tasks.append({'image_id': item['image_id'], 'image_shape_chw': list(rgb.shape),
            'rgb8_bytes': rgb.size, 'text_utf8_bytes': len(text), 'business_bytes': size,
            'input_sha256': hashlib.sha256(rgb.tobytes()+text).hexdigest(),
            'original_jpeg_file_bytes_reference_only': Path(item['path']).stat().st_size})
        items.append(item)
        total += size
    if not tasks or (wanted is None and args.business_bytes is None and len(tasks) != args.samples):
        raise ValueError('Empty/incomplete fixed business task selection')
    document = {'schema': 'fixed-business-v3', 'split': 'validation', 'split_sha256': split_sha,
        'reference': 'center-cropped model-input RGB8 + normalized first caption UTF8',
        'image_count': len(tasks), 'text_count': len(tasks), 'business_bytes': total,
        'requested_upper_byte_budget': args.business_bytes, 'tasks': tasks}
    if prior is not None and document['tasks'] != prior['tasks']:
        raise ValueError('Frozen business input bytes/hashes changed')
    if prior is not None and (prior['business_bytes'] != total or prior['image_count'] != len(tasks) or prior['text_count'] != len(tasks)):
        raise ValueError('Inconsistent frozen business volume/count')
    return items, document


def main(args):
    import numpy as np
    import torch
    import v10.model as model_module
    from v10.model import build_model
    from v10.channels import PROFILES
    from v10.data import build_loader, sha256
    from v10.io import begin_run, atomic_json
    from v10.runtime import load_config, preflight
    from v10.protocol import MODE_NAMES
    from utils.checkpoint import load_checkpoint
    from utils.experiment import seed_everything
    from utils.metrics import batch_image_metrics, sentence_bleu2
    from eval_v10 import save_image
    from business_link.adapter import BusinessFrame, receiver_text_hook
    from business_link.plots import constellation_svg
    from business_link.report import make_report, write_report
    if min(args.samples, args.repeats) <= 0 or args.save_images < 0 or not 0 <= args.fusion_weight <= 1:
        raise ValueError('Invalid task/repeat/fusion budget')
    if args.business_bytes is not None and args.business_bytes <= 0:
        raise ValueError('Invalid business byte budget')
    for name in ('profiles', 'snr_list', 'policies', 'receivers'):
        values = getattr(args, name)
        if not values or len(values) != len(set(values)):
            raise ValueError('Duplicate/empty experimental axis: '+name)
    if not all(np.isfinite(args.snr_list)) or not np.isfinite(args.task_psnr_threshold):
        raise ValueError('Nonfinite SNR/quality threshold')
    if not np.isfinite(args.quantizer_limit) or args.quantizer_limit <= 0:
        raise ValueError('Quantizer limit must be finite and positive')
    args.batch_size, args.num_workers, args.no_amp = 1, 0, True
    args.fixed_mode = None if args.mode == 'adaptive' else MODE_NAMES.index(args.mode)
    device = preflight(args)
    config = load_config(args)
    metadata = load_checkpoint(args.checkpoint)
    if not str(metadata.get('version', '')).startswith('V10-') or metadata.get('split_sha256') != sha256(args.split_manifest):
        raise ValueError('Need exact V10 checkpoint with matching split SHA256')
    config['v10'] = copy.deepcopy(metadata['config']['v10'])
    config['v10'].update(blocks=1, csi='pilot', fixed_mode=args.fixed_mode)
    del metadata
    settings = {k: getattr(args, k) for k in ('fs', 'sps', 'rolloff', 'span', 'qbits', 'quantizer_limit',
        'dac_bits', 'bandwidth', 'packet_bytes', 'guard_samples', 'max_attempts', 'ideal_ack_arq', 'feedback_rate', 'distance_m')}
    # Validate physical settings BEFORE allocating the model/output.
    from business_link.engine import Transport
    Transport(settings, 1, .01, 0)
    directory, identity = begin_run(args.output_dir, args, config, EXT)
    identity.update(experiment='fixed-business-v3', evaluation_only_protocol_change=True,
                    measured_hardware_throughput=False, model_retrained=False,
                    physical_source_sha256={str(p.relative_to(EXT)): sha256(p) for p in
                        sorted(list((EXT/'business_link').glob('*.py'))+list((EXT/'physical_link').glob('*.py')))})
    atomic_json(directory/'run_identity.json', identity)
    started, rows = time.monotonic(), []
    try:
        model, loaded = build_model(config, device, args.checkpoint)
        model.eval()
        atomic_json(directory/'checkpoint_load.json', loaded)
        refiner = model.base.decoder_text_to_image_refiner
        if 'received_text' in args.receivers and refiner is None:
            raise ValueError('Checkpoint has no decoder text-to-image refiner; do not silently omit fusion')
        loader = build_loader(config, args.split_manifest, 'validation', None, args.seed, workers=0)
        items, business = freeze_business(loader.dataset, args, identity['split_sha256'])
        atomic_json(directory/'business_manifest.json', business)
        design = {'schema': 'business-design-v3', 'image_ids': [t['image_id'] for t in business['tasks']],
            'profiles': args.profiles, 'snr_list': args.snr_list, 'policies': args.policies,
            'receivers': args.receivers, 'repeats': args.repeats, 'settings': settings,
            'task_psnr_threshold': args.task_psnr_threshold,
            'business_manifest_sha256': sha256(directory/'business_manifest.json'),
            'business_bytes': business['business_bytes'], 'mode': args.mode,
            'checkpoint_sha256': identity['checkpoint_sha256'], 'split_sha256': identity['split_sha256']}
        atomic_json(directory/'design.json', design)
        for folder in ('images', 'constellations'):
            (directory/folder).mkdir()
        tokenizer = model.base.text_encoder.tokenizer
        with torch.no_grad(), (directory/'samples.jsonl').open('x', encoding='utf-8') as stream, (directory/'packets.jsonl').open('x', encoding='utf-8') as packet_stream:
            for index, (item, task) in enumerate(zip(items, business['tasks'])):
                image_id = item['image_id']
                images = item['image'][None].to(device)
                seed_everything(identity_seed(args.seed, image_id, 'vae') % (2**31))
                latent = model.base.vae_wrapper.encode_images(images)
                save_this = index < args.save_images
                if save_this:
                    save_image(images[0], directory/'images'/('{}__source.png'.format(image_id)))
                for profile, snr, repeat, policy, receiver in itertools.product(
                        args.profiles, args.snr_list, range(args.repeats), args.policies, args.receivers):
                    if time.monotonic()-started >= args.max_hours*3600:
                        raise TimeoutError('V3 evaluation time budget exhausted; preserve partial output')
                    context = {'policy': policy, 'source_text': item['text'],
                        'field_seed': identity_seed(args.seed, image_id, profile, repeat, 'field'),
                        'noise_seed': identity_seed(args.seed, image_id, profile, snr, repeat, 'noise')}
                    def frame_factory(*values, **kwargs):
                        return BusinessFrame(*values, settings=settings, context=context, **kwargs)
                    weight = args.fusion_weight if receiver == 'received_text' else 0.
                    hook = receiver_text_hook(model, refiner.forward, weight) if refiner is not None else None
                    seed_everything(identity_seed(args.seed, image_id, profile, snr, repeat, 'model') % (2**31))
                    before = time.monotonic()
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    with patch.object(model_module, 'LinkFrame', frame_factory):
                        if hook is not None:
                            with patch.object(refiner, 'forward', hook):
                                out = model(images, [item['text']], profile, snr, 'full', latent)
                        else:
                            out = model(images, [item['text']], profile, snr, 'full', latent)
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    inference_wall = time.monotonic()-before
                    frame = model.last_frame
                    metric = batch_image_metrics(out['image'], images)
                    neural_caption = tokenizer.batch_decode(out['text_logits'].argmax(-1), skip_special_tokens=True)[0]
                    psnr, ssim = float(metric['psnr'][0]), float(metric['ssim'][0])
                    link = frame.last_stats
                    delivered = (None if policy == 'analog' else bool(link['feature_object_crc_accepted'] and link['text_exact_match_diagnostic']))
                    quality_ok = bool(link['text_exact_match_diagnostic'] and (policy == 'analog' or delivered) and psnr >= args.task_psnr_threshold)
                    row = dict(task, image_id=image_id, profile=profile, nominal_snr_db=snr, repeat=repeat,
                        policy=policy, receiver=receiver, psnr=psnr, ssim=ssim,
                        source_caption_evaluation_only=item['text'], received_digital_text=frame.received_text,
                        neural_received_caption=neural_caption, neural_bleu2=sentence_bleu2(item['text'], neural_caption),
                        fusion_used=frame.fusion_used, fusion_weight=weight,
                        requested_mode=MODE_NAMES[int(frame.requested_mode[0])],
                        sender_mode=MODE_NAMES[int(frame.tx_mode[0])], receiver_mode=MODE_NAMES[int(frame.rx_mode[0])],
                        input_delivery_complete=delivered, task_quality_success=quality_ok,
                        inference_plus_channel_simulation_wall_seconds=inference_wall,
                        **text_metrics(item['text'], frame.received_text), **link)
                    row['delivered_business_goodput_bps'] = (None if delivered is None else task['business_bytes']*8/row['sequential_link_seconds']*int(delivered))
                    row['quality_business_goodput_bps'] = task['business_bytes']*8/row['sequential_link_seconds']*int(quality_ok)
                    if link['feature_object_wire_bytes'] is not None:
                        row['source_payload_fraction'] = (link['feature_object_wire_bytes']+task['text_utf8_bytes'])/task['business_bytes']
                    else:
                        row['source_payload_fraction'] = None  # analog symbols are not byte compression
                    suffix = '{}__{}__snr{}__r{}__{}__{}'.format(image_id, profile, snr, repeat, policy, receiver)
                    if save_this:
                        row['image_file'] = 'images/'+suffix+'.png'
                        save_image(out['image'][0], directory/row['image_file'])
                        constellation_svg(frame.transport.constellations, directory/'constellations'/(suffix+'.svg'))
                        atomic_json(directory/'constellations'/(suffix+'.json'), frame.transport.constellations)
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                    stream.flush()
                    if receiver == args.receivers[0]:
                        for event in frame.transport.events:
                            packet_stream.write(json.dumps(dict(event, image_id=image_id, profile=profile,
                                nominal_snr_db=snr, repeat=repeat, experiment_policy=policy), allow_nan=False)+'\n')
                        packet_stream.flush()
                    rows.append(row)
                    if len(rows) == 1 or len(rows) % 10 == 0:
                        print('rows={} image={}/{} {} {}dB {} {} elapsed={:.1f}s'.format(
                            len(rows), index+1, len(items), profile, snr, policy, receiver, time.monotonic()-started), flush=True)
                print('fixed business images={}/{} rows={} elapsed={:.1f}s'.format(index+1, len(items), len(rows), time.monotonic()-started), flush=True)
        summary = make_report(rows, design)  # strict independent lattice/pairing check
        write_report(directory, summary)
        atomic_json(directory/'COMPLETE.json', {'rows': len(rows), 'expected_rows': summary['expected_rows'],
            'wall_seconds': time.monotonic()-started, 'experiment_finished_not_all_tasks_successful': True})
        identity['status'] = 'complete'
    except BaseException as exc:
        identity.update(status='failed_or_interrupted', exception=type(exc).__name__, error=str(exc))
        atomic_json(directory/'run_identity.json', identity)
        raise
    atomic_json(directory/'run_identity.json', identity)


if __name__ == '__main__':
    main(parser().parse_args())
