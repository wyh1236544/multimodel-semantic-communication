from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from v10 import VERSION
from v10.channels import PROFILES
from v10.data import build_loader, sha256
from v10.io import atomic_json, begin_run
from v10.model import build_model
from v10.protocol import MODE_NAMES
from v10.runtime import add_shared_args, load_config, preflight

def build_parser():
    p = add_shared_args(argparse.ArgumentParser(description='Physical multi-channel evaluation, including real Common-only payload', allow_abbrev=False))
    p.add_argument('--profiles', nargs='+', choices=list(PROFILES), default=list(PROFILES))
    p.add_argument('--snr-list', nargs='+', type=float, default=[-5, 0, 5, 10, 20])
    p.add_argument('--split', choices=('validation', 'calibration', 'test'), default='validation')
    p.add_argument('--samples', type=int, default=100)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--ablation', choices=('full', 'no_common', 'common_only', 'common_only_random'), default='full')
    p.add_argument('--save-images', type=int, default=8, help='Maximum unique images; all channel conditions for these IDs')
    p.add_argument('--allow-settings-change', action='store_true', help='Explicit diagnostic ablation when evaluation differs from trained settings')
    return p

def sample_seed(seed, image_id, profile, snr, repeat):
    key = '{}|{}|{}|{}|{}'.format(seed, image_id, profile, float(snr), repeat)
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (2**31)

def save_image(tensor, path):
    import numpy as np
    from PIL import Image
    a = tensor.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    Image.fromarray(np.rint(a*255).astype('uint8')).save(path)

@torch.no_grad()
def main(args):
    from utils.checkpoint import load_checkpoint
    from utils.experiment import seed_everything
    from utils.metrics import batch_image_metrics, token_accuracy, sentence_bleu2
    if min(args.samples, args.repeats) <= 0 or args.save_images < 0:
        raise ValueError('Invalid sample/repeat/image budget')
    device = preflight(args)
    config = load_config(args)
    metadata = load_checkpoint(args.checkpoint)
    is_v10 = str(metadata.get('version', '')).startswith('V10-')
    if is_v10:
        if metadata.get('split_sha256') != sha256(args.split_manifest):
            raise ValueError('Checkpoint and evaluation split manifests differ')
        trained = metadata['config']['v10']
        keys = ('blocks', 'pilots', 'control_length', 'control_threshold', 'feedback_snr_db',
                'csi', 'equalizer', 'uep', 'adapter', 'interleave', 'early_fusion', 'fixed_mode', 'profile_overrides',
                'generation_threshold_db', 'reconstruction_threshold_db')
        changed = [k for k in keys if trained.get(k) != config['v10'].get(k)]
        if changed and not args.allow_settings_change:
            raise ValueError('Settings differ from trained model: {}. Match config or label an explicit ablation.'.format(changed))
    del metadata
    directory, identity = begin_run(args.output_dir, args, config, Path(__file__).resolve().parent)
    started = time.monotonic()
    rows, groups = [], {}
    embeddings = {}
    image_dir = directory/'images'
    image_dir.mkdir()
    try:
        model, report = build_model(config, device, args.checkpoint)
        model.eval()
        atomic_json(directory/'checkpoint_load.json', report)
        loader = build_loader(config, args.split_manifest, args.split, args.samples, args.seed)
        tokenizer = model.base.text_encoder.tokenizer
        seen_ids = []
        with (directory/'samples.jsonl').open('x', encoding='utf-8') as stream:
            for batch in loader:
                # Per-image RNG identity makes batching/order changes irrelevant.
                for b, image_id in enumerate(batch['image_ids']):
                    images = batch['images'][b:b+1].to(device)
                    texts = [batch['texts'][b]]
                    seen_ids.append(image_id)
                    seed_everything(sample_seed(args.seed, image_id, 'vae', 0, 0))
                    latent = model.base.vae_wrapper.encode_images(images)
                    save_this = len(seen_ids) <= args.save_images
                    if save_this:
                        save_image(images[0], image_dir/('{}__source.png'.format(image_id)))
                    for profile in args.profiles:
                        for snr in args.snr_list:
                            for repeat in range(args.repeats):
                                if time.monotonic()-started >= args.max_hours*3600:
                                    raise TimeoutError('Evaluation time budget exhausted')
                                seed_everything(sample_seed(args.seed, image_id, profile, snr, repeat))
                                with torch.cuda.amp.autocast(enabled=device.type == 'cuda' and not args.no_amp):
                                    out = model(images, texts, profile, snr, args.ablation, latent)
                                received = tokenizer.batch_decode(out['text_logits'].argmax(-1), skip_special_tokens=True)[0]
                                common_caption = tokenizer.batch_decode(out['common_base_text_logits'].argmax(-1), skip_special_tokens=True)[0]
                                metric = batch_image_metrics(out['image'], images)
                                correct, count = token_accuracy(out['text_logits'], out['text_input_ids'], out['text_attention_mask'])
                                stats = {k: v.item() if isinstance(v, torch.Tensor) else v for k,v in out['link_stats'].items()}
                                frame = model.last_frame
                                row = {'image_id': image_id, 'profile': profile, 'nominal_snr_db': snr, 'repeat': repeat,
                                       'ablation': args.ablation, 'psnr': metric['psnr'].item(), 'ssim': metric['ssim'].item(),
                                       'token_acc': correct/max(count, 1), 'bleu2': sentence_bleu2(texts[0], received),
                                       'common_caption_bleu2': sentence_bleu2(texts[0], common_caption),
                                       'common_cosine': F.cosine_similarity(out['common_receiver_semantic'], out['common_text_semantic_target']).item(),
                                       'requested_mode': MODE_NAMES[int(frame.requested_mode[0])],
                                       'sender_mode': MODE_NAMES[int(frame.tx_mode[0])], 'receiver_mode': MODE_NAMES[int(frame.rx_mode[0])],
                                       'source_caption_evaluation_only': texts[0], 'received_caption': received,
                                       'common_caption': common_caption, **stats}
                                if save_this:
                                    filename = '{}__{}__snr{}__r{}__{}.png'.format(image_id, profile, snr, repeat, args.ablation)
                                    save_image(out['image'][0], image_dir/filename)
                                    row['image_file'] = 'images/'+filename
                                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                                stream.flush()
                                rows.append(row)
                                key = profile+'/'+str(snr)
                                groups.setdefault(key, []).append(row)
                                ekey = key+'/'+str(repeat)
                                embeddings.setdefault(ekey, []).append((image_id, out['common_receiver_semantic'][0].float().cpu(), out['common_text_semantic_target'][0].float().cpu()))
                    if len(seen_ids) == 1 or len(seen_ids) % 10 == 0:
                        print('images={}/{} rows={} elapsed={:.1f}s'.format(len(seen_ids), args.samples, len(rows), time.monotonic()-started), flush=True)
        expected = args.samples*len(args.profiles)*len(args.snr_list)*args.repeats
        if len(rows) != expected or len(set(seen_ids)) != args.samples:
            raise RuntimeError('Incomplete/duplicate evaluation sample coverage')
        summary = {}
        fields = ('psnr', 'ssim', 'token_acc', 'bleu2', 'common_caption_bleu2', 'common_cosine',
                  'header_rejected', 'header_undetected_error', 'feedback_payload_error', 'estimated_outage', 'estimated_snr_db', 'tx_average_power')
        for key, values in groups.items():
            summary[key] = {field: sum(float(v[field]) for v in values)/len(values) for field in fields}
            summary[key].update(rows=len(values), unique_images=len({v['image_id'] for v in values}),
                                physical_unit=values[0]['physical_unit'], forward_uses=values[0]['forward_uses'])
            recalls = []
            for repeat in range(args.repeats):
                items = embeddings[key+'/'+str(repeat)]
                if len(items) < 2:
                    continue
                a = F.normalize(torch.stack([v[1] for v in items]), dim=-1)
                b = F.normalize(torch.stack([v[2] for v in items]), dim=-1)
                recalls.append(float(((a@b.t()).argmax(1) == torch.arange(len(items))).float().mean()))
            summary[key]['common_to_text_retrieval_r1'] = sum(recalls)/len(recalls) if recalls else None
            summary[key]['retrieval_gallery_size'] = args.samples
        atomic_json(directory/'summary.json', {'version': VERSION, 'rows': len(rows), 'summary': summary,
                    'split': args.split, 'pretrained_v9_zero_shot_on_new_protocol': not is_v10,
                    'source_caption_used_by_receiver': False, 'diffusion_executed': False,
                    'common_only_rate_is_not_full_rate': args.ablation.startswith('common_only')})
        atomic_json(directory/'COMPLETE.json', {'rows': len(rows), 'expected_rows': expected, 'wall_seconds': time.monotonic()-started})
        identity['status'] = 'complete'
    except BaseException as exc:
        identity.update(status='failed_or_interrupted', exception=type(exc).__name__, error=str(exc))
        atomic_json(directory/'run_identity.json', identity)
        raise
    atomic_json(directory/'run_identity.json', identity)

if __name__ == '__main__':
    main(build_parser().parse_args())
