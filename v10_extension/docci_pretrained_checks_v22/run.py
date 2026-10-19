"""Independent, explicit stages. No pipeline silently starts training/downloads."""
import argparse
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_pretrained_checks_v22.core import (PROJECT, FOLDER, CONFIG, SCHEMA, safe,
    atomic, sha, read, rows, local, prepared, seal_tree, complete, source_check, Budget, case_name)

def inventory(a, root):
    packages = {}
    for name in ('torch', 'transformers', 'diffusers', 'Pillow', 'numpy', 'safetensors', 'sentencepiece'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    paths = dict(v20_root=a.v20_root, bart=a.bart_path, vae=a.vae_path,
        lpips=a.lpips_assets, caption=a.caption_path, diffusion=a.diffusion_path,
        bpg_encoder=a.bpg_encoder, bpg_decoder=a.bpg_decoder)
    status = {k: dict(path=v, exists=bool(v and Path(v).exists())) for k, v in paths.items()}
    value = dict(schema=SCHEMA, action='inventory', packages=packages, assets=status,
        training=False, downloads=False, GPU_allocated=False,
        weights_loaded=False, new_inference_executed=False,
        next='prepare smoke needs old V20 assets; caption/enhance need separate sealed local model assets',
        note='Existence is NOT proof of valid weights, compatibility or available VRAM')
    atomic(root/'ASSETS.json', value)
    complete(root, 'inventory', dict(no_training=True, no_model_loading=True))
    print(json.dumps(value, indent=2, ensure_ascii=False), flush=True)

def lock_assets(a, root):
    if not a.model_path or not a.kind or not a.origin_id or not a.revision:
        raise ValueError('Need --kind, --model-path, --origin-id and --revision; no guessed provenance')
    tree = seal_tree(a.model_path)
    cfg = Path(a.model_path)/('config.json' if a.kind=='caption' else 'model_index.json')
    meta = read(cfg)
    if a.kind=='caption' and meta.get('model_type') != 'blip-2':
        raise ValueError('Only BLIP-2 supported by this adapter')
    if a.kind=='diffusion' and meta.get('_class_name') not in ('StableDiffusionPipeline', 'StableDiffusionImg2ImgPipeline'):
        raise ValueError('Use SD1/2 img2img assets; not SDXL or unsupported ControlNet')
    atomic(root/'ASSET_LOCK.json', dict(schema=SCHEMA, kind=a.kind, tree=tree,
        declared_origin=a.origin_id, declared_revision=a.revision, origin_independently_verified=False,
        overlap_audit=a.overlap_audit, model_training_data_overlap_known=a.overlap_audit!='unknown',
        remote_code_allowed=False, downloads=False))
    complete(root, 'assets', dict(kind=a.kind, bytes=sum((Path(a.model_path)/n).stat().st_size for n in tree['files'])))
    print('Asset bytes frozen. Origin and training contamination still require documentary audit.', flush=True)

def digital(a, root, budget):
    import numpy as np
    from PIL import Image
    from docci_pretrained_checks_v22.digital import (PROFILES, encode, receive, channel,
        source_encode, reconstruct, binary, llrs)
    index, data, examples = prepared(a.prepared)
    encoder = decoder = None
    if a.codec=='bpg':
        if not a.bpg_encoder or not a.bpg_decoder or not a.bpg_encoder_sha256 or not a.bpg_decoder_sha256:
            raise ValueError('BPG requires explicit project-local Linux binaries AND their SHA256; no auto install/fallback')
        encoder = binary(safe(a.bpg_encoder), a.bpg_encoder_sha256)
        decoder = binary(safe(a.bpg_decoder), a.bpg_decoder_sha256)
    profiles = list(PROFILES)
    (root/'images').mkdir()
    (root/'tx').mkdir()
    payloads = {}
    # Source coding is TX-only, once per example/profile, and never uses SNR or output metrics.
    for eid, ex in examples.items():
        with Image.open(local(a.prepared, ex['source_image'])) as im:
            source = im.convert('RGB')
        for profile in profiles:
            print('V22 SOURCE {} {} {}'.format(eid, a.codec, profile), flush=True)
            payload, choice = source_encode(source, profile, a.codec, root, encoder, budget)
            kind = 0 if a.codec=='jpeg' else 1
            # Assert a clean actual full packet roundtrip, not a bit-budget approximation.
            signal, allocation, private = encode(payload, kind, profile)
            check = receive(signal, profile)
            if check['payload'] != payload or not check['packet_crc']:
                raise RuntimeError('Actual digital back-to-back gate failed')
            tx = root/'tx'/(eid+'__'+profile+('.jpg' if kind==0 else '.bpg'))
            tx.write_bytes(payload)
            payloads[eid, profile] = payload, choice, signal, allocation, private
    with (root/'rows.jsonl').open('x', encoding='utf-8') as f:
        for i, r in enumerate(data):
            budget()
            for profile in profiles:
                payload, choice, signal, allocation, private = payloads[r['example_id'], profile]
                y = channel(signal, r['example_id'], r['snr_db'], r['repeat'])
                rx = receive(y, profile)
                decoded = rx.pop('decoded_bits')
                value = rx.pop('payload')
                # Source truth is checked ONLY after RX has returned, for BER/undetected-error auditing.
                wrong = None if decoded is None else int(np.count_nonzero(decoded != private['packet']))
                if rx['packet_crc'] and value != payload:
                    raise RuntimeError('Undetected wrong packet; preserve run, never repair from source')
                if rx['packet_crc']:
                    restored = reconstruct(value, rx['kind'], decoder, root)
                else:
                    restored = Image.new('RGB', (512, 512), (128, 128, 128))
                path = root/'images'/(case_name(r['example_id'], r['snr_db'], r['repeat'])+'__'+profile+'.png')
                restored.save(path)
                header = (llrs(y[:680], 1., 2)<0).astype(np.uint8)
                body = (llrs(y[680:], 1., allocation['bps'], allocation['body_gain'])<0).astype(np.uint8)
                hard = np.concatenate((header, body))
                row = dict(example_id=r['example_id'], snr_db=r['snr_db'], repeat=r['repeat'],
                    control=a.codec+'_'+profile, image_file=path.relative_to(root).as_posix(), image_sha256=sha(path),
                    source_coding=choice, allocation=allocation, rx=rx,
                    pre_fec_errors=int(np.count_nonzero(hard != private['bits'])), pre_fec_bits=len(hard),
                    post_fec_errors=wrong, post_fec_bits=0 if wrong is None else len(private['packet']),
                    source_image_at_RX=False, source_text_at_RX=False,
                    post_noise_renormalization=False, failure_fill='public constant RGB128',
                    actual_average_power=float(np.mean(np.abs(signal)**2)),
                    post_fec_BER_conditional_on_decodable_header=True)
                f.write(json.dumps(row, allow_nan=False)+'\n'); f.flush()
            print('V22 DIGITAL {}/{} four actual MCS'.format(i+1, len(data)), flush=True)
    atomic(root/'identity.json', dict(prepared_sha256=sha(Path(a.prepared)/'COMPLETE.json'),
        codec=a.codec, BPG_implemented=a.codec=='bpg', BPG_executed=a.codec=='bpg',
        bpg_encoder_sha256=a.bpg_encoder_sha256, bpg_decoder_sha256=a.bpg_decoder_sha256,
        total_symbols=4128, image_payload_and_all_header_body=4096, pilots=32,
        same_image_frame_energy=True, ARQ=False, actual_receiver=True,
        JPEG_is_low_rate_reference_not_best_digital=True, no_per_SNR_best_MCS_selection=True,
        no_fading=True, no_waveform_bandwidth_or_bitrate_claim=True))
    complete(root, 'digital', dict(rows=len(data)*4, all_clean_packets_roundtrip=True,
        perceptual_scoring='NOT_RUN; use score stage', generative_postprocessing='NOT_RUN'))

def parser():
    from docci_text_guided_image_v20.run import arguments
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('action', choices=('inventory', 'lock-assets', 'prepare', 'digital', 'caption', 'enhance', 'score'))
    p.add_argument('output')
    p.add_argument('--max-hours', type=float, default=2.)
    p.add_argument('--profile', choices=tuple(CONFIG['profiles']), default='smoke')
    p.add_argument('--v20-root', default=str(PROJECT/'experiment_results/v20_text_guided_image_fix1_01'))
    p.add_argument('--prepared'); p.add_argument('--target')
    p.add_argument('--smoke-root')
    p.add_argument('--receiver-stage')
    p.add_argument('--receiver-control')
    p.add_argument('--caption-path', default=str(PROJECT/'pretrained/blip2-opt-2.7b'))
    p.add_argument('--diffusion-path', default=str(PROJECT/'pretrained/stable-diffusion-v1-5'))
    p.add_argument('--model-path'); p.add_argument('--asset-lock')
    p.add_argument('--kind', choices=('caption', 'diffusion'))
    p.add_argument('--origin-id'); p.add_argument('--revision')
    p.add_argument('--overlap-audit', choices=('unknown', 'documented_no_known_DOCCI', 'documented_overlap'), default='unknown')
    p.add_argument('--codec', choices=('jpeg', 'bpg'), default='jpeg')
    p.add_argument('--bpg-encoder'); p.add_argument('--bpg-decoder')
    p.add_argument('--bpg-encoder-sha256'); p.add_argument('--bpg-decoder-sha256')
    arguments(p)
    return p

def main():
    a = parser().parse_args()
    source_check()
    if a.config:
        raise ValueError('V22 fixed protocol does not use historical V20 --config')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # No custom HF remote code, no dependency downloads, no environment dump.
    root = safe(a.output)
    if root.exists():
        raise FileExistsError('Existing output preserved; choose NEW output')
    if a.action in ('digital', 'caption', 'enhance', 'score') and not a.prepared:
        raise ValueError('Explicit verified --prepared required')
    if a.action in ('caption', 'enhance') and not a.asset_lock:
        raise ValueError('Separate sealed --asset-lock required before model loading')
    if a.action == 'score' and not a.target:
        raise ValueError('Explicit --target required')
    if a.action in ('prepare', 'caption', 'enhance'):
        from docci_pretrained_checks_v22.core import verify_files
        profile = a.profile if a.action=='prepare' else read(Path(a.prepared)/'index.json')['profile']
        if profile != 'smoke':
            if not a.smoke_root:
                raise ValueError('Verified actual --smoke-root mandatory before expansion')
            verified = verify_files(a.smoke_root, a.action)
            if a.action=='prepare':
                if read(Path(a.smoke_root)/'index.json')['profile'] != 'smoke':
                    raise ValueError('Not a real smoke preparation')
            elif read(Path(a.smoke_root)/'identity.json')['asset_lock_sha256'] != sha(a.asset_lock):
                raise ValueError('Expanded run must use identical smoke-checked model assets')
    budget = Budget(a.max_hours)
    root.mkdir(parents=True, exist_ok=False)
    versions = {}
    for name in ('torch', 'transformers', 'diffusers', 'numpy', 'Pillow'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    atomic(root/'RUN_IDENTITY.json', dict(schema=SCHEMA, action=a.action,
        interpreter=sys.executable, python=sys.version, packages=versions,
        source_manifest_sha256=sha(FOLDER/'MANIFEST.json'),
        physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'), arguments=vars(a),
        training_updates=0, no_automatic_download_or_install=True))
    started = time.monotonic()
    try:
        if a.action=='inventory':
            inventory(a, root)
        elif a.action=='lock-assets':
            lock_assets(a, root)
        elif a.action=='digital':
            digital(a, root, budget)
        else:
            from docci_pretrained_checks_v22 import neural
            getattr(neural, a.action)(a, root, budget)
        source_check()
        print('V22 {} FINISHED in {:.1f}s; semantic success not established'.format(a.action, time.monotonic()-started), flush=True)
    except BaseException as error:
        atomic(root/'FAILED.json', dict(stage=a.action, error_type=type(error).__name__,
            message=str(error), partial_preserved=True, automatic_retry=False))
        raise

if __name__=='__main__':
    main()
