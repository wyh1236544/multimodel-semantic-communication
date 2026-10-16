"""One bounded stage, explicit inputs, offline existing GPU1 environment only."""
import argparse
import gc
import os
from pathlib import Path
import shutil
import sys
import time
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_rx_bridge_v16.core import PROJECT, FOLDER, atomic, safe, load, check_source, sha, read, seal, lines


def snapshot(root, manifest):
    for name in list(manifest['files'])+['MANIFEST.json']:
        target = root/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FOLDER/name, target)
    for name in manifest['dependencies']:
        target = root/'dependency_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT/name, target)


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('action', choices=('cache', 'oracle', 'actual', 'smoke_train'))
    p.add_argument('output')
    p.add_argument('--cache')
    p.add_argument('--oracle')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--smoke-root')
    p.add_argument('--config')
    p.add_argument('--max-hours', type=float, default=8)
    p.add_argument('--data-root', default=str(PROJECT))
    p.add_argument('--bart-path', default=str(PROJECT/'pretrained/bart-base'))
    p.add_argument('--vae-path', default=str(PROJECT/'pretrained/sd-vae-ft-mse'))
    p.add_argument('--longclip-path', default=str(PROJECT/'longclip-B.pt'))
    p.add_argument('--text-checkpoint')
    p.add_argument('--image-checkpoint')
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or not 0 < a.max_hours <= 24:
        p.error('Explicit physical GPU1 and finite0..24h budget required')
    if (a.action == 'smoke_train') != (a.smoke and a.action != 'cache'):
        p.error('smoke_train must be explicitly smoke; production training is a separate action')
    c = load(a.config, a.smoke)
    manifest = check_source()
    root = safe(a.output)
    if root.exists():
        raise FileExistsError('Existing run preserved; choose new output')
    if shutil.disk_usage(PROJECT).free < (8 if a.smoke else c['min_disk_gib'])*2**30:
        raise RuntimeError('Insufficient free project disk; no automatic deletion')
    root.mkdir(parents=True, exist_ok=False)
    atomic(root/'config.json', c)
    snapshot(root, manifest)
    start = time.monotonic()
    try:
        import torch
        import transformers
        if torch.__version__.split('+')[0] != '1.12.0' or transformers.__version__ != '4.30.2':
            raise RuntimeError('Pinned torch1.12.0/transformers4.30.2 required; do not upgrade environment')
        if not torch.cuda.is_available():
            raise RuntimeError('Actual offline GPU smoke required')
        torch.set_num_threads(4)
        device = torch.device('cuda:0')
        free, total = torch.cuda.mem_get_info(device)
        print('GPU1 free={:.2f}GiB total={:.2f}GiB'.format(free/2**30, total/2**30), flush=True)
        if free < c['min_free_gib']*2**30:
            raise RuntimeError('Startup GPU free threshold not met; do not stop unrelated jobs')
        def budget():
            if time.monotonic()-start > a.max_hours*3600:
                raise TimeoutError('Stage time budget exhausted; no automatic retry')
            if torch.cuda.mem_get_info(device)[0] < c['reserve_free_gib']*2**30:
                raise RuntimeError('Free GPU reserve crossed; no automatic batch change')
            if shutil.disk_usage(PROJECT).free < 4*2**30:
                raise RuntimeError('Free disk reserve crossed; no deletion')
        identity = dict(action=a.action, smoke=a.smoke, physical_gpu=1,
                        source_manifest_sha256=sha(FOLDER/'MANIFEST.json'),
                        python=sys.version, torch=torch.__version__, transformers=transformers.__version__,
                        pid=os.getpid(), FSO=False, Common=False, diffusion=False,
                        full_semantic_success=False, config_frozen_before_evaluation=True)
        if a.action == 'cache':
            from docci_rx_bridge_v16.cache import create
            if not a.smoke:
                smoke = Path(a.smoke_root).resolve(strict=True) if a.smoke_root else None
                if smoke is None or not read(smoke/'VERIFIED.json')['smoke'] or read(smoke/'identity.json')['arms'] != ['self', 'oracle_clip', 'rx_clip', 'rx_i0']:
                    raise ValueError('Authentic all-arm saved-backbone smoke must pass before full caching')
                identity['smoke_root'] = str(smoke)
                identity['smoke_verified_sha256'] = sha(smoke/'VERIFIED.json')
            atomic(root/'progress.json', dict(stage='cache frozen pipelines', pid=os.getpid()))
            create(a, c, root, device, budget)
        else:
            from docci_rx_bridge_v16.cache import Dataset, text_model
            from docci_rx_bridge_v16.train import fit
            from docci_rx_bridge_v16.evaluate import evaluate
            from docci_rx_bridge_v16.report import verify_seal, index_check, verify
            cache = Path(a.cache).resolve(strict=True)
            verify_seal(cache, budget)
            index_check(cache, False, budget)
            receipt = read(cache/'VERIFIED.json')
            if not receipt['arrays_verified'] or receipt['smoke'] != a.smoke or read(cache/'config.json') != c:
                raise ValueError('Verified cache/smoke/config identity differs')
            if receipt.get('light_manifest_sha256') != sha(cache/'LIGHT_MANIFEST.json'):
                raise ValueError('Cache verification receipt does not bind sealed cache')
            identity.update(cache=str(cache), cache_index_sha256=sha(cache/'index.json'))
            arms = ['self', 'oracle_clip', 'rx_clip', 'rx_i0'] if a.smoke else ['self', 'oracle_clip'] if a.action == 'oracle' else ['rx_clip', 'rx_i0']
            identity['arms'] = arms
            if a.action == 'actual':
                oracle = Path(a.oracle).resolve(strict=True)
                verify_seal(oracle, budget)
                # Recalculate the screen from sealed output rows, not a mutable
                # success Boolean in an earlier CPU receipt.
                outcome = verify(oracle, arrays=False)
                old = read(oracle/'identity.json')
                if old['cache_index_sha256'] != identity['cache_index_sha256'] or read(oracle/'config.json') != c or outcome['smoke']:
                    raise ValueError('Oracle/self screen used different cache/config')
                if not outcome['comparisons']['oracle_clip']['numerical_screen_pass']:
                    raise ValueError('Oracle numerical screen NOT passed; inspect interface/fit, no actual-stage promotion')
                ref = root/'self_reference'
                shutil.copytree(oracle/'self/qual_dev', ref/'qual_dev')
                atomic(ref/'identity.json', dict(cache_index_sha256=identity['cache_index_sha256'], config=c,
                       trainable_parameters=outcome['trainable_parameters'], origin=str(oracle),
                       original_verification_sha256=sha(oracle/'VERIFIED.json')))
                atomic(ref/'oracle_screen.json', outcome['comparisons']['oracle_clip'])
            baseline = lines(cache/'baseline.jsonl')
            # Include only the8 training diagnostics plus100 validation texts in light records.
            shutil.copyfile(cache/'baseline.jsonl', root/'baseline.jsonl')
            shutil.copytree(cache/'images', root/'images')
            model, tok, backbone = text_model(a, device)
            from docci_longclip_joint_v8.backbones import signature
            initial = signature(model)
            training = Dataset(cache, 'train', c['memory_lru'])
            validation = Dataset(cache, 'qual_dev', c['memory_lru'])
            diagnostic = Dataset(cache, 'train', c['memory_lru'])
            diagnostic.entries = diagnostic.entries[:8]
            for arm in arms:
                budget()
                folder = root/arm
                folder.mkdir()
                atomic(root/'progress.json', dict(stage='fit', arm=arm, pid=os.getpid()))
                bridge = fit(model.bart, training, c, arm, folder, device, budget, a.smoke)
                try:
                    evaluate(model.bart, bridge, tok, diagnostic, arm, folder/'train', device, budget, baseline)
                    evaluate(model.bart, bridge, tok, validation, arm, folder/'qual_dev', device, budget, baseline)
                finally:
                    bridge.close()
                if signature(model) != initial:
                    raise RuntimeError('Original frozen codec/BART changed during bridge training')
                del bridge
                gc.collect()
                torch.cuda.empty_cache()
            atomic(root/'original_pipeline_audit.json', dict(before=initial, after=signature(model),
                   frozen_codec_and_BART_unchanged=initial == signature(model)))
        check_source()
        identity['wall_seconds'] = time.monotonic()-start
        atomic(root/'identity.json', identity)
        seal(root)
        print('STAGE_FINISHED; independent CPU verification/collection still required', flush=True)
    except BaseException as exc:
        atomic(root/'FAILED.json', dict(exception=type(exc).__name__, message=str(exc)))
        raise


if __name__ == '__main__':
    main()
