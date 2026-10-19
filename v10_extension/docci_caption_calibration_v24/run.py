"""Explicit two-image offline diagnosis. All neural paths require real GPU assets."""
import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import shutil
import sys
import time
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_caption_calibration_v24.core import (FOLDER, PROJECT, SCHEMA, PROTOCOL,
    Budget, atomic, line, read, sha, safe, local, source_check, make_plan, digest,
    generation_kwargs, require_review, verify_assets)

def parser():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('action', choices=('caption', 'qa'))
    p.add_argument('output')
    p.add_argument('--prepared', default=str(PROJECT/'experiment_results/v23_inputs_smoke_01'))
    p.add_argument('--asset-lock', default=str(PROJECT/'experiment_results/v23_caption_lock_01/ASSET_LOCK.json'))
    p.add_argument('--dtype', choices=('float16', 'float32'), default='float16')
    p.add_argument('--max-minutes', type=float, default=30.)
    p.add_argument('--caption-root')
    p.add_argument('--review')
    return p

def import_inputs(a, root):
    from PIL import Image
    from docci_evidence_v23.core import prepared, verify_files
    src = Path(a.prepared).resolve(strict=True)
    index, data, _ = prepared(src)
    plan = make_plan(index, data, a.action)
    before = verify_files(src, 'prepare')
    folder = root/'prepared'; folder.mkdir()
    for n in sorted(list(before['files'])+['COMPLETE.json']):
        if Path(n).suffix not in ('.json', '.jsonl', '.png'):
            raise ValueError('Unexpected heavy/executable prepared file '+n)
        path = local(folder, n); path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local(src, n), path)
    # Re-audit copied bytes independently, and preserve the historical root read-only.
    prepared(folder)
    if verify_files(src, 'prepare') != before:
        raise ValueError('Input changed during copy')
    gray = root/'gray.png'; Image.new('RGB', (512, 512), (128, 128, 128)).save(gray)
    for r in plan:
        if r['source_file'] is None:
            r['image_file'] = 'gray.png'; r['image_sha256'] = sha(gray)
        else:
            r['image_file'] = 'prepared/'+r.pop('source_file')
        r.pop('source_file', None)
        with Image.open(local(root, r['image_file'])) as im:
            if im.mode != 'RGB' or im.size != (512, 512):
                raise ValueError('Only actual RGB512 full-view PNGs accepted')
    atomic(root/'PLAN.json', plan)
    return plan

def complete(root):
    files = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    atomic(root/'COMPLETE.json', dict(schema=SCHEMA, files=files,
        source_manifest_sha256=sha(FOLDER/'MANIFEST.json'), training_updates=0,
        semantic_success=False, facts='PENDING', state='INFERENCE_FINISHED'))

def main():
    a = parser().parse_args(); source_check(); budget = Budget(a.max_minutes)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('Use launcher with physical GPU1; no implicit device choice')
    root = safe(a.output)
    if root.exists():
        raise FileExistsError('Use a NEW output; existing run preserved')
    if a.action == 'qa' and not (a.caption_root and a.review):
        raise ValueError('QA requires reviewed native-caption run; no automatic promotion')
    if a.action == 'caption' and (a.caption_root or a.review):
        raise ValueError('Caption diagnosis does not consume a previous QA authorization')
    from docci_evidence_v23.core import prepared
    index, data, _ = prepared(a.prepared); make_plan(index, data, a.action)
    lock = read(a.asset_lock)
    if (lock.get('kind') != 'caption' or lock.get('revision') != '1ef7f63a8f0a144c13fdca8103eb7b4691c74cec' or
            lock.get('origin') != 'Salesforce/blip2-opt-2.7b'):
        raise ValueError('Need existing verified V23 official-2023 asset lock')
    authorization = None
    if a.action == 'qa':
        from docci_caption_calibration_v24.report import verify
        result = verify(a.caption_root)
        prior = read(Path(a.caption_root)/'RUN.json')
        if prior['stage'] != 'caption' or prior['prepared_complete_sha256'] != sha(Path(a.prepared)/'COMPLETE.json') or prior['asset_lock_sha256'] != sha(a.asset_lock) or prior['dtype'] != a.dtype:
            raise ValueError('Reviewed caption input/model/precision differs')
        authorization = require_review(a.caption_root, a.review, result)
    root.mkdir(parents=True, exist_ok=False)
    atomic(root/'RUN.json', dict(schema=SCHEMA, stage=a.action, arguments=vars(a),
        interpreter=sys.executable, python=sys.version, dtype=a.dtype, protocol=PROTOCOL,
        prepared_complete_sha256=sha(Path(a.prepared)/'COMPLETE.json'),
        asset_lock_sha256=sha(a.asset_lock), source_manifest_sha256=sha(FOLDER/'MANIFEST.json'),
        training_updates=0, automatic_next_stage=False))
    try:
        shutil.copyfile(a.asset_lock, root/'ASSET_LOCK.json')
        if authorization is not None:
            atomic(root/'QA_AUTHORIZATION.json', dict(review=authorization,
                caption_run=read(Path(a.caption_root)/'RUN.json'),
                caption_verification=result))
        plan = import_inputs(a, root)
        snapshot = root/'source_snapshot'; snapshot.mkdir()
        m = read(FOLDER/'MANIFEST.json')
        for name in list(m['files'])+['MANIFEST.json']:
            shutil.copyfile(FOLDER/name, snapshot/name)
        print('V24 VERIFY existing sealed model bytes; no download/retraining', flush=True)
        folder = verify_assets(lock, budget); budget()
        asset_stats = {n: [local(folder, n).stat().st_size, local(folder, n).stat().st_mtime_ns]
            for n in lock['tree']['files']}
        cfg = read(folder/'config.json')
        if cfg.get('model_type') != 'blip-2' or cfg.get('image_token_index') is not None:
            raise ValueError('Pinned old-transformers BLIP-2 configuration required')
        import torch
        import transformers
        if torch.__version__.split('+')[0] != '1.12.0' or transformers.__version__ != '4.30.2':
            raise ValueError('Expected server Torch1.12.0/HF4.30.2; do not upgrade environment')
        if not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA required, no implicit CPU/full-model fallback')
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        device = torch.device('cuda:0')
        free, total = torch.cuda.mem_get_info()
        floor = 20 if a.dtype == 'float16' else 36
        print('GPU1 free={:.2f}GiB total={:.2f}GiB; {} start floor={}GiB'.format(
            free/2**30, total/2**30, a.dtype, floor), flush=True)
        if free < floor*2**30:
            raise RuntimeError('Insufficient free GPU reserve; no job termination/batch changes')
        torch.cuda.reset_peak_memory_stats(); budget.gpu = True
        from docci_caption_calibration_v24.runtime import load_model, state_digest, generate_one
        dtype = torch.float16 if a.dtype == 'float16' else torch.float32
        print('V24 LOAD frozen BLIP-2; source text is evaluation-only', flush=True)
        model, processor, loading = load_model(folder, dtype, device); budget()
        print('V24 AUDIT frozen loaded tensors before inference', flush=True)
        before = state_digest(model, budget)
        atomic(root/'MODEL.json', dict(loading_info=loading,
            config=model.config.to_dict(), generation_config=model.generation_config.to_dict(),
            explicit_generation_kwargs=generation_kwargs(a.action),
            processor=processor.image_processor.to_dict(),
            versions={n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'Pillow', 'numpy')},
            state_before_sha256=before, model_file_set_and_hash_verified=True,
            pretraining_overlap='unknown', trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
        cache = {}; calls = []; start = time.monotonic()
        from PIL import Image
        with (root/'rows.jsonl').open('x', encoding='utf-8') as f:
            for i, request in enumerate(plan):
                budget()
                # Identical model inputs share ONE actual inference. Donor/gray copies
                # are not independent trials or evidence of visual insensitivity.
                call_id = digest([request['image_sha256'], request['prompt'], a.action, a.dtype])
                reused = call_id in cache
                if not reused:
                    with Image.open(local(root, request['image_file'])) as im:
                        value = generate_one(model, processor, im.convert('RGB'),
                            request['prompt'], a.action, device, dtype, budget)
                    cache[call_id] = value; calls.append(call_id)
                value = cache[call_id]
                line(f, dict(request, inference_id=call_id, reused_inference=reused,
                    output_sha256=hashlib.sha256(value['output_text'].encode()).hexdigest(), **value))
                print('V24 {} {}/{} control={} snr={} new_tokens={} empty={} reused={}'.format(
                    a.action, i+1, len(plan), request['control'], request['snr_db'],
                    value['telemetry']['new_token_count'], value['text_audit']['empty'], reused), flush=True)
        print('V24 AUDIT frozen loaded tensors after inference', flush=True)
        after = state_digest(model, budget)
        if after != before or any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise RuntimeError('Frozen model changed')
        atomic(root/'INFERENCE.json', dict(unique_inferences=len(calls), unique_inference_ids=calls,
            rows=len(plan), elapsed_inference_seconds=time.monotonic()-start,
            state_before_sha256=before, state_after_sha256=after,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            startup_free_gib=free/2**30, GPU='physical1', training_updates=0,
            source_images_privileged_diagnosis_only=True, source_text_at_model=False,
            received_text_at_model=False, facts='PENDING', semantic_success=False))
        # Check size/mtime after loading without rereading 15GB twice.
        # The full sealed tree was hashed before deserialization; weights never written.
        final_stats = {n: [local(folder, n).stat().st_size, local(folder, n).stat().st_mtime_ns]
            for n in lock['tree']['files']}
        if asset_stats != final_stats:
            raise RuntimeError('Model asset stat changed during inference')
        atomic(root/'ASSET_AUDIT.json', dict(before=asset_stats, after=final_stats,
            full_hash_before_load=True, full_hash_after_load=False))
        source_check(); complete(root)
        print('V24 INFERENCE_FINISHED; factual usability requires human review; no next stage', flush=True)
    except BaseException as error:
        if not (root/'FAILED.json').exists():
            atomic(root/'FAILED.json', dict(error_type=type(error).__name__, message=str(error),
                partial_preserved=True, automatic_retry=False))
        raise

if __name__ == '__main__':
    main()
