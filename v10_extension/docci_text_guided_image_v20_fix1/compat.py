"""Scoped pooling repair; original source, Torch globals and study unchanged."""
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import shutil
from types import SimpleNamespace
from docci_text_guided_image_v20.core import (FOLDER as ORIGINAL, PROJECT, safe,
    sha, read, atomic, check_source, load, Budget)

FOLDER = Path(__file__).resolve().parent
ORIGINAL_MANIFEST = 'dc17591df832119faf2858e36dd73128b73cf32872cafcef6803943e8d7f90b6'
SCHEMA = 'v20-deterministic-pooling-fix1'

def layout(shape, output_size=(4,8)):
    if tuple(output_size)!=(4,8) or len(shape)!=4:
        raise ValueError('Only the declared V20 4x8 self-pooling is repaired')
    b,c,h,w = shape
    if min(b,c,h,w)<=0 or h%4 or w%8:
        raise ValueError('V20 pooling requires positive dimensions divisible by 4x8')
    return b,c,4,h//4,8,w//8

def fixed_pool(image, output_size=(4,8)):
    # Non-overlapping blocks: mathematically identical adaptive bins at all
    # current 16/32/64 scales. No detach, parameter, CPU transfer or atomic add.
    return image.reshape(layout(image.shape,output_size)).mean(dim=(3,5))

@contextmanager
def pooling_patch(module):
    original = module.F
    module.F = SimpleNamespace(adaptive_avg_pool2d=fixed_pool)
    try:
        yield
    finally:
        module.F = original

def source_check():
    if sha(ORIGINAL/'MANIFEST.json')!=ORIGINAL_MANIFEST:
        raise ValueError('Expected original immutable V20; do not rewrite old manifests')
    check_source()
    manifest = read(FOLDER/'MANIFEST.json')
    if manifest['original_manifest_sha256']!=ORIGINAL_MANIFEST:
        raise ValueError('Wrong fix parent')
    for n,h in manifest['files'].items():
        if PurePosixPath(n).name!=n or (FOLDER/n).is_symlink() or sha(FOLDER/n)!=h:
            raise ValueError('Fix1 source missing/changed: '+n)
    return manifest

def validate_gate(value):
    if (value.get('schema')!=SCHEMA or value.get('repair_manifest_sha256')!=sha(FOLDER/'MANIFEST.json')
            or value.get('CUDA_gate_passed') is not True or value.get('strict_determinism') is not True
            or value.get('GPU_tests_run')!=3 or value.get('GPU_tests_skipped')!=0
            or value.get('physical_GPU')!='1'):
        raise ValueError('Actual current-fix CUDA gate missing or invalid')
    return value

def snapshot(root, gate_path):
    manifest = source_check()
    gate = validate_gate(read(gate_path))
    dest = root/'compatibility_snapshot/docci_text_guided_image_v20_fix1'
    dest.mkdir(parents=True,exist_ok=False)
    for n in list(manifest['files'])+['MANIFEST.json']:
        shutil.copyfile(FOLDER/n,dest/n)
    atomic(root/'FIX1_GPU_GATE.json',gate)
    atomic(root/'V20_FIX1.json',dict(schema=SCHEMA,
        repair_manifest_sha256=sha(FOLDER/'MANIFEST.json'),
        original_manifest_sha256=ORIGINAL_MANIFEST,
        change='Fusion module-local adaptive pooling -> fixed 4x8 block means',
        global_determinism_disabled=False,parameters_or_protocol_changed=False,
        historical_files_overwritten=False,exact_training_resume=False))

def verify_fix(root, original_verify, *args, **kwargs):
    root = Path(root)
    result = original_verify(root,*args,**kwargs)
    receipt = read(root/'V20_FIX1.json')
    if receipt.get('schema')!=SCHEMA or receipt['repair_manifest_sha256']!=sha(FOLDER/'MANIFEST.json'):
        raise ValueError('Wrong repair provenance')
    snap = root/'compatibility_snapshot/docci_text_guided_image_v20_fix1'
    manifest = source_check()
    for n in list(manifest['files'])+['MANIFEST.json']:
        if sha(snap/n)!=sha(FOLDER/n):
            raise ValueError('Fix snapshot changed: '+n)
    validate_gate(read(root/'FIX1_GPU_GATE.json'))
    for label in result['stages']:
        if 'train_' in label or 'eval_' in label:
            runtime = read(root/label/'FIX1_RUNTIME.json')
            if (runtime.get('repair_manifest_sha256')!=sha(FOLDER/'MANIFEST.json')
                    or runtime.get('strict_determinism') is not True
                    or runtime.get('module_local_pooling_replaced') is not True):
                raise ValueError('Verified neural stage did not execute the deterministic repair')
    return dict(result,compatibility_fix1=dict(schema=SCHEMA,
        repair_manifest_sha256=sha(FOLDER/'MANIFEST.json'),
        CUDA_gate_sha256=sha(root/'FIX1_GPU_GATE.json'),
        global_determinism_disabled=False,original_files_overwritten=False))

@contextmanager
def installed(neural=False):
    source_check()
    from docci_text_guided_image_v20 import report, collect
    old_verify, old_collect_verify = report.verify, collect.verify
    def verified(root,*args,**kwargs):
        return verify_fix(root,old_verify,*args,**kwargs)
    report.verify = collect.verify = verified
    try:
        if neural:
            from docci_text_guided_image_v20 import model
            with pooling_patch(model):
                yield verified
        else:
            yield verified
    finally:
        report.verify,collect.verify = old_verify,old_collect_verify

def reuse_cache(source, target, max_hours=1):
    """Verify -> byte-copy only 12 smoke arrays -> verify. Never alter the source."""
    from docci_text_guided_image_v20.report import check_cache
    source,target = safe(source),safe(target)
    if target.exists() or source==target or source in target.parents:
        raise FileExistsError('Reuse requires a NEW separate cache destination')
    if not source.is_dir() or read(source/'index.json').get('smoke') is not True:
        raise ValueError('Only a completed original smoke cache can be reused')
    if read(source.parent/'config.json')!=read(target.parent/'config.json'):
        raise ValueError('Source/new experiment configurations differ')
    c = load(target.parent/'config.json',True)
    budget = Budget(max_hours)
    check_cache(source,c,True,True,budget)
    index = read(source/'index.json')
    if index.get('image_codec_outputs_cached') is not False or index.get('V18_text_wire_frozen') is not True:
        raise ValueError('Wrong cache semantics')
    files = {'CACHE_COMPLETE.json','backbones.json','frozen_audit.json',
             'index.json','parents.json','protected_split.json','source_identity.json'}
    for row in index['records']:
        n = row['cache_file']
        if PurePosixPath(n).parts!=('arrays',row['example_id']+'.npz'):
            raise ValueError('Unsafe/noncanonical array path')
        files.add(n)
    actual = set()
    for p in source.rglob('*'):
        if p.is_symlink():
            raise ValueError('Symlink in source cache rejected')
        if p.is_file():
            actual.add(p.relative_to(source).as_posix())
    if actual!=files:
        raise ValueError('Unexpected/missing source cache file')
    before = {n:sha(source/n) for n in files}
    target.mkdir(parents=True,exist_ok=False)
    # Completion receipt is copied last; no hardlinks/shared mutable arrays.
    order = sorted(files-{'CACHE_COMPLETE.json'})+['CACHE_COMPLETE.json']
    for n in order:
        budget()
        p = target/n
        p.parent.mkdir(parents=True,exist_ok=True)
        with (source/n).open('rb') as src, p.open('xb') as dst:
            shutil.copyfileobj(src,dst)
        if sha(p)!=before[n] or sha(source/n)!=before[n]:
            raise ValueError('Cache bytes changed during reuse')
    check_cache(target,c,True,True,budget)
    if any(sha(source/n)!=h for n,h in before.items()):
        raise ValueError('Preserved cache changed')
    atomic(target.parent/'SMOKE_CACHE_REUSE.json',dict(source=str(source.resolve()),
        original_files_sha256=before,cache_generation_repeated=False,
        reuse='verified independent byte copy, not hardlink/symlink',
        original_failure_preserved=True))
    print('FIX1 reused verified 8+2+2 smoke cache; source unchanged',flush=True)
