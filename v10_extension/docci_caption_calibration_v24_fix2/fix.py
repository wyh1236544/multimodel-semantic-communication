"""Metadata-only dtype serialization repair; frozen V24 and Fix1 remain unchanged."""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_caption_calibration_v24 import core as c
from docci_caption_calibration_v24_fix1 import fix as previous

FOLDER = Path(__file__).resolve().parent
REPAIR = 'v24-fix2-strict-metadata-dtype-copy'

def source_check():
    m = c.read(FOLDER/'MANIFEST.json')
    for base, mapping in ((FOLDER, m['files']), (c.PROJECT, m['dependencies'])):
        for name, h in mapping.items():
            if c.sha(c.local(base, name)) != h:
                raise ValueError('Fix2 pinned source differs '+name)
    previous.source_check()
    return m

def metadata_copy(value, dtype_type=None):
    """Only actual Torch dtypes gain a new representation; never default=str.

    dtype_type injection is solely for explicitly labelled local CPU unit tests.
    Production obtains the actual torch.dtype class from the loaded torch module.
    JSON keys follow stdlib spelling, with collisions rejected rather than lost.
    """
    if dtype_type is None:
        dtype_type = getattr(sys.modules.get('torch'), 'dtype', ())
    events, active = [], set()
    def walk(item, path):
        if isinstance(item, dtype_type):
            before = str(item)
            if not before.startswith('torch.') or not before[6:].isidentifier():
                raise TypeError('Invalid dtype spelling at '+repr(path))
            events.append(dict(path=list(path), kind='torch.dtype', before=before, after=before[6:]))
            return before[6:]
        if item is None or type(item) in (str, bool, int):
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError('Nonfinite metadata at '+repr(path))
            return item
        if isinstance(item, (dict, list, tuple)):
            if id(item) in active:
                raise ValueError('Cyclic metadata at '+repr(path))
            active.add(id(item))
            try:
                if isinstance(item, dict):
                    result = {}
                    for key, child in item.items():
                        if type(key) is str:
                            name = key
                        elif key is None or type(key) in (bool, int, float):
                            if type(key) is float and not math.isfinite(key):
                                raise ValueError('Nonfinite JSON key at '+repr(path))
                            name = json.dumps(key, allow_nan=False)
                        else:
                            raise TypeError('Unsupported JSON key at '+repr(path))
                        if name in result:
                            raise ValueError('Colliding JSON keys at '+repr(path+[name]))
                        result[name] = walk(child, path+[name])
                    return result
                return [walk(child, path+[i]) for i, child in enumerate(item)]
            finally:
                active.remove(id(item))
        raise TypeError('Unsupported metadata {} at {}'.format(type(item).__name__, repr(path)))
    result = walk(value, [])
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    return result, events

@contextmanager
def repaired(root):
    """Only MODEL/FIX1 metadata in this run is normalized, before strict writes."""
    from docci_caption_calibration_v24 import runtime, run
    root = Path(root).absolute()
    old_atomic, old_run_atomic = c.atomic, run.atomic
    with previous.repaired():
        old_load = runtime.load_model
        preflight, writes = [], {}
        def load(*args, **kwargs):
            model, processor, info = old_load(*args, **kwargs)
            _, events = metadata_copy(dict(loading_info=info, config=model.config.to_dict(),
                generation_config=model.generation_config.to_dict(),
                processor=processor.image_processor.to_dict()))
            preflight.extend(events)
            print('V24 FIX2 metadata preflight passed; dtype fields='+
                json.dumps([x['path'] for x in events]), flush=True)
            return model, processor, info
        def atomic(path, value):
            target = Path(path).absolute()
            if target.parent != root or target.name not in ('MODEL.json', 'FIX1.json'):
                return old_atomic(path, value)
            normalized, events = metadata_copy(value)
            # Validate before the original writer creates any .writing file.
            old_atomic(path, normalized)
            writes[target.name] = events
            if target.name == 'FIX1.json':
                if 'MODEL.json' not in writes:
                    raise ValueError('Fix2 requires the actual MODEL write')
                m = source_check()
                snapshot = root/'fix2_source_snapshot'; snapshot.mkdir()
                for name in list(m['files'])+['MANIFEST.json']:
                    shutil.copyfile(FOLDER/name, snapshot/name)
                old_atomic(root/'FIX2.json', dict(repair=REPAIR,
                    repair_manifest_sha256=c.sha(FOLDER/'MANIFEST.json'),
                    fix1_manifest_sha256=c.sha(previous.FOLDER/'MANIFEST.json'),
                    preflight_dtype_conversions=preflight, metadata_dtype_conversions=writes,
                    scope=['MODEL.json', 'FIX1.json'], strict_json_allow_nan=False,
                    original_sources_modified=False, model_config_mutated=False,
                    model_weights_changed=False, generation_kwargs_changed=False,
                    quality_rules_changed=False))
        runtime.load_model, c.atomic, run.atomic = load, atomic, atomic
        try:
            yield
        finally:
            runtime.load_model, c.atomic, run.atomic = old_load, old_atomic, old_run_atomic

def verify(root):
    result = previous.verify(root)
    root = Path(root); info = c.read(root/'FIX2.json')
    flags = ('original_sources_modified', 'model_config_mutated', 'model_weights_changed',
        'generation_kwargs_changed', 'quality_rules_changed', 'strict_json_allow_nan')
    if (info['repair'] != REPAIR or
            info['repair_manifest_sha256'] != c.sha(FOLDER/'MANIFEST.json') or
            info['fix1_manifest_sha256'] != c.sha(previous.FOLDER/'MANIFEST.json') or
            info['scope'] != ['MODEL.json', 'FIX1.json'] or any(info[k] is not False for k in flags)):
        raise ValueError('Fix2 provenance differs')
    conversions = info['metadata_dtype_conversions']
    if set(conversions) != {'MODEL.json', 'FIX1.json'}:
        raise ValueError('Fix2 write scopes differ')
    for name, events in conversions.items():
        value = c.read(root/name); seen = set()
        for event in events:
            path = event['path']; key = tuple(path)
            if (not path or key in seen or event['kind'] != 'torch.dtype' or
                    event['before'] != 'torch.'+event['after'] or not event['after'].isidentifier()):
                raise ValueError('Invalid dtype conversion audit')
            seen.add(key); found = value
            for component in path:
                found = found[component]
            if found != event['after']:
                raise ValueError('Dtype conversion does not match saved metadata')
    m = c.read(FOLDER/'MANIFEST.json')
    for name, h in dict(m['files'], **{'MANIFEST.json':c.sha(FOLDER/'MANIFEST.json')}).items():
        if c.sha(c.local(root/'fix2_source_snapshot', name)) != h:
            raise ValueError('Fix2 snapshot differs '+name)
    result['repair'] = REPAIR
    result['dtype_fields_converted'] = sum(len(x) for x in conversions.values())
    return result

def collect(root, destination):
    from docci_caption_calibration_v24 import collect as original
    old = original.verify; original.verify = verify
    try:
        return original.collect(root, destination)
    finally:
        original.verify = old

def launch(args):
    from docci_caption_calibration_v24.run import parser
    from docci_caption_calibration_v24.launch import invoke
    a = parser().parse_args(args); c.Budget(a.max_minutes); root = c.safe(a.output)
    suffixes = ('.contracts.log', '.fix1_contracts.log', '.fix2_contracts.log',
        '.hf_contracts.log', '.log', '.collect.log')
    logs = [root.with_name(root.name+s) for s in suffixes]
    archive = root.with_name(root.name+'_records.tar.gz')
    occupied = [root, archive, archive.with_name(archive.name+'.contents'),
        archive.with_name(archive.name+'.sha256')]+logs
    if any(p.exists() for p in occupied):
        raise FileExistsError('Existing outputs preserved; choose a NEW Fix2 run name')
    root.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ); env.pop('SUDO_PW', None)
    env.update(CUDA_VISIBLE_DEVICES='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
    start = time.monotonic(); deadline = start+a.max_minutes*60; pids = []; status = 'FAILED'
    try:
        commands = [[sys.executable, '-B', str(c.FOLDER/'test_core.py')],
            [sys.executable, '-B', str(previous.FOLDER/'test_fix.py')],
            [sys.executable, '-B', str(FOLDER/'test_fix.py')],
            [sys.executable, '-B', str(FOLDER/'test_hf.py')]]
        for command, log in zip(commands, logs):
            pids.append(invoke(command, log, env, deadline))
        reserve = min(180, a.max_minutes*60*.15)
        pids.append(invoke([sys.executable, '-B', '-u', str(FOLDER/'fix.py'), 'run']+args,
            logs[4], env, deadline-reserve))
        pids.append(invoke([sys.executable, '-B', '-u', str(FOLDER/'fix.py'), 'collect',
            str(root), str(archive)], logs[5], env, deadline))
        status = 'VERIFIED'
    finally:
        if root.is_dir():
            c.atomic(root/'LAUNCH.json', dict(status=status, owned_PIDs=pids,
                elapsed_seconds=time.monotonic()-start, training_updates=0, semantic_success=False,
                facts='PENDING', automatic_next_stage=False, repair=REPAIR,
                archive=str(archive) if status == 'VERIFIED' else None))
    print('V24 FIX2 '+status+'; send PRIVATE light records; no automatic next stage', flush=True)

def main():
    source_check()
    if len(sys.argv)<2:
        raise SystemExit('Use check | launch | run | verify | collect')
    action, args = sys.argv[1], sys.argv[2:]
    if action == 'launch':
        launch(args)
    elif action == 'run':
        from docci_caption_calibration_v24 import run
        parsed = run.parser().parse_args(args); old = sys.argv
        try:
            sys.argv = [str(c.FOLDER/'run.py')]+args
            with repaired(c.safe(parsed.output)):
                run.main()
        finally:
            sys.argv = old
    elif action == 'verify' and len(args) == 1:
        print(json.dumps(verify(args[0]), ensure_ascii=False, indent=2))
    elif action == 'collect' and len(args) == 2:
        collect(*args)
    elif action == 'check' and not args:
        import unittest
        from docci_caption_calibration_v24_fix2 import test_fix
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_fix))
        if not result.wasSuccessful():
            raise SystemExit(1)
    else:
        raise SystemExit('Unknown action/arguments')

if __name__ == '__main__':
    main()
