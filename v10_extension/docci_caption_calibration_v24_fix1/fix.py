"""Additive V24 metadata repair. No model/config mutation, new decoding or training."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sys
import time
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_caption_calibration_v24 import core as c

FOLDER = Path(__file__).resolve().parent
REPAIR = 'v24-fix1-inner-language-model-generation-config'

def source_check():
    m = c.read(FOLDER/'MANIFEST.json')
    for base, mapping in ((FOLDER,m['files']),(c.PROJECT,m['dependencies'])):
        for name,h in mapping.items():
            if c.sha(c.local(base,name)) != h:
                raise ValueError('Fix1 pinned source differs '+name)
    c.source_check()
    return m

class ModelView:
    """Only the metadata read is redirected. Bound generate stays on original model."""
    def __init__(self, model):
        self._wrapped = model._wrapped if isinstance(model, ModelView) else model
    @property
    def generation_config(self):
        lm = getattr(self._wrapped, 'language_model', None)
        value = getattr(lm, 'generation_config', None)
        if value is None:
            raise ValueError('Actual language_model.generation_config missing; do not invent EOS/defaults')
        return value
    def __getattr__(self, name):
        return getattr(self._wrapped, name)

def model_view(model):
    value = model if isinstance(model, ModelView) else ModelView(model)
    cfg = value.generation_config
    if getattr(cfg, 'eos_token_id', None) is None:
        raise ValueError('Actual language model has no EOS definition')
    return value

@contextmanager
def repaired():
    """Process-local binding scope; original V24 files and Torch globals untouched."""
    from docci_caption_calibration_v24 import runtime, run
    old_load, old_generate, old_complete = runtime.load_model, runtime.generate_one, run.complete
    observations = []
    def load(*args, **kwargs):
        model, processor, info = old_load(*args, **kwargs)
        view = model_view(model)
        observations.append(dict(outer_generation_config_is_none=getattr(model,'generation_config',None) is None,
            generation_config_owner='language_model',
            language_model_generation_config=view.generation_config.to_dict(),
            model_config_mutated=False, generation_method_replaced=False))
        return view, processor, info
    def generate(model, *args, **kwargs):
        return old_generate(model_view(model), *args, **kwargs)
    def complete(root):
        source_check()
        if not observations:
            raise ValueError('Actual fixed model load provenance missing')
        root = Path(root)
        snapshot = root/'fix1_source_snapshot'; snapshot.mkdir()
        m = c.read(FOLDER/'MANIFEST.json')
        for name in list(m['files'])+['MANIFEST.json']:
            shutil.copyfile(FOLDER/name,snapshot/name)
        c.atomic(root/'FIX1.json', dict(repair=REPAIR,
            repair_manifest_sha256=c.sha(FOLDER/'MANIFEST.json'),
            original_v24_manifest_sha256=c.sha(c.FOLDER/'MANIFEST.json'),
            observations=observations, original_sources_modified=False,
            generation_kwargs_changed=False, model_weights_changed=False,
            quality_rules_changed=False))
        old_complete(root)
    runtime.load_model, runtime.generate_one, run.complete = load, generate, complete
    try:
        yield
    finally:
        runtime.load_model, runtime.generate_one, run.complete = old_load, old_generate, old_complete

def verify(root):
    from docci_caption_calibration_v24.report import verify as old_verify
    result = old_verify(root)
    root = Path(root); info = c.read(root/'FIX1.json')
    if (info['repair'] != REPAIR or info['repair_manifest_sha256'] != c.sha(FOLDER/'MANIFEST.json') or
            info['original_v24_manifest_sha256'] != c.sha(c.FOLDER/'MANIFEST.json') or
            any(info[k] is not False for k in ('original_sources_modified','generation_kwargs_changed',
                'model_weights_changed','quality_rules_changed')) or not info['observations']):
        raise ValueError('Fix1 provenance differs')
    for obs in info['observations']:
        if (obs['generation_config_owner'] != 'language_model' or obs['model_config_mutated'] is not False or
                obs['generation_method_replaced'] is not False or
                obs['language_model_generation_config'].get('eos_token_id') is None):
            raise ValueError('Actual generation configuration owner differs')
    m = c.read(FOLDER/'MANIFEST.json')
    for name,h in dict(m['files'], **{'MANIFEST.json':c.sha(FOLDER/'MANIFEST.json')}).items():
        if c.sha(c.local(root/'fix1_source_snapshot',name)) != h:
            raise ValueError('Fix1 runtime snapshot differs '+name)
    model = c.read(root/'MODEL.json')
    if model['generation_config'] != info['observations'][-1]['language_model_generation_config']:
        raise ValueError('MODEL metadata is not the actual language model configuration')
    result['repair'] = REPAIR
    return result

def collect(root, destination):
    from docci_caption_calibration_v24 import collect as old
    original = old.verify
    old.verify = verify
    try:
        return old.collect(root,destination)
    finally:
        old.verify = original

def launch(args):
    from docci_caption_calibration_v24.run import parser
    from docci_caption_calibration_v24.launch import invoke
    a = parser().parse_args(args); c.Budget(a.max_minutes)
    root = c.safe(a.output)
    suffixes = ('.contracts.log','.fix1_contracts.log','.hf_contracts.log','.log','.collect.log')
    logs = [root.with_name(root.name+s) for s in suffixes]
    archive = root.with_name(root.name+'_records.tar.gz')
    occupied = [root,archive,archive.with_name(archive.name+'.contents'),archive.with_name(archive.name+'.sha256')]+logs
    if any(p.exists() for p in occupied):
        raise FileExistsError('Preserve old outputs/logs; use NEW fix1 run name')
    root.parent.mkdir(parents=True,exist_ok=True)
    env = dict(os.environ);env.pop('SUDO_PW',None)
    env.update(CUDA_VISIBLE_DEVICES='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    start=time.monotonic();deadline=start+a.max_minutes*60;pids=[];status='FAILED'
    try:
        commands = [[sys.executable,'-B',str(c.FOLDER/'test_core.py')],
            [sys.executable,'-B',str(FOLDER/'test_fix.py')],
            [sys.executable,'-B',str(FOLDER/'test_hf.py')]]
        for command,log in zip(commands,logs):
            pids.append(invoke(command,log,env,deadline))
        reserve=min(180,a.max_minutes*60*.15)
        pids.append(invoke([sys.executable,'-B','-u',str(FOLDER/'fix.py'),'run']+args,logs[3],env,deadline-reserve))
        pids.append(invoke([sys.executable,'-B','-u',str(FOLDER/'fix.py'),'collect',str(root),str(archive)],logs[4],env,deadline))
        status='VERIFIED'
    finally:
        if root.is_dir():
            c.atomic(root/'LAUNCH.json',dict(status=status,owned_PIDs=pids,elapsed_seconds=time.monotonic()-start,
                training_updates=0,semantic_success=False,facts='PENDING',automatic_next_stage=False,
                repair=REPAIR,repair_manifest_sha256=c.sha(FOLDER/'MANIFEST.json'),
                archive=str(archive) if status=='VERIFIED' else None))
    print('V24 FIX1 '+status+'; send PRIVATE light records; no automatic next stage',flush=True)

def main():
    source_check()
    if len(sys.argv)<2: raise SystemExit('Use check | launch | run | collect | verify')
    action,args=sys.argv[1],sys.argv[2:]
    if action=='launch': launch(args)
    elif action=='run':
        from docci_caption_calibration_v24 import run
        old_argv=sys.argv
        try:
            sys.argv=[str(c.FOLDER/'run.py')]+args
            with repaired(): run.main()
        finally:sys.argv=old_argv
    elif action=='verify' and len(args)==1:
        print(json.dumps(verify(args[0]),ensure_ascii=False,indent=2))
    elif action=='collect' and len(args)==2: collect(*args)
    elif action=='check' and not args:
        import unittest
        import test_fix
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_fix))
        if not result.wasSuccessful():raise SystemExit(1)
    else:raise SystemExit('Unknown action/arguments')

if __name__=='__main__':main()
