"""Explicit child entry: repaired neural execution, unchanged legacy checks."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT),str(EXT.parent)]
from docci_text_guided_image_v20.core import safe,sha,atomic
from docci_text_guided_image_v20_fix1.compat import (FOLDER,ORIGINAL,SCHEMA,
    installed,source_check,reuse_cache,pooling_patch)

def remove_option(argv,name):
    argv = list(argv)
    if name not in argv:
        return argv,None
    at = argv.index(name)
    if argv.count(name)!=1 or at+1>=len(argv):
        raise ValueError('One explicit option value required: '+name)
    value = argv[at+1]
    return argv[:at]+argv[at+2:],value

def tests(receipt=None):
    source_check()
    from docci_text_guided_image_v20 import test_model as legacy
    from docci_text_guided_image_v20_fix1 import test_fix, test_gpu
    if receipt:
        if not test_gpu.READY:
            raise RuntimeError('Existing Torch/HF required; no automatic installation')
        test_gpu.require_cuda()
    suite = unittest.TestSuite()
    for module in (test_fix,legacy):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    # Contract fixtures use their own isolated report wrappers. Only the model
    # alias needs repairing here; do not wrap all fixture reports twice.
    from docci_text_guided_image_v20 import model as model_module
    with pooling_patch(model_module):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful() or result.skipped:
            raise RuntimeError('Required CPU/Torch contracts failed/skipped')
        if receipt:
            gpu = unittest.TextTestRunner(verbosity=2).run(
                unittest.defaultTestLoader.loadTestsFromModule(test_gpu))
            if not gpu.wasSuccessful() or gpu.skipped or gpu.testsRun!=3:
                raise RuntimeError('Required strict CUDA contracts failed/skipped')
            import torch
            value = dict(schema=SCHEMA,repair_manifest_sha256=sha(FOLDER/'MANIFEST.json'),
                CUDA_gate_passed=True,strict_determinism=torch.are_deterministic_algorithms_enabled(),
                GPU_tests_run=gpu.testsRun,GPU_tests_skipped=len(gpu.skipped),
                physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),
                torch_version=torch.__version__,device=torch.cuda.get_device_name(0))
            p = safe(receipt)
            with p.open('x',encoding='utf-8') as f:
                json.dump(value,f,indent=2,allow_nan=False)
                f.write('\n')
            print('FIX1_STRICT_CUDA_GATE_PASSED',flush=True)

def main():
    action,argv = sys.argv[1],sys.argv[2:]
    source_check()
    if action=='test':
        p=argparse.ArgumentParser();p.add_argument('--receipt')
        tests(p.parse_args(argv).receipt)
        return
    if action=='run':
        argv,cache = remove_option(argv,'--reuse-smoke-cache')
        if cache:
            from docci_text_guided_image_v20.run import arguments
            p=argparse.ArgumentParser(allow_abbrev=False)
            p.add_argument('action',choices=('cache',));p.add_argument('output')
            p.add_argument('--smoke',action='store_true');p.add_argument('--max-hours',type=float,required=True)
            arguments(p);a=p.parse_args(argv)
            if not a.smoke:
                raise ValueError('Full cache cannot be relabelled/reused by this repair')
            reuse_cache(cache,a.output,min(a.max_hours,1))
            return
        from docci_text_guided_image_v20 import run
        target = safe(argv[1])
        if target.exists():
            raise FileExistsError('Existing neural stage preserved')
        with installed(neural=True):
            try:
                sys.argv = [str(ORIGINAL/'run.py')]+argv
                run.main()
            finally:
                if target.is_dir():
                    import torch
                    atomic(target/'FIX1_RUNTIME.json',dict(schema=SCHEMA,
                        repair_manifest_sha256=sha(FOLDER/'MANIFEST.json'),
                        strict_determinism=torch.are_deterministic_algorithms_enabled(),
                        module_local_pooling_replaced=True,torch_version=torch.__version__))
        return
    with installed() as verify:
        if action=='report':
            p=argparse.ArgumentParser();p.add_argument('root')
            p.add_argument('--full',action='store_true');p.add_argument('--max-hours',type=float,default=1)
            a=p.parse_args(argv)
            print(json.dumps(verify(a.root,a.full,a.max_hours),indent=2,ensure_ascii=False))
        elif action=='collect':
            from docci_text_guided_image_v20.collect import collect
            p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('destination')
            p.add_argument('--max-minutes',type=float,default=60);a=p.parse_args(argv)
            collect(a.root,a.destination,a.max_minutes)
        else:
            raise ValueError('Unknown repair action')

if __name__=='__main__':
    main()
