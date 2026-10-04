"""Persist bounded NumPy test evidence. Does not import/run a neural checkpoint."""
import argparse
import ast
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    folder=Path(__file__).resolve().parent; root=folder.parents[1]
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('output'); a=p.parse_args()
    sys.path.insert(0,str(folder.parent)); from v10.paths import write_path
    out=write_path(a.output,root)
    if out.exists(): raise FileExistsError('Preserve previous verification')
    out.mkdir(parents=True)
    suites=['optical_realism_v2/test_dsp.py','optical_realism_v2/test_core.py',
            'optical_link_v1/test_core.py','final_compare_v1/test_core.py','physical_link/test_core.py']
    results=[]
    for name in suites:
        started=time.monotonic()
        done=subprocess.run([sys.executable,'-B',str(folder.parent/name)],cwd=str(root),
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        (out/(name.replace('/','_')+'.log')).write_text(done.stdout,encoding='utf-8')
        results.append(dict(suite=name,exit_code=done.returncode,wall_seconds=time.monotonic()-started))
        if done.returncode: raise RuntimeError('Test failed: '+name)
    files=sorted(folder.glob('*.py'))
    for path in files: ast.parse(path.read_text(),filename=str(path),feature_version=(3,8))
    record=dict(suites=results,python=sys.version,python38_ast_files=len(files),
        torch_available=importlib.util.find_spec('torch') is not None,
        neural_checkpoint_GPU_COCO_executed=False,hardware_calibrated=False)
    (out/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__': main()
