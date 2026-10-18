"""Reuse original bounded pipeline; route only its explicit child entry points."""
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT),str(EXT.parent)]
from docci_text_guided_image_v20.core import safe,read
from docci_text_guided_image_v20_fix1.compat import (FOLDER,ORIGINAL,source_check,
    installed,snapshot)
from docci_text_guided_image_v20_fix1.entry import remove_option

def rewrite(command,gate=None,reuse=None):
    cmd = list(command)
    test = str(ORIGINAL/'test_model.py')
    if test in cmd:
        at = cmd.index(test)
        cmd[at:at+1] = [str(FOLDER/'entry.py'),'test']
        if gate:
            cmd += ['--receipt',str(gate)]
        return cmd
    for name in ('run','report','collect'):
        path = str(ORIGINAL/(name+'.py'))
        if path in cmd:
            at = cmd.index(path)
            cmd[at:at+1] = [str(FOLDER/'entry.py'),name]
            if name=='run' and cmd[at+2]=='cache' and '--smoke' in cmd and reuse:
                cmd += ['--reuse-smoke-cache',str(reuse)]
            return cmd
    return cmd

@contextmanager
def routed(gate,reuse):
    from docci_text_guided_image_v20 import launch
    old_popen,old_snapshot = subprocess.Popen,launch.snapshot
    def popen(cmd,*args,**kwargs):
        return old_popen(rewrite(cmd,gate,reuse),*args,**kwargs)
    def captured(root):
        old_snapshot(root)
        snapshot(root,gate)
    subprocess.Popen,launch.snapshot = popen,captured
    try:
        yield launch
    finally:
        subprocess.Popen,launch.snapshot = old_popen,old_snapshot

def main():
    source_check()
    argv,reuse = remove_option(sys.argv[1:],'--reuse-smoke-cache')
    if not argv or argv[0] not in ('check','smoke','pipeline'):
        raise SystemExit('Use check | smoke NEW_OUTPUT | pipeline NEW_OUTPUT [--reuse-smoke-cache CACHE]')
    gate = None
    if argv[0]!='check':
        if len(argv)<2:
            raise ValueError('NEW output required')
        root = safe(argv[1])
        gate = root.with_name(root.name+'_fix1_gpu_gate.json')
        if root.exists() or gate.exists():
            raise FileExistsError('Existing run/gate preserved; choose NEW output')
    elif reuse:
        raise ValueError('check does not reuse/create a GPU cache')
    if reuse:
        reuse = safe(reuse)
        if not (reuse/'CACHE_COMPLETE.json').is_file() or read(reuse/'index.json').get('smoke') is not True:
            raise ValueError('Require existing completed SMOKE cache')
    sys.argv = [str(ORIGINAL/'launch.py')]+argv
    with installed(),routed(gate,reuse) as launch:
        launch.main()

if __name__=='__main__':
    main()
