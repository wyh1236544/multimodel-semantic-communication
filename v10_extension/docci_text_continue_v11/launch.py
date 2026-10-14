"""Offline existing environment; bounded owned processes; never retry training."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT)); sys.path.insert(0,str(EXT.parent))
from docci_text_continue_v11.core import FOLDER,PROJECT,load,parent,check_sources,safe_write,read,atomic_json


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('action',choices=('check','smoke','pipeline'))
    p.add_argument('output',nargs='?'); p.add_argument('--phase-config')
    p.add_argument('--parent-run',default=str(PROJECT/read(FOLDER/'parent_contract.json')['default_parent']))
    p.add_argument('--data-root',default=str(PROJECT)); p.add_argument('--bart-path',default=str(PROJECT/'pretrained/bart-base'))
    p.add_argument('--max-hours',type=float,default=4)
    a=p.parse_args(); load(a.phase_config)
    if not 0<a.max_hours<=24: p.error('Finite0..24h budget required')
    if (a.action=='check')!=(a.output is None): p.error('check takes no output; execution requires NEW output')
    exe=os.environ.get('PYTHON_BIN') or sys.executable
    if not Path(exe).is_file(): raise FileNotFoundError('Existing interpreter missing; no install')
    env={k:v for k,v in os.environ.items() if k!='SUDO_PW'}
    env.update(CUDA_VISIBLE_DEVICES='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        PYTHONDONTWRITEBYTECODE='1',V11_REQUIRE_TORCH='1',V10_REQUIRE_TORCH='1',PYTHONPATH=os.pathsep.join([str(EXT),str(PROJECT)]))
    start=time.monotonic()
    def invoke(command,log=None,reserve=0):
        cap=a.max_hours*3600-(time.monotonic()-start)-reserve
        if cap<=0: raise TimeoutError('Overall budget/reserve exhausted')
        stream=log.open('x') if log else None; child=None
        try:
            child=subprocess.Popen(command,cwd=str(PROJECT),env=env,stdout=stream,stderr=subprocess.STDOUT if stream else None)
            print('Owned PID={}{}'.format(child.pid,' log='+str(log) if log else ''),flush=True)
            code=child.wait(timeout=cap)
            if code: raise RuntimeError('Owned child exit {}; preserve logs; no automatic retry'.format(code))
        except BaseException:
            if child is not None and child.poll() is None:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
            raise
        finally:
            if stream: stream.close()
    invoke([exe,'-B',str(FOLDER/'install.py'),'--project-root',str(PROJECT),'--verify-only'])
    check_sources()
    for name in ('test_core.py','test_report.py','test_model.py'): invoke([exe,'-B',str(FOLDER/name)])
    # Inherited real Torch/HF public-receiver contracts also remain mandatory.
    invoke([exe,'-B',str(EXT/'docci_reconstruction_v10/test_model.py')])
    print('CPU contracts passed; actual approved parent/BART/GPU smoke still required.',flush=True)
    if a.action=='check': return
    output=safe_write(a.output)
    if output.relative_to(PROJECT).parts[0]!='experiment_results' or not output.name.startswith('v10_'): raise ValueError('Use project experiment_results/v10_*')
    run_log=output.with_name(output.name+'_run.log')
    if output.exists() or run_log.exists(): raise FileExistsError('Existing output/log preserved; use NEW attempt')
    parent(a.parent_run,True)
    command=[exe,'-B','-u',str(FOLDER/'run.py'),'--output',str(output),'--parent-run',str(Path(a.parent_run).resolve()),
        '--data-root',str(Path(a.data_root).resolve()),'--bart-path',str(Path(a.bart_path).resolve()),
        '--max-hours',str(max(.01,(a.max_hours*3600-(time.monotonic()-start)-600)/3600))]
    if a.phase_config: command+=['--phase-config',str(Path(a.phase_config).resolve())]
    if a.action=='smoke': command+=['--smoke-only']
    invoke(command,run_log,600)
    shutil.copyfile(run_log,output/'run.log')
    if a.action=='smoke':
        print('SMOKE only; no cumulative20epoch run. Main requires a separate NEW pipeline output.',flush=True); return
    archive=output/'v10_docci_text_continue_v11_records.tar.gz'
    # No hard180s cap, no redundant standalone report before collect.
    invoke([exe,'-B','-u',str(FOLDER/'collect.py'),str(output),str(archive)])
    print('Send '+str(archive)+'; large checkpoints stay server.',flush=True)


if __name__=='__main__': main()
