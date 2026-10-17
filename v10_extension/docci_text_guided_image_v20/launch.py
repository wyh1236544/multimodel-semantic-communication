"""Single sequential GPU1 pipeline, real smoke first, finite owned-process budget."""
import argparse
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT),str(EXT.parent)]
from docci_text_guided_image_v20.core import FOLDER,PROJECT,ARMS,load,safe,atomic,check_source
from docci_text_guided_image_v20.run import arguments

def stop_owned(child):
    if child.poll() is not None:
        return
    if os.getpgid(child.pid)!=child.pid:
        raise RuntimeError('Unexpected child process-group ownership')
    os.killpg(child.pid,signal.SIGTERM)
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if child.poll() is None and os.getpgid(child.pid)==child.pid:
            os.killpg(child.pid,signal.SIGKILL)
        child.wait()

def snapshot(root):
    m = check_source()
    root.mkdir(parents=True,exist_ok=False)
    for base,files,label in ((FOLDER,list(m['files'])+['MANIFEST.json'],'source_snapshot'),
                             (PROJECT,list(m['dependencies']),'dependency_snapshot')):
        for name in files:
            p = root/label/name
            p.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(base/name,p)

def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('action',choices=('check','smoke','pipeline'))
    p.add_argument('output',nargs='?')
    p.add_argument('--max-hours',type=float,default=48)
    arguments(p)
    a = p.parse_args()
    if (a.action=='check')!=(a.output is None) or not math.isfinite(a.max_hours) or not .5<=a.max_hours<=48:
        p.error('NEW output required except check; finite 0.5..48h')
    c = load(a.config)
    check_source()
    root = safe(a.output) if a.output else None
    if root and root.exists():
        raise FileExistsError('Existing output preserved; choose NEW run')
    env = {k:v for k,v in os.environ.items() if k!='SUDO_PW'}
    env.update(CUDA_VISIBLE_DEVICES='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',V20_REQUIRE_TORCH='1',
        V20_LPIPS_ASSETS=str(Path(a.lpips_assets).resolve()),
        PYTHONPATH=os.pathsep.join((str(EXT),str(PROJECT))))
    exe = os.environ.get('PYTHON_BIN') or sys.executable
    deadline = time.monotonic()+a.max_hours*3600
    reserve = 1800 if a.action=='pipeline' else 180
    def invoke(command,log=None,collection=False):
        remaining = deadline-time.monotonic()-(0 if collection else reserve)
        if remaining<=0:
            raise TimeoutError('Remaining budget reserved for independent collection')
        handle = log.open('x',encoding='utf-8') if log else None
        child = None
        try:
            child = subprocess.Popen(command,cwd=str(PROJECT),env=env,stdout=handle,
                stderr=subprocess.STDOUT if handle else None,start_new_session=True)
            print('Owned PID={}{}'.format(child.pid,' log='+str(log) if log else ''),flush=True)
            if child.wait(timeout=remaining):
                raise RuntimeError('Owned stage failed; preserve logs; no auto retry')
        finally:
            if child is not None and child.poll() is None:
                stop_owned(child)
            if handle:
                handle.close()
    for test in ('test_core.py','test_storage.py','test_model.py'):
        invoke([exe,'-B',str(FOLDER/test)])
    print('CPU contracts passed; real saved parents/backbones/GPU still require smoke. No dependency changes.',flush=True)
    if a.action=='check':
        return
    if shutil.disk_usage(PROJECT).free<c['minimum_disk_gib']*2**30:
        raise RuntimeError('Insufficient project disk for fresh joint features/epoch checkpoints')
    snapshot(root)
    atomic(root/'config.json',c)
    stages = []
    for prefix in (['smoke_'] if a.action=='smoke' else ['smoke_','']):
        stages.append(prefix+'cache')
        for arm in (ARMS if prefix else ('B_tx_text','C_tx_rx_text','A_self')):
            stages.extend([prefix+'train_'+arm,prefix+'eval_'+arm])
    state = dict(status='RUNNING',stages={s:'NOT_RUN' for s in stages},max_hours=a.max_hours,
        no_other_process_control=True,automatic_retry=False,FSO='NOT_RUN',bidirectional='NOT_IMPLEMENTED')
    atomic(root/'PIPELINE.json',state)
    failure = None
    try:
        for label in stages:
            prefix = 'smoke_' if label.startswith('smoke_') else ''
            stage = label[len(prefix):]
            action = 'cache' if stage=='cache' else 'train' if stage.startswith('train_') else 'evaluate'
            command = [exe,'-B','-u',str(FOLDER/'run.py'),action,str(root/label),
                '--max-hours',str((deadline-time.monotonic()-reserve)/3600),
                '--config',str(root/'config.json')]
            for name in ('data_root','bart_path','vae_path','longclip_path','clip_tokenizer_path',
                         'lpips_assets','image_checkpoint','text_checkpoint','image_parent_metadata','text_parent_metadata'):
                command.extend(['--'+name.replace('_','-'),str(getattr(a,name))])
            if prefix:
                command.append('--smoke')
            if action!='cache':
                arm = stage[len('train_' if action=='train' else 'eval_'):]
                command.extend(['--cache',str(root/(prefix+'cache')),'--arm',arm])
                if action=='evaluate':
                    command.extend(['--checkpoint',str(root/(prefix+'train_'+arm)/'last.pth')])
            state['stages'][label] = 'RUNNING'
            atomic(root/'PIPELINE.json',state)
            invoke(command,root/(label+'.log'))
            verifier = [exe,'-B','-u',str(FOLDER/'report.py'),str(root),'--max-hours','1']
            # Actual array hashes are mandatory BEFORE the first training stage.
            if action=='cache':
                verifier.append('--full')
            invoke(verifier,root/(label+'_verify.log'))
            state['stages'][label] = 'VERIFIED'
            atomic(root/'PIPELINE.json',state)
        state['status'] = 'SMOKE_COMPLETE' if a.action=='smoke' else 'COMPLETE'
    except (TimeoutError,subprocess.TimeoutExpired) as exc:
        state['status'] = 'PARTIAL_BUDGET'
        failure = exc
    except BaseException as exc:
        state['status'] = 'FAILED'
        failure = exc
    for k,v in state['stages'].items():
        if v=='RUNNING':
            state['stages'][k] = 'UNVERIFIED_PARTIAL'
    if failure:
        state['exception'] = dict(type=type(failure).__name__,message=str(failure))
    atomic(root/'PIPELINE.json',state)
    try:
        invoke([exe,'-B','-u',str(FOLDER/'collect.py'),str(root),
                str(root.with_name(root.name+'_records.tar.gz'))],root/'collection.log',True)
    except BaseException as exc:
        atomic(root/'COLLECTION_FAILED.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    print('V20 '+state['status']+'; send PRIVATE light records, leave arrays/weights on server.',flush=True)
    if failure:
        raise RuntimeError('Partial/failed run preserved; no automatic restart')

if __name__=='__main__':
    main()
