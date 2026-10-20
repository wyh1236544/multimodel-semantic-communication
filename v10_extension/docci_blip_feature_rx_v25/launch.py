"""Small smoke first; pilot requires independently verified smoke, never automatic."""
import argparse
import os
from pathlib import Path
import sys
import time
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT),str(EXT.parent)]
from docci_blip_feature_rx_v25.core import (FOLDER,SCHEMA,sha,read,safe,atomic,source_check,Budget)

def main():
    from docci_blip_feature_rx_v25.run import parser as worker_parser
    from docci_caption_calibration_v24.launch import invoke
    from docci_blip_feature_rx_v25.report import verify_pipeline
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('action',choices=('inventory','smoke','pilot'))
    p.add_argument('output'); p.add_argument('--max-hours',type=float,default=2.)
    p.add_argument('--smoke-root')
    # Keep explicit offline paths; validation is delegated to worker parser.
    known,extra = p.parse_known_args(); source_check(); Budget(known.max_hours)
    if any(x.split('=',1)[0] in ('--profile','--cache','--arm','--training-root','--max-hours') for x in extra):
        raise ValueError('Launcher owns stage/profile/cache/arm/budget arguments')
    worker_parser().parse_args(['inventory',known.output]+extra)
    if known.action=='pilot':
        if not known.smoke_root: raise ValueError('Explicit verified actual --smoke-root required')
        result = verify_pipeline(known.smoke_root,True)
        if result['cache']['profile']!='smoke': raise ValueError('Not a smoke run')
        if read(Path(known.smoke_root)/'LAUNCH_FINISHED.json')['status']!='VERIFIED':
            raise ValueError('Actual smoke launch/collection not finished')
    root = safe(known.output)
    if root.exists(): raise FileExistsError('Existing output preserved; choose a NEW run name')
    root.mkdir(parents=True)
    env = dict(os.environ); env.pop('SUDO_PW',None)
    env.update(CUDA_VISIBLE_DEVICES='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    start = time.monotonic(); deadline = start+known.max_hours*3600; pids = []; status = 'FAILED'
    profile = 'pilot' if known.action=='pilot' else 'smoke'
    def child(label,cmd,reserve=0):
        pids.append(invoke(cmd,root/(label+'.log'),env,deadline-reserve))
    def worker(stage,folder,more=None):
        child(folder,[sys.executable,'-B','-u',str(FOLDER/'run.py'),stage,str(root/folder),
            '--profile',profile,'--max-hours',str(known.max_hours)]+extra+(more or []),120)
    try:
        child('contracts',[sys.executable,'-B',str(FOLDER/'test_core.py')])
        if known.action=='inventory': worker('inventory','inventory')
        else:
            # Mandatory actual tiny HF generation/FP16 metadata and new feature/adapter tests.
            child('v24_hf_contracts',[sys.executable,'-B',str(FOLDER.parent/'docci_caption_calibration_v24_fix2/test_hf.py')])
            child('hf_contracts',[sys.executable,'-B',str(FOLDER/'test_hf.py')])
            worker('cache','cache')
            child('cache_verify',[sys.executable,'-B',str(FOLDER/'report.py'),'cache',str(root/'cache')])
            for arm in ('self','visual'):
                worker('train','train_'+arm,['--arm',arm,'--cache',str(root/'cache')])
                child('train_'+arm+'_verify',[sys.executable,'-B',str(FOLDER/'report.py'),'train',str(root/('train_'+arm)),
                    '--cache',str(root/'cache')])
            worker('evaluation','evaluation',['--cache',str(root/'cache'),'--training-root',str(root)])
            child('evaluation_verify',[sys.executable,'-B',str(FOLDER/'report.py'),'evaluation',str(root/'evaluation'),
                '--cache',str(root/'cache')])
        status = 'INVENTORY_COMPLETE' if known.action=='inventory' else 'COMPLETE'
    finally:
        atomic(root/'PIPELINE.json',dict(schema=SCHEMA,status=status,profile=profile,
            source_manifest_sha256=sha(FOLDER/'MANIFEST.json'),owned_PIDs=pids,
            elapsed_seconds=time.monotonic()-start,automatic_next_stage=False,
            facts='PENDING',semantic_success=False,smoke_root=known.smoke_root))
    if status=='COMPLETE':
        child('final_verify',[sys.executable,'-B',str(FOLDER/'report.py'),'pipeline',str(root)])
        child('collect',[sys.executable,'-B',str(FOLDER/'collect.py'),str(root),str(root.with_name(root.name+'_records.tar.gz'))])
        archive = root.with_name(root.name+'_records.tar.gz')
        checksum = archive.with_name(archive.name+'.sha256')
        if not archive.is_file() or not checksum.is_file() or checksum.read_text().split()[0]!=sha(archive):
            raise ValueError('Collector returned without the verified actual archive')
        atomic(root/'LAUNCH_FINISHED.json',dict(status='VERIFIED',archive=str(root.with_name(root.name+'_records.tar.gz')),
            elapsed_seconds=time.monotonic()-start,automatic_next_stage=False,semantic_success=False,facts='PENDING'))
    print('V25 '+status+'; keep arrays/weights on server; no automatic next stage',flush=True)
if __name__=='__main__': main()
