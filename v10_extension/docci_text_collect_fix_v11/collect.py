"""Verify original records, copy light bytes to NEW output, report and pack on CPU."""
import argparse
import gzip
import io
import json
import math
from pathlib import Path
import signal
import sys
import tarfile
import time
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT)); sys.path.insert(0,str(EXT.parent))
from docci_text_collect_fix_v11.engine import FOLDER,PROJECT,sources,verify,legacy,POLICY
from docci_text_continue_v11.core import safe_write,atomic_json,read,digest
ALLOWED={'.json','.jsonl','.md','.py','.txt','.html','.npz','.log','.svg','.png'}


def inputs(root):
    selected=[]; derived={'summary.json','actual_text_recovery.html','COMPLETE.json'}
    for f in sorted(root.rglob('*')):
        if f.is_symlink(): raise ValueError('Symlink input preserved, not followed')
        if f.is_file() and f.suffix in ALLOWED and '.cache' not in f.parts:
            name=str(f.relative_to(root))
            if name not in derived: selected.append(dict(path=name,sha256=digest(f),bytes=f.stat().st_size))
    if not selected or sum(r['bytes'] for r in selected)>512*2**20: raise ValueError('Unexpected light record budget')
    return selected


def copy_records(root,output,files):
    for i,r in enumerate(files):
        source=safe_write(root/r['path']); destination=safe_write(output/r['path']); destination.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as src,destination.open('xb') as dst: dst.write(src.read())
        if digest(source)!=r['sha256'] or digest(destination)!=r['sha256']: raise ValueError('Source/copy changed: '+r['path'])
        if (i+1)%100==0: print('COPY {}/{} unchanged light files'.format(i+1,len(files)),flush=True)


def pack(root,destination):
    partial=destination.with_name(destination.name+'.part')
    if destination.exists() or partial.exists(): raise FileExistsError('Existing archive/partial preserved')
    files=[f for f in sorted(root.rglob('*')) if f.is_file() and f.suffix in ALLOWED]
    with partial.open('xb') as stream,gzip.GzipFile(fileobj=stream,mode='wb',filename='',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT) as tar:
        for i,f in enumerate(files):
            if f.is_symlink(): raise ValueError('Symlink output rejected')
            data=f.read_bytes(); info=tarfile.TarInfo('v10_docci_text_continue_v11_records/'+str(f.relative_to(root))); info.size=len(data); info.mode=0o644
            tar.addfile(info,io.BytesIO(data))
            if (i+1)%100==0: print('PACK {}/{} light files'.format(i+1,len(files)),flush=True)
    partial.rename(destination)
    return digest(destination)


def collect(root,output):
    root=safe_write(root).resolve(strict=True); output=safe_write(output)
    if output.relative_to(PROJECT).parts[0]!='experiment_results' or not output.name.startswith('v10_'): raise ValueError('Use NEW experiment_results/v10_* output')
    if output==root or root in output.parents or output in root.parents: raise ValueError('Recovery must be separate from original input')
    if output.exists(): raise FileExistsError('Existing recovery preserved; use recollect03')
    source_manifest=sources(); started=time.monotonic()
    print('CPU-only recovery: no training, no GPU, no checkpoint pickle loading',flush=True)
    files=inputs(root)
    print('VERIFY full original matrices/frames/history/source + bounded FP32 aggregates',flush=True)
    result=verify(root)
    # Reading/hashing is the only operation on the original run, including saved weights.
    if any(digest(root/r['path'])!=r['sha256'] for r in files): raise ValueError('Original light records changed during verification')
    output.mkdir(parents=True,exist_ok=False)
    try:
        copy_records(root,output,files)
        snapshot=output/'verification_fix/source_snapshot'; snapshot.mkdir(parents=True)
        for name in list(source_manifest['files'])+['MANIFEST.json']:
            with (snapshot/name).open('xb') as f: f.write((FOLDER/name).read_bytes())
        audit=result['aggregation_audit']
        receipt=dict(schema='docci-text-collect-fix-v11',original_run=str(root),same_neural_run_NOT_new_training=True,
            original_run_identity_sha256=digest(root/'run_identity.json'),original_light_files=files,
            original_raw_files_NOT_rewritten=True,raw_aggregates_NOT_replaced=True,
            numeric_policy=POLICY,numeric_rule_change_only='five post-hoc prefix weighted means; other legacy comparisons unchanged',
            quality_gates_unchanged=True,training_or_GPU_started=False,source_manifest_sha256=digest(FOLDER/'MANIFEST.json'),
            aggregation_checks=len(audit),old_strict_rejections=sum(not r['old_strict_check_passed'] for r in audit),aggregation_audit=audit)
        atomic_json(output/'verification_fix/receipt.json',receipt)
        legacy.finish(output,result)
        if any(digest(root/r['path'])!=r['sha256'] for r in files): raise ValueError('Original evidence changed after reporting')
        print('LEGACY QUALITY '+read(output/'summary.json')['legacy_quality_status']+' (unchanged rule)',flush=True)
        archive=output/'v10_docci_text_continue_v11_records_collect02.tar.gz'; sha=pack(output,archive)
        atomic_json(output/'RECOVERY_COMPLETE.json',dict(schema='docci-text-collect-fix-v11',archive_sha256=sha,
            archive=str(archive),source_manifest_sha256=receipt['source_manifest_sha256'],receipt_sha256=digest(output/'verification_fix/receipt.json'),
            wall_seconds=time.monotonic()-started,engineering_completion_NOT_quality=True))
        print('Send '+str(archive),flush=True)
    except BaseException as exc:
        atomic_json(output/'RECOVERY_FAILED.json',dict(exception=type(exc).__name__,message=str(exc)))
        raise


def main():
    p=argparse.ArgumentParser(allow_abbrev=False); p.add_argument('root'); p.add_argument('new_output'); p.add_argument('--max-minutes',type=float,default=30)
    a=p.parse_args()
    if not math.isfinite(a.max_minutes) or not 1<=a.max_minutes<=120: p.error('Finite 1..120min CPU-only cap required')
    def timeout(signum,frame): raise TimeoutError('Owned recovery time budget reached; source unchanged')
    previous=signal.signal(signal.SIGALRM,timeout); signal.setitimer(signal.ITIMER_REAL,a.max_minutes*60)
    try: collect(a.root,a.new_output)
    finally: signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous)


if __name__=='__main__': main()
