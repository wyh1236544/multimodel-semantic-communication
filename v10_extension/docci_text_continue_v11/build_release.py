"""Deterministic source package, real install/restore and atomic-publication data."""
import argparse
import ast
import math
import time
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from install import install
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT)); sys.path.insert(0,str(EXT.parent))
from docci_text_continue_v11.metrics import metrics,diagnostic


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def blob(path):
    data=Path(path).read_bytes(); return hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest()


def regression(folder,records):
    records=Path(records).resolve(strict=True); contract=json.loads((folder/'parent_contract.json').read_text())
    for name,sha in contract['files'].items():
        f=records/name
        if f.is_symlink() or not f.is_file() or digest(f)!=sha: raise ValueError('Approved record regression bytes differ: '+name)
    start=time.monotonic(); counts={}; matched=[]
    for file in sorted(records.glob('*/rows.jsonl')):
        count=0
        for line in file.read_text().splitlines():
            row=json.loads(line)
            if 'source_text' not in row or 'received_caption' not in row: continue
            for key,value in metrics(row['source_text'],row['received_caption']).items():
                if key not in row: continue
                if isinstance(value,(int,float)):
                    if not math.isclose(float(value),float(row[key]),rel_tol=1e-12,abs_tol=1e-12): raise ValueError('Actual old metric differs: '+key)
                elif value!=row[key]: raise ValueError('Actual old metric differs: '+key)
            count+=1
            if file.parent.name=='clean_validation' and row.get('control')=='matched': matched.append(row)
        counts[file.parent.name]=count
    if len(matched)!=100 or counts.get('clean_validation')!=300: raise ValueError('Incomplete actual approved regression')
    d=diagnostic(matched); full=d['slot_groups']['full62']
    if (full['slots'],full['early_EOS'],full['EOS_at_expected_length'],full['capacity_stop'])!=(196,68,62,66): raise ValueError('Corrected EOS regression differs')
    return dict(approved_parent_checkpoint_sha256=contract['checkpoint_sha256'],verified_record_hashes=len(contract['files']),metric_rows_by_folder=counts,
        rows=sum(counts.values()),source_length_diagnostics=d,elapsed_seconds=time.monotonic()-start,scope='Actual historical stored records, not new neural re-inference')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--approved-records',required=True); p.add_argument('--release-suffix',choices=('','fix1'),default=''); args=p.parse_args()
    folder=Path(__file__).resolve().parent; project=folder.parents[1]; release=project/'releases'; release.mkdir(exist_ok=True)
    release_name='v10_docci_text_continue_v11'+('_'+args.release_suffix if args.release_suffix else '')
    archive=release/(release_name+'.tar.gz')
    if archive.exists(): raise FileExistsError('Existing source archive preserved; use a new revision')
    files=sorted(f for f in folder.iterdir() if f.is_file() and f.name!='MANIFEST.json' and f.suffix in ('.py','.json','.md','.txt','.svg'))
    for file in files:
        if file.suffix=='.py': ast.parse(file.read_text(),filename=str(file),feature_version=(3,8))
    parent=project/'v10_extension/docci_reconstruction_v10'; required=dict(json.loads((parent/'MANIFEST.json').read_text())['required_sources'])
    for file in parent.iterdir():
        if file.is_file() and file.suffix in ('.py','.json','.md'): required[str(file.relative_to(project))]=digest(file)
    for name in ('utils/__init__.py','utils/metrics.py','utils/checkpoint.py','utils/config.py'): required[name]=digest(project/name)
    backup=project/'v10_extension/deployment_records/HANDOFF_before_docci_text_continue_v11_20261014_local.md'; handoff=project/'v10_extension/docs/HANDOFF.md'; entry=(folder/'SHARED_HANDOFF_ENTRY.md').read_bytes()
    if not backup.is_file() or handoff.read_bytes()!=backup.read_bytes()+b'\n'+entry: raise ValueError('Exact local old HANDOFF backup/append differs')
    if any(digest(project/name)!=sha for name,sha in required.items()): raise ValueError('Historical prerequisite changed')
    # Test BEFORE creating any final source release artifact.
    for name in ('test_core.py','test_report.py','test_model.py'): subprocess.run([sys.executable,'-B',str(folder/name)],cwd=str(project),check=True)
    historical=regression(folder,args.approved_records)
    manifest=dict(schema='docci-text-continue-v11-source',source_only=True,python_min='3.8',files={f.name:digest(f) for f in files},required_sources=required,
        data_and_pretrained_weights_included=False,local_torch_pretrained_gpu_tests='NOT_EXECUTED',server_cpu_and_authentic_saved_parent_smoke_required=True)
    if (folder/'MANIFEST.json').exists():
        if json.loads((folder/'MANIFEST.json').read_text())!=manifest: raise ValueError('Explicit source manifest certification required; existing archive unchanged')
    else:
        with (folder/'MANIFEST.json').open('x') as f: json.dump(manifest,f,indent=2); f.write('\n')
    published=files+[folder/'MANIFEST.json']
    restore=Path(tempfile.mkdtemp(prefix='docci_text_continue_v11_restore_',dir=str(release))); source=restore/'_v10_docci_text_continue_v11'; source.mkdir()
    for file in published:
        with (source/file.name).open('xb') as f: f.write(file.read_bytes())
    def fixture(label):
        root=restore/label; root.mkdir()
        for name in required:
            dest=root/name; dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f: f.write((project/name).read_bytes())
        file=root/'v10_extension/docs/HANDOFF.md'; file.parent.mkdir(parents=True,exist_ok=True)
        with file.open('xb') as f: f.write(b'# Exact previous user handoff\n')
        return root
    target=fixture('multimodel'); install(source,target); install(source,target,True); install(source,target)
    backups=list((target/'v10_extension/deployment_records').glob('HANDOFF_before_docci_text_continue_v11_*.md'))
    if len(backups)!=1 or backups[0].read_bytes()!=b'# Exact previous user handoff\n': raise ValueError('Exact backup/reinstall failed')
    for name in ('test_core.py','test_report.py'): subprocess.run([sys.executable,'-B',str(target/'v10_extension/docci_text_continue_v11'/name)],cwd=str(target),check=True)
    if any(digest(target/name)!=sha for name,sha in required.items()): raise ValueError('Historical source mutated by install')
    conflict=fixture('conflict'); dest=conflict/'v10_extension/docci_text_continue_v11'; dest.mkdir()
    with (dest/'run.py').open('xb') as f: f.write(b'user-owned source\n')
    try: install(source,conflict)
    except FileExistsError: pass
    else: raise ValueError('User source conflict accepted')
    if (dest/'run.py').read_bytes()!=b'user-owned source\n' or (conflict/'v10_extension/docs/HANDOFF.md').read_bytes()!=b'# Exact previous user handoff\n': raise ValueError('Conflict mutated old data')
    drift=fixture('drift'); key=next(iter(required))
    with (drift/key).open('ab') as f: f.write(b'user drift')
    try: install(source,drift)
    except ValueError: pass
    else: raise ValueError('Prerequisite drift accepted')
    if (drift/'v10_extension/docci_text_continue_v11').exists(): raise ValueError('Drift failure installed anything')
    link=fixture('symlink'); outside=restore/'outside'; outside.mkdir(); (link/'v10_extension/docci_text_continue_v11').symlink_to(outside,target_is_directory=True)
    try: install(source,link)
    except ValueError: pass
    else: raise ValueError('Symlink destination accepted')
    if list(outside.iterdir()): raise ValueError('Symlink target mutated')
    divergent=fixture('divergent-handoff'); file=divergent/'v10_extension/docs/HANDOFF.md'
    with file.open('ab') as f: f.write(b'\n'+entry.splitlines()[0]+b'\nuser alternate entry\n')
    before=file.read_bytes()
    try: install(source,divergent)
    except FileExistsError: pass
    else: raise ValueError('Divergent handoff accepted')
    if file.read_bytes()!=before or (divergent/'v10_extension/docci_text_continue_v11').exists(): raise ValueError('Divergent handoff mutated target')
    with archive.open('xb') as out,gzip.GzipFile(fileobj=out,mode='wb',filename='',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT) as tar:
        for file in published:
            data=file.read_bytes(); member=tarfile.TarInfo('_v10_docci_text_continue_v11/'+file.name); member.size=len(data); member.mode=0o644
            tar.addfile(member,io.BytesIO(data))
    checked=restore/'archive_roundtrip'; checked.mkdir()
    with tarfile.open(archive) as tar:
        expected={'_v10_docci_text_continue_v11/'+f.name for f in published}
        if set(tar.getnames())!=expected or any(not m.isfile() for m in tar.getmembers()): raise ValueError('Unsafe source archive')
        for member in tar.getmembers():
            with (checked/Path(member.name).name).open('xb') as f: f.write(tar.extractfile(member).read())
    if any(digest(checked/f.name)!=digest(f) for f in published): raise ValueError('Archive restored bytes differ')
    sha=digest(archive)
    with archive.with_name(archive.name+'.sha256').open('x') as f: f.write(sha+'  '+archive.name+'\n')
    result=dict(archive=str(archive),archive_sha256=sha,bytes=archive.stat().st_size,source_files=len(published),prerequisite_files=len(required),python38_parse=True,historical_records_regression=historical,
        local_core_tests=8,local_report_tests=11,restore_install_reinstall_conflict_drift_symlink_handoff_backup=True,
        local_torch_transformers_pretrained_GPU_DOCCI='NOT_EXECUTED; unavailable; actual server gates mandatory',restore_directory=str(restore))
    with (release/(release_name+'_build_verification.json')).open('x') as f: json.dump(result,f,indent=2)
    publication=dict(files=[dict(path=str(f.relative_to(project)),content=f.read_text(),git_blob_sha1=blob(f)) for f in published+[backup,handoff]],
        prerequisite_git_blob_sha1={n:blob(project/n) for n in required},previous_handoff_git_blob_sha1=blob(backup),archive_sha256=sha)
    with (release/(release_name+'_publication_payload.json')).open('x',encoding='utf-8') as f: json.dump(publication,f,ensure_ascii=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
