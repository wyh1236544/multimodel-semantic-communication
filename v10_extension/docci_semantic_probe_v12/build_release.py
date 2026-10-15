"""Build source-only USTAR after CPU/historical/install/restore negative tests."""
import argparse
import ast
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT)); sys.path.insert(0,str(EXT.parent))
from docci_semantic_probe_v12.install import install
from docci_semantic_probe_v12.report import text_evaluation


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def blob(path):
    data=Path(path).read_bytes(); return hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--approved-text-records',required=True); p.add_argument('--check-only',action='store_true'); a=p.parse_args()
    folder=Path(__file__).resolve().parent; project=folder.parents[1]; release=project/'releases'; release.mkdir(exist_ok=True)
    archive=release/'v10_docci_semantic_probe_v12.tar.gz'
    if not a.check_only and (archive.exists() or (folder/'MANIFEST.json').exists()): raise FileExistsError('Existing certified release preserved')
    files=sorted(f for f in folder.iterdir() if f.is_file() and f.name!='MANIFEST.json' and f.suffix in ('.py','.json','.md'))
    for f in files:
        if f.suffix=='.py': ast.parse(f.read_text(),filename=str(f),feature_version=(3,8))
    for name in ('test_core.py','test_report.py','test_model.py'):
        subprocess.run([sys.executable,'-B',str(folder/name)],cwd=str(project),check=True)
    approved=Path(a.approved_text_records).resolve(strict=True)
    start=time.monotonic(); data=json.loads((approved/'data_identity.json').read_text())
    receipts={r['example_id']:r for r in data['validation']['records']}
    historical=text_evaluation(approved/'g20480_validation',receipts,'validation')
    if len(historical)!=300: raise ValueError('Expected complete actual V11 100x3 regression')
    regression=dict(rows=len(historical),descriptions=len(receipts),frame_metric_public_layout_and_donor_verification=True,
        scope='actual historical saved records; NOT new model inference',seconds=time.monotonic()-start)
    required={}
    for module in ('docci_text_continue_v11','docci_text_collect_fix_v11'):
        parent=project/'v10_extension'/module; m=json.loads((parent/'MANIFEST.json').read_text())
        for name,sha in m['required_sources'].items():
            if name in required and required[name]!=sha: raise ValueError('Conflicting legacy source pin')
            required[name]=sha
        for file in parent.iterdir():
            if file.is_file() and file.suffix in ('.py','.json','.md'): required[str(file.relative_to(project))]=digest(file)
    if any(digest(project/name)!=sha for name,sha in required.items()): raise ValueError('Historical prerequisite drift')
    manifest=dict(schema='docci-semantic-probe-v12-source',source_only=True,python_min='3.8',
        files={f.name:digest(f) for f in files},required_sources=required,data_or_weights_included=False,
        local_Torch_HF_GPU_DOCCI='NOT_EXECUTED',server_authentic_saved_parent_512_smoke_mandatory=True)
    restore=Path(tempfile.mkdtemp(prefix='docci_semantic_probe_v12_restore_',dir=str(release)))
    source=restore/'_v10_docci_semantic_probe_v12'; source.mkdir()
    for f in files: shutil.copyfile(f,source/f.name)
    with (source/'MANIFEST.json').open('x') as f: json.dump(manifest,f,indent=2); f.write('\n')
    entry=(folder/'SHARED_HANDOFF_ENTRY.md').read_bytes()
    def fixture(name):
        root=restore/name; root.mkdir(); (root/'releases').mkdir()
        for path in required:
            target=root/path; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(project/path,target)
        target=root/'v10_extension/docs/HANDOFF.md'; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(b'# Previous exact handoff\n')
        return root
    target=fixture('successful'); install(source,target); install(source,target,True); install(source,target)
    backups=list((target/'v10_extension/deployment_records').glob('HANDOFF_before_docci_semantic_probe_v12_*.md'))
    if len(backups)!=1 or backups[0].read_bytes()!=b'# Previous exact handoff\n': raise ValueError('Exact backup/reinstall differs')
    if (target/'v10_extension/docs/HANDOFF.md').read_bytes()!=backups[0].read_bytes()+b'\n'+entry: raise ValueError('Handoff append differs')
    for name in ('test_core.py','test_report.py'):
        subprocess.run([sys.executable,'-B',str(target/'v10_extension/docci_semantic_probe_v12'/name)],cwd=str(target),check=True)
    for kind in ('conflict','drift','symlink','handoff'):
        root=fixture(kind); old=(root/'v10_extension/docs/HANDOFF.md').read_bytes()
        if kind=='conflict':
            d=root/'v10_extension/docci_semantic_probe_v12'; d.mkdir(); (d/'run.py').write_bytes(b'user source\n')
        elif kind=='drift':
            f=root/next(iter(required)); f.write_bytes(f.read_bytes()+b'\n# user drift\n')
        elif kind=='symlink':
            outside=restore/'outside'; outside.mkdir(); (root/'v10_extension/docci_semantic_probe_v12').symlink_to(outside,target_is_directory=True)
        else:
            old+=b'\n'+entry.splitlines()[0]+b'\nDivergent user entry\n'; (root/'v10_extension/docs/HANDOFF.md').write_bytes(old)
        try: install(source,root)
        except (ValueError,FileExistsError): pass
        else: raise ValueError('Unsafe '+kind+' accepted')
        if (root/'v10_extension/docs/HANDOFF.md').read_bytes()!=old: raise ValueError('Failed preflight mutated handoff')
        if kind=='conflict' and (root/'v10_extension/docci_semantic_probe_v12/run.py').read_bytes()!=b'user source\n': raise ValueError('User source overwritten')
        if kind=='symlink' and list(outside.iterdir()): raise ValueError('Symlink target changed')
    if any(digest(target/name)!=sha for name,sha in required.items()): raise ValueError('Installer changed legacy source')
    result=dict(source_files=len(files)+1,prerequisite_files=len(required),python38_parse=True,
        local_CPU_tests=16,local_real_Torch_HF_tests_passed=0,local_real_Torch_HF_tests_skipped=6,
        actual_historical_saved_record_regression=regression,restore_install_reinstall_conflict_drift_symlink_handoff=True,
        restore_directory=str(restore),GPU_pretrained_DOCCI_quality_speed='NOT_EXECUTED')
    if a.check_only:
        print(json.dumps(result,indent=2)); return
    # Only now certify immutable source bytes and create final distributable artifacts.
    shutil.copyfile(source/'MANIFEST.json',folder/'MANIFEST.json')
    published=files+[folder/'MANIFEST.json']
    with archive.open('xb') as out,gzip.GzipFile(fileobj=out,mode='wb',filename='',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT) as tar:
        for f in published:
            data=f.read_bytes(); member=tarfile.TarInfo('_v10_docci_semantic_probe_v12/'+f.name); member.size=len(data); member.mode=0o644
            tar.addfile(member,io.BytesIO(data))
    with tarfile.open(archive) as tar:
        if set(tar.getnames())!={'_v10_docci_semantic_probe_v12/'+f.name for f in published} or any(not m.isfile() for m in tar.getmembers()): raise ValueError('Archive unsafe')
        for member in tar.getmembers():
            if hashlib.sha256(tar.extractfile(member).read()).hexdigest()!=digest(folder/Path(member.name).name): raise ValueError('Archive roundtrip differs')
    sha=digest(archive)
    with archive.with_name(archive.name+'.sha256').open('x') as f: f.write(sha+'  '+archive.name+'\n')
    result.update(archive=str(archive),archive_sha256=sha,bytes=archive.stat().st_size)
    with (release/'v10_docci_semantic_probe_v12_build_verification.json').open('x') as f: json.dump(result,f,indent=2)
    payload=dict(files=[dict(path=str(f.relative_to(project)),content=f.read_text(),git_blob_sha1=blob(f)) for f in published],
        required_git_blob_sha1={name:blob(project/name) for name in required},archive_sha256=sha)
    with (release/'v10_docci_semantic_probe_v12_publication_payload.json').open('x',encoding='utf-8') as f: json.dump(payload,f,ensure_ascii=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
