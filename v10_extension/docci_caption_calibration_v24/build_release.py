"""Source-only additive USTAR release; no old source changes or data publication."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import tarfile
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_caption_calibration_v24.core import FOLDER, PROJECT, sha, atomic

def main():
    p = argparse.ArgumentParser(); p.add_argument('--manifest-only', action='store_true')
    p.add_argument('--refresh-unreleased', action='store_true'); a = p.parse_args()
    from docci_evidence_v23.core import source_check, FOLDER as V23
    old = source_check(); deps = dict(old['dependencies'])
    for n in list(old['files'])+['MANIFEST.json']:
        deps[(V23/n).relative_to(PROJECT).as_posix()] = sha(V23/n)
    files = {p.name:sha(p) for p in sorted(FOLDER.iterdir()) if p.is_file() and p.name != 'MANIFEST.json'}
    for n in files:
        if n.endswith('.py'): ast.parse((FOLDER/n).read_text(), feature_version=(3,8))
    value = dict(schema='v24-source-release', files=files, dependencies=deps)
    manifest = FOLDER/'MANIFEST.json'; dest = PROJECT/'releases/v10_docci_caption_calibration_v24.tar.gz'
    if manifest.exists() and json.loads(manifest.read_text()) != value:
        if not a.refresh_unreleased or dest.exists(): raise ValueError('Frozen released manifest preserved')
        # This is mechanical generated metadata for an UNRELEASED candidate only.
        manifest.write_text(json.dumps(value,indent=2)+'\n')
    elif not manifest.exists(): atomic(manifest,value)
    if a.manifest_only:
        print('{} sources / {} unchanged dependencies'.format(len(files),len(deps))); return
    if dest.exists() or dest.with_name(dest.name+'.sha256').exists(): raise FileExistsError('Existing release preserved')
    mapping = {'v10_extension/docci_caption_calibration_v24/'+n:FOLDER/n for n in list(files)+['MANIFEST.json']}
    with dest.open('xb') as f, tarfile.open(fileobj=f,mode='w:gz',format=tarfile.USTAR_FORMAT) as tf:
        for n,path in mapping.items():
            info=tarfile.TarInfo(n); info.size=path.stat().st_size; info.mode=0o644
            with path.open('rb') as src: tf.addfile(info,src)
    with tarfile.open(dest,'r:gz') as tf:
        members=tf.getmembers()
        if len(members)!=len(mapping) or {m.name for m in members}!=set(mapping): raise ValueError('Archive matrix differs')
        for m in members:
            if not m.isfile() or hashlib.sha256(tf.extractfile(m).read()).hexdigest()!=sha(mapping[m.name]):
                raise ValueError('Archive byte integrity differs')
    with dest.with_name(dest.name+'.sha256').open('x') as f: f.write(sha(dest)+'  '+dest.name+'\n')
    print(dest); print(sha(dest))

if __name__=='__main__': main()
