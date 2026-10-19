"""Source-only reproducible layout and immutable dependency pins; no weights."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import tarfile
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_pretrained_checks_v22.core import FOLDER, PROJECT, sha

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest-only', action='store_true')
    p.add_argument('--refresh-unreleased', action='store_true')
    a = p.parse_args()
    from docci_text_guided_image_v20_fix1.compat import source_check as old_check, FOLDER as FIX
    from docci_text_guided_image_v20.core import FOLDER as OLD, check_source
    fix, old = old_check(), check_source()
    files = {p.name:sha(p) for p in sorted(FOLDER.iterdir()) if p.is_file() and p.name != 'MANIFEST.json'}
    deps = dict(old['dependencies'])
    for folder, names in ((OLD, list(old['files'])+['MANIFEST.json']),
                          (FIX, list(fix['files'])+['MANIFEST.json'])):
        for name in names:
            deps[(folder/name).relative_to(PROJECT).as_posix()] = sha(folder/name)
    for name in ('docci_text_aid_v21/core.py', 'docci_text_aid_v21/runtime.py',
                 'docci_digital_compare_v19/digital.py', 'docci_digital_compare_v19/ldpc.py',
                 'docci_text_continue_v11/metrics.py'):
        deps['v10_extension/'+name] = sha(PROJECT/'v10_extension'/name)
    for n in files:
        if n.endswith('.py'):
            ast.parse((FOLDER/n).read_text(), feature_version=(3, 8))
    value = dict(schema='v22-source-release', files=files, dependencies=deps)
    dest = PROJECT/'releases/v10_docci_pretrained_checks_v22.tar.gz'
    manifest = FOLDER/'MANIFEST.json'
    if manifest.exists() and json.loads(manifest.read_text()) != value:
        if not a.refresh_unreleased or dest.exists():
            raise ValueError('Different released manifest preserved')
    if not manifest.exists() or a.refresh_unreleased:
        manifest.write_text(json.dumps(value, indent=2)+'\n')
    if a.manifest_only:
        print('Manifest frozen: {} sources / {} dependency pins'.format(len(files), len(deps)))
        return
    if dest.exists() or dest.with_name(dest.name+'.sha256').exists():
        raise FileExistsError('Existing release preserved')
    mapping = {'v10_extension/docci_pretrained_checks_v22/'+n:FOLDER/n for n in list(files)+['MANIFEST.json']}
    with dest.open('xb') as f, tarfile.open(fileobj=f, mode='w:gz', format=tarfile.USTAR_FORMAT) as tf:
        for n, file in mapping.items():
            info = tarfile.TarInfo(n); info.size = file.stat().st_size; info.mode = 0o644
            with file.open('rb') as source:
                tf.addfile(info, source)
    with tarfile.open(dest, 'r:gz') as tf:
        members = tf.getmembers()
        if len(members) != len(mapping) or {m.name for m in members} != set(mapping):
            raise ValueError('Archive file matrix differs')
        for member in members:
            if not member.isfile() or hashlib.sha256(tf.extractfile(member).read()).hexdigest() != sha(mapping[member.name]):
                raise ValueError('Archive byte verification failed')
    with dest.with_name(dest.name+'.sha256').open('x') as f:
        f.write(sha(dest)+'  '+dest.name+'\n')
    print(str(dest)); print(sha(dest))

if __name__=='__main__':
    main()
