"""Bundle both verified policies, B2B and logs, no datasets or checkpoints."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def bundle(root):
    root=Path(root).resolve(strict=True)
    allowed=[]
    for policy in ('qpsk_conv','16qam_repeat2'):
        archive=root/(policy+'_records.tar.gz'); sidecar=archive.with_name(archive.name+'.sha256')
        expected=sidecar.read_text().split()[0]
        if archive.is_symlink() or hashlib.sha256(archive.read_bytes()).hexdigest()!=expected:
            raise ValueError('Missing/changed policy collection')
        allowed += [archive,sidecar]
    for name in ('smoke.log','qpsk_conv.log','16qam_repeat2.log'):
        p=root/name
        if p.is_file(): allowed.append(p)
    if (root/'b2b').exists():
        complete=json.loads((root/'b2b/COMPLETE.json').read_text())
        if not complete['b2b_functional_gate'] or hashlib.sha256((root/'b2b/b2b.json').read_bytes()).hexdigest()!=complete['b2b_sha256']:
            raise ValueError('B2B completion mismatch')
        allowed += [root/'b2b/COMPLETE.json',root/'b2b/b2b.json']+sorted((root/'b2b').glob('*.svg'))
    if (root/'receiver_diagnostic').exists():
        complete=json.loads((root/'receiver_diagnostic/COMPLETE.json').read_text())
        for name,key in (('diagnostic.json','diagnostic_sha256'),('summary.json','summary_sha256')):
            p=root/'receiver_diagnostic'/name
            if hashlib.sha256(p.read_bytes()).hexdigest()!=complete[key]:
                raise ValueError('Changed receiver replay diagnostic')
            allowed.append(p)
        allowed.append(root/'receiver_diagnostic/COMPLETE.json')
    if any(p.is_symlink() for p in allowed): raise ValueError('No linked collection inputs')
    output=root/'v10_optical_realism_v3_records.tar.gz'
    if output.exists() or output.with_name(output.name+'.sha256').exists(): raise FileExistsError('Existing bundle preserved')
    with output.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz') as tar:
            for path in allowed: tar.add(str(path),arcname=str(path.relative_to(root)),recursive=False)
    sha=hashlib.sha256(output.read_bytes()).hexdigest()
    with output.with_name(output.name+'.sha256').open('x') as f: f.write(sha+'  '+output.name+'\n')
    print('Return '+str(output),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('root'); a=p.parse_args(); bundle(a.root)
