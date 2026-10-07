"""Collect an independently verified completed run, including actual images/text."""
import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
from pathlib import Path
EXT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXT))
from v10.paths import write_path
from optical_realism_v3.report import verify, digest


def collect(directory, output):
    directory=Path(directory).resolve(strict=True)
    design,rows,_=verify(directory,require_complete=True)
    output=write_path(output,EXT.parent)
    sidecar=output.with_name(output.name+'.sha256')
    if output.exists() or sidecar.exists(): raise FileExistsError('Existing collection preserved')
    complete=json.loads((directory/'COMPLETE.json').read_text(encoding='utf-8'))
    permitted=set(complete['result_sha256'])|{'run_identity.json','progress.json','COMPLETE.json'}
    permitted.update(r[key] for r in rows for key in ('source_image_file','image_file','constellation_file') if key in r)
    files=sorted(p for p in directory.rglob('*') if p.is_file())
    allowed={'.json','.jsonl','.md','.html','.svg','.png'}
    if any(p.is_symlink() or p.suffix not in allowed or str(p.relative_to(directory)) not in permitted for p in files):
        raise ValueError('Only actual records/images; no weights/data/unknown linked files')
    hashes={str(p.relative_to(directory)):digest(p) for p in files}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz',format=tarfile.PAX_FORMAT) as archive:
            for p in files: archive.add(str(p),arcname=directory.name+'/'+str(p.relative_to(directory)),recursive=False)
            payload=json.dumps(dict(schema='optical-realism-records-v3',files=hashes),sort_keys=True,indent=2).encode()
            info=tarfile.TarInfo(directory.name+'/COLLECTION_MANIFEST.json'); info.size=len(payload)
            info.mode=0o644; info.mtime=0
            archive.addfile(info,io.BytesIO(payload))
        stream.flush(); os.fsync(stream.fileno())
    with tarfile.open(output,'r:gz') as archive:
        for name, expected in hashes.items():
            if hashlib.sha256(archive.extractfile(directory.name+'/'+name).read()).hexdigest()!=expected:
                raise ValueError('Collected content checksum mismatch')
    value=digest(output)
    with sidecar.open('x',encoding='utf-8') as stream: stream.write(value+'  '+output.name+'\n')
    print('Return this archive: '+str(output)+'\nSHA256: '+value,flush=True)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('directory'); p.add_argument('output')
    a=p.parse_args(); collect(a.directory,a.output)
