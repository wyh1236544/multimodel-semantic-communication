"""Source-only USTAR, independent archive readback, separate offline metric assets."""
import argparse
import ast
import json
from pathlib import Path
import sys
import tarfile
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_text_guided_image_v20.core import FOLDER,PROJECT,read,sha,check_source

def archive(path,files):
    if path.exists() or path.with_name(path.name+'.sha256').exists():
        raise FileExistsError('Existing delivered archive preserved')
    with path.open('xb') as raw:
        with tarfile.open(fileobj=raw,mode='w:gz',format=tarfile.USTAR_FORMAT) as tf:
            for name,p in sorted(files.items()):
                info=tarfile.TarInfo(name)
                info.mode=0o644
                info.size=p.stat().st_size
                info.mtime=0
                with p.open('rb') as f:
                    tf.addfile(info,f)
    with tarfile.open(path,'r:gz') as tf:
        members=tf.getmembers()
        if {m.name for m in members}!=set(files) or len(members)!=len(files) or any(not m.isfile() for m in members):
            raise ValueError('Built archive member matrix differs')
        import hashlib
        for m in members:
            h=hashlib.sha256()
            handle=tf.extractfile(m)
            for block in iter(lambda:handle.read(1048576),b''):
                h.update(block)
            if h.hexdigest()!=sha(files[m.name]):
                raise ValueError('Archived source/asset differs')
    with path.with_name(path.name+'.sha256').open('x',encoding='utf-8') as f:
        f.write(sha(path)+'  '+path.name+'\n')
    print(str(path));print(sha(path))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest-only',action='store_true')
    p.add_argument('--assets-folder')
    a=p.parse_args()
    dependencies={}
    for module in ('docci_baseline_v14','docci_rx_bridge_v16','docci_text_channel_v18'):
        m=read(EXT/module/'MANIFEST.json')
        dependencies.update(m['dependencies'])
        for name in list(m['files'])+['MANIFEST.json']:
            dependencies['v10_extension/'+module+'/'+name]=sha(EXT/module/name)
    dependencies['utils/metrics.py']=sha(PROJECT/'utils/metrics.py')
    for name,h in dependencies.items():
        if sha(PROJECT/name)!=h:
            raise ValueError('Historical dependency changed: '+name)
    files={p.name:sha(p) for p in sorted(FOLDER.iterdir()) if p.is_file() and p.name!='MANIFEST.json'}
    for name in files:
        if name.endswith('.py'):
            ast.parse((FOLDER/name).read_text(),feature_version=(3,8))
    manifest=dict(schema='v20-source-release',files=files,dependencies=dependencies)
    (FOLDER/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    check_source()
    if a.manifest_only:
        print('V20 manifest prepared; no model/training execution')
        return
    release=PROJECT/'releases'
    release.mkdir(exist_ok=True)
    archive(release/'v10_docci_text_guided_image_v20.tar.gz',
        {'v10_extension/docci_text_guided_image_v20/'+n:FOLDER/n for n in list(files)+['MANIFEST.json']})
    if a.assets_folder:
        folder=Path(a.assets_folder)
        expected={'alexnet.pth':'7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02',
                  'linear.pth':'df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0'}
        for n,h in expected.items():
            if sha(folder/n)!=h:
                raise ValueError('Official metric weight differs')
        names=list(expected)+['LPIPS_LICENSE.txt','TORCHVISION_LICENSE.txt']
        archive(release/'v20_lpips_offline_assets.tar.gz',
            {'_v20_lpips_assets/'+n:folder/n for n in names})

if __name__=='__main__':
    main()
