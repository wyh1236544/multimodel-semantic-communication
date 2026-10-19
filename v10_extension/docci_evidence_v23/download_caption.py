"""Optional LOCAL-only stdlib downloader. Default prints a plan, downloads nothing.

Pinned pre-2024 image-token migration; validates Git blobs and LFS SHA256.
No model import, dependency installation, TLS bypass, mirrors or auto-inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

REPO='Salesforce/blip2-opt-2.7b'
REVISION='1ef7f63a8f0a144c13fdca8103eb7b4691c74cec'
# Official /api/models/.../revision/<REVISION>?blobs=true, checked 2026-10-19.
FILES={
 'README.md':(6607,'git','0b4e4ae1baa059bd676e0ae073513692f3cbda6a'),
 'config.json':(6963,'git','496fc3c7537bfacbf6cb429d2b036a4204ccefd5'),
 'merges.txt':(456356,'git','6636bda4a1fd7a63653dffb22683b8162c8de956'),
 'preprocessor_config.json':(432,'git','0b0fa393c4e1fa80a41fd3b84e022c6e3488230e'),
 'pytorch_model-00001-of-00002.bin':(9996239804,'lfs','83f4604e9f2c81dace48cbbb245cbe9acadddce7471c17eedc10cd675bf9af62'),
 'pytorch_model-00002-of-00002.bin':(5497724774,'lfs','b224ac0c148bf3aa0a211e5d043d38918ef57c2d3b714771a7c4b124129dbd48'),
 'pytorch_model.bin.index.json':(121727,'git','10539203b28f48aa69807e2526c9257cac3f4d28'),
 'special_tokens_map.json':(548,'git','fd05f8e8b4a81d5d4987d20d7fd4258e73e5e51a'),
 'tokenizer.json':(2108602,'git','4d1656ea9b723eda0760e7f05d5fcf4301977292'),
 'tokenizer_config.json':(904,'git','383c1ae016838f1427e52e513780716aa00fd3a6'),
 'vocab.json':(798293,'git','4ebe4bb3f3114daf2e4cc349f24873a1175a35d7')}

def verify_file(path,spec):
    size,kind,expected=spec
    if path.stat().st_size!=size:raise ValueError('File size differs '+path.name)
    h=hashlib.sha256() if kind=='lfs' else hashlib.sha1(('blob '+str(size)+'\0').encode())
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block);digest.update(block)
    if h.hexdigest()!=expected:raise ValueError('Official file hash differs '+path.name)
    return digest.hexdigest()

def download_one(folder,name,spec,deadline,opener=urllib.request.urlopen):
    target=folder/name;partial=folder/(name+'.part');size=spec[0]
    if target.exists():return verify_file(target,spec)
    url='https://huggingface.co/'+REPO+'/resolve/'+REVISION+'/'+name
    for attempt in range(3):
        try:
            offset=partial.stat().st_size if partial.exists() else 0
            if offset>size:raise ValueError('Oversized partial preserved '+name)
            if offset==size:break
            headers={'User-Agent':'multimodel-v23-offline-assets','Accept-Encoding':'identity'}
            if offset:headers['Range']='bytes={}-'.format(offset)
            with opener(urllib.request.Request(url,headers=headers),timeout=30) as response:
                if offset and (response.status!=206 or not response.headers.get('Content-Range','').startswith('bytes '+str(offset)+'-')):
                    raise ValueError('Server did not honor resume range; partial preserved')
                with partial.open('ab' if partial.exists() else 'xb') as f:
                    last=time.monotonic()
                    while True:
                        if time.monotonic()>deadline:raise TimeoutError('Download budget exceeded; partial preserved')
                        chunk=response.read(1024*1024)
                        if not chunk:break
                        offset+=len(chunk)
                        if offset>size:raise ValueError('Response exceeds official file size')
                        f.write(chunk)
                        if time.monotonic()-last>=5:
                            print('{} {:.2f}% ({:.3f}/{:.3f} GB)'.format(name,100*offset/size,offset/1e9,size/1e9),flush=True);last=time.monotonic()
            if offset!=size:raise OSError('Incomplete transfer')
            break
        except (OSError,TimeoutError) as error:
            if attempt==2 or time.monotonic()>deadline:raise
            print('Transfer interrupted; same pinned partial retained: '+str(error),flush=True)
    digest=verify_file(partial,spec);partial.rename(target)
    print('VERIFIED '+name,flush=True);return digest

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--output',required=True)
    p.add_argument('--download',action='store_true');p.add_argument('--max-hours',type=float,default=12.)
    a=p.parse_args();total=sum(v[0] for v in FILES.values())
    print(json.dumps(dict(repo=REPO,revision=REVISION,bytes=total,GB=total/1e9,GiB=total/2**30,
        download_requested=a.download,RAM='streaming 1MiB chunks; no model loaded',
        model_GPU_memory='NOT_MEASURED; mandatory server smoke',pretraining_overlap='unknown'),indent=2),flush=True)
    if not a.download:return
    if not 0<a.max_hours<=24:raise ValueError('Explicit finite <=24h local download budget')
    folder=Path(a.output).absolute()
    if folder.is_symlink() or any(q.is_symlink() for q in folder.parents):raise ValueError('No symlink asset output')
    folder.mkdir(parents=True,exist_ok=True)
    allowed=set(FILES)|{n+'.part' for n in FILES}|{'DOWNLOAD_RECEIPT.json'}
    if any(p.name not in allowed or not p.is_file() or p.is_symlink() for p in folder.iterdir()):raise ValueError('Unrelated existing files preserved; use new folder')
    remaining=sum(max(0,spec[0]-(folder/n).stat().st_size if (folder/n).exists() else spec[0]-(folder/(n+'.part')).stat().st_size if (folder/(n+'.part')).exists() else spec[0]) for n,spec in FILES.items())
    if shutil.disk_usage(folder).free<remaining+2*2**30:raise RuntimeError('Need remaining model bytes plus 2GiB free disk')
    deadline=time.monotonic()+a.max_hours*3600;hashes={}
    for name,spec in FILES.items():hashes[name]=download_one(folder,name,spec,deadline)
    result=dict(repo=REPO,revision=REVISION,state='COMPLETE',files=hashes,official_identity='Git blob or LFS SHA verified',overlap_audit='unknown')
    receipt=folder/'DOWNLOAD_RECEIPT.json'
    if receipt.exists():
        if json.loads(receipt.read_text())!=result:raise ValueError('Existing receipt differs')
    else:
        with receipt.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print('All model files VERIFIED. No inference or server changes.',flush=True)

if __name__=='__main__':main()
