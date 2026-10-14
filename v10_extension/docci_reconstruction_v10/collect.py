"""Verified matched comparisons + light records; no weights/caches/native dataset."""
import argparse
import gzip
import html
import io
from pathlib import Path
import sys
import tarfile
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT)); sys.path.insert(0,str(EXT.parent))
from docci_reconstruction_v10.core import safe_write,read,atomic_json
from docci_reconstruction_v10.report import verify
from docci_longclip_joint_v8.metrics import paired


def compare(results):
    comparisons={}
    for a,b in (('I0_latent','I1_pixel'),('I1_pixel','I2_shifted'),('T0_whole','T1_chunks')):
        if a not in results or b not in results: continue
        x,y=results[a],results[b]; ix,iy=x['identity'],y['identity']
        if ix['config']!=iy['config'] or ix['schedule']!=iy['schedule']: raise ValueError('Comparison schedule/config differs')
        hx=x['histories']['train_clean']; hy=y['histories']['train_clean']
        if [(r['batch_example_ids'],r['lr']) for r in hx]!=[(r['batch_example_ids'],r['lr']) for r in hy]: raise ValueError('Comparison TRAIN budget/order/LR not matched')
        if (a,b)==('I0_latent','I1_pixel') and ix['initial_state_hashes']!=iy['initial_state_hashes']: raise ValueError('Loss-only comparison did not share initialization')
        if (a,b)==('T0_whole','T1_chunks'):
            xa={k:v for k,v in ix['initial_state_hashes'].items() if k!='source_position'}; ya={k:v for k,v in iy['initial_state_hashes'].items() if k!='source_position'}
            if xa!=ya: raise ValueError('Text non-position initialization differs')
        xr=[r for r in x['evaluations']['clean_validation'] if r['control']=='matched']; yr=[r for r in y['evaluations']['clean_validation'] if r['control']=='matched']
        if [(r['example_id'],r.get('source_text_sha256',r.get('source_image_sha256'))) for r in xr]!=[(r['example_id'],r.get('source_text_sha256',r.get('source_image_sha256'))) for r in yr]: raise ValueError('Pairwise source cohort differs')
        metric='normalized_wer' if a.startswith('T') else 'psnr'; diff=[ry[metric]-rx[metric] for rx,ry in zip(xr,yr)]
        comparisons[b+'-minus-'+a]=dict(metric=metric,paired_delta=paired(diff,ix['config']['seed']),
            positive_is_better=metric=='psnr',payload_a=ix['geometry']['payload_complex_symbols'],payload_b=iy['geometry']['payload_complex_symbols'],
            parameters_a=ix['actual_trainable_parameters'],parameters_b=iy['actual_trainable_parameters'],
            scope='matched initialization/data/steps/resources; LOSS change only' if a=='I0_latent' else
                'same payload/data/steps, NEW architecture has different initialization/parameter count; NOT isolated architecture effect' if a=='I1_pixel' else
                'segmentation plus9x payload and different position capacity; NOT equal-rate compression gain')
    return comparisons


def collect(root,destination):
    root=safe_write(root).resolve(strict=True); destination=safe_write(destination)
    if destination.exists(): raise FileExistsError('Existing record archive preserved')
    launch=read(root/'launch_identity.json'); results={}; statuses={}
    for arm in launch['arms']:
        name='smoke_'+arm+'_512' if launch['action']=='smoke' else launch['action']+'_'+arm
        result=verify(root/name); results[arm]=result; statuses[arm]=dict(status=result['identity']['status'],folder=name,
            train_gate=result['identity']['train_gate'],validation_gate=result['identity']['validation_gate'],
            parameters=result['identity']['actual_trainable_parameters'],resources=result['identity']['geometry'])
    for child in root.iterdir():
        if child.is_dir() and (child/'run_identity.json').is_file(): verify(child)
    comparisons={} if launch['action']=='smoke' else compare(results)
    aggregate=dict(schema='docci-reconstruction-v10-records',engineering_verified=True,arms=statuses,comparisons=comparisons,
        test_used=False,quality_stop_retained=True,weights_included=False,warning='No Common/joint gain, compression savings, analog-vs-digital or hardware/paper replication claim.')
    if (root/'aggregate.json').exists():
        if read(root/'aggregate.json')!=aggregate: raise ValueError('Existing aggregate differs')
    else: atomic_json(root/'aggregate.json',aggregate)
    if not (root/'index.html').exists():
        with (root/'index.html').open('x',encoding='utf-8') as f:
            f.write('<!doctype html><meta charset="utf-8"><h1>DOCCI 图片/文字恢复 V10</h1><p>相同损失对照与增加资源/网络容量的对照分别解释；失败结果保留。</p>')
            for arm,r in statuses.items(): f.write('<p><a href="'+html.escape(r['folder'])+'/actual_recovery.html">'+html.escape(arm+' — '+r['status'])+'</a>；复符号 '+str(r['resources']['payload_complex_symbols'])+'</p>')
    allowed={'.json','.jsonl','.html','.png','.svg','.npz','.md','.py','.txt','.log'}
    files=[]
    for file in sorted(root.rglob('*')):
        if file.is_symlink(): raise ValueError('Symlink record rejected')
        if file.is_file() and file.suffix in allowed and '.cache' not in file.parts: files.append(file)
    with destination.open('xb') as out,gzip.GzipFile(fileobj=out,mode='wb',filename='',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT) as tar:
        for file in files:
            data=file.read_bytes(); member=tarfile.TarInfo('v10_docci_reconstruction_v10_records/'+str(file.relative_to(root))); member.size=len(data); member.mode=0o644
            tar.addfile(member,io.BytesIO(data))
    print('Verified {} arm(s); {} bytes; no weights'.format(len(statuses),destination.stat().st_size))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('destination'); a=p.parse_args(); collect(a.root,a.destination)
