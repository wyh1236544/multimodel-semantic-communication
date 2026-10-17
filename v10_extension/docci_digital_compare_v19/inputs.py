"""Read-only import from an independently verified COMPLETE V18 run."""
import hashlib
import json
from pathlib import Path
from docci_digital_compare_v19.common import read,lines,sha,atomic,seal,CHECKPOINTS

def validate_sources(rows):
    if len(rows)!=100 or len({r['example_id'] for r in rows})!=100:raise ValueError('Pinned 100 descriptions required')
    for r in rows:
        if r['official_split']!='qual_dev' or hashlib.sha256(r['text'].encode()).hexdigest()!=r['text_sha256']:raise ValueError('Source identity differs')
    if len({r['text_sha256'] for r in rows})!=100:raise ValueError('Duplicate evaluation source')

def prepare(v18,output,budget):
    from docci_text_channel_v18.report import pipeline
    v18=Path(v18).resolve(strict=True)
    print('READ-ONLY V18 light verification; no training/cache regeneration',flush=True)
    verified=pipeline(v18,full=False,max_minutes=30);budget()
    if verified['status']!='COMPLETE':raise ValueError('Complete approved V18 run required')
    c=read(v18/'config.json');sources=read(v18/'inputs/qual_dev.json');validate_sources(sources)
    files=['PIPELINE.json','config.json','inputs/qual_dev.json','inputs/INPUTS.json']
    imported=[]
    for arm in CHECKPOINTS:
        if arm!='frozen':
            p='train_'+arm+'/TRAIN_COMPLETE.json';files.append(p)
            if read(v18/p)['checkpoint_sha256']!=CHECKPOINTS[arm]:raise ValueError('Pinned V18 final checkpoint differs')
        p='eval_'+arm+'/rows.jsonl';files.append(p)
        rows=lines(v18/p)
        for r in rows:
            if r['control']=='matched':
                imported.append(dict(example_id=r['example_id'],arm=arm,snr_db=r['snr_db'],repeat=r['repeat'],
                    received_caption=r['received_caption'],chunk_generated_token_ids=r['chunk_generated_token_ids'],
                    stopped_by_capacity=r['stopped_by_capacity'],normalized_wer=r['normalized_wer'],
                    normalized_word_errors=r['normalized_word_errors'],normalized_source_words=r['normalized_source_words'],
                    text_exact=r['text_exact'],provenance='V18_saved_output',epsilon_sha256=r['own_stats']['epsilon_sha256']))
    if len(imported)!=4800:raise ValueError('V18 matched matrix incomplete')
    output.mkdir(exist_ok=False)
    atomic(output/'sources.json',sources)
    with (output/'old_rows.jsonl').open('x',encoding='utf-8') as f:
        for r in imported:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    atomic(output/'identity.json',dict(v18_root=str(v18),old_files={n:sha(v18/n) for n in files},
        features=read(v18/'inputs/INPUTS.json')['features'],checkpoints=CHECKPOINTS,
        source_count=100,old_rows=4800,old_config=c,old_light_verified=True,
        old_large_arrays_rechecked_here=False,no_old_result_writes=True))
    seal(output)

def historical_unchanged(prepared):
    p=read(Path(prepared)/'identity.json');v18=Path(p['v18_root'])
    for n,h in p['old_files'].items():
        if sha(v18/n)!=h:raise ValueError('Read-only V18 source changed: '+n)

def inherited(source,output):
    # Only derived metadata is copied; no regeneration, hardlinks, or shared output writes.
    import shutil
    from docci_digital_compare_v19.common import unseal
    unseal(source);output.mkdir(exist_ok=False)
    for name in ('sources.json','old_rows.jsonl','identity.json','SEALED.json'):shutil.copyfile(Path(source)/name,output/name)

