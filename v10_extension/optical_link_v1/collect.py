"""Independently validate matrix then collect records/images, no weights/data."""
import argparse
import hashlib
import json
import math
import sys
import tarfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from v10.paths import write_path
from optical_link_v1.report import summarize


def verify(folder):
    complete=json.loads((folder/'COMPLETE.json').read_text())
    design=json.loads((folder/'design.json').read_text())
    rows=[json.loads(line) for line in (folder/'samples.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows or len(rows)!=complete['rows'] or len(rows)!=complete['expected_rows']:
        raise ValueError('Incomplete result count')
    if 'effective_cases' in design:
        ids=json.loads((folder/'validation_ids.json').read_text())
        if len(set(ids))!=len(ids) or len(ids)!=design['val_samples']: raise ValueError('Duplicate/incomplete validation IDs')
        repeats=range(design['repeats'])
        wanted={(i,'quantized_clean',0) for i in ids}|{(i,c,r) for i in ids for c in design['effective_cases'] for r in repeats}
        actual=[(r['image_id'],r['case'],r['repeat']) for r in rows]
        if len(actual)!=len(set(actual)) or set(actual)!=wanted or len(rows)!=design['expected_rows']:
            raise ValueError('Missing/duplicate/extra image-case-repeat cells')
        for row in rows:
            if row.get('raw_utf8_wire_bytes')!=0 or row.get('variant')!='A' or row.get('nominal_snr_db') is not None:
                raise ValueError('Changed A/no-text-side-channel/physical-SNR contract')
            if not all(math.isfinite(row[k]) for k in ('psnr','ssim','token_acc','bleu2')):
                raise ValueError('Nonfinite semantic metrics')
            for key in ('image_file','source_image_file','constellation_file'):
                if row.get(key):
                    target=write_path(folder/row[key])
                    target.relative_to(folder)
                    if not target.is_file(): raise ValueError('Missing image/constellation')
    else:
        if len(rows)!=design['expected_rows']: raise ValueError('Incomplete declared diagnostic budget')
        wanted={(c,r) for c in design['optical_config']['cases'] for r in range(design['frames'])}
        actual=[(r['case'],r['repeat']) for r in rows]
        if len(actual)!=len(set(actual)) or set(actual)!=wanted: raise ValueError('Incomplete diagnostic matrix')
    events=[json.loads(line) for line in (folder/'packets.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    grouped={}
    for event in events:
        key=(event.get('image_id'),event['case'],event['repeat'])
        grouped.setdefault(key,[]).append(event)
    expected_keys=set()
    for row in rows:
        if row['case']=='quantized_clean': continue
        key=(row.get('image_id'),row['case'],row['repeat']); expected_keys.add(key)
        packet_events=grouped.get(key,[])
        if len(packet_events)!=row['packet_attempts']+1 or sum(e['purpose']=='probe' for e in packet_events)!=1:
            raise ValueError('Missing/extra packet/probe evidence')
        for field in ('pre_errors','pre_bits','post_errors','post_bits','awg_samples','guard_samples'):
            if sum(e[field] for e in packet_events)!=row[field]: raise ValueError('Packet sum mismatch: '+field)
        for field in ('scheduled_seconds','snr_signal_energy_a2','snr_noise_energy_a2'):
            if not math.isclose(sum(e[field] for e in packet_events),row[field],rel_tol=1e-10,abs_tol=1e-30):
                raise ValueError('Packet energy/time mismatch: '+field)
        accepted=row.get('feature_object_crc_accepted',row.get('object_accepted',False))
        if accepted and any(e['crc_accepted'] is False for e in packet_events):
            raise ValueError('Accepted object contains rejected packet')
    if set(grouped)!=expected_keys: raise ValueError('Extra packet cells')
    summarize(rows)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output'); p.add_argument('archive'); args=p.parse_args()
    folder,archive=write_path(args.output),write_path(args.archive)
    verify(folder)
    if archive.exists(): raise FileExistsError('Existing archive preserved')
    archive.parent.mkdir(parents=True,exist_ok=True)
    files=sorted(path for path in folder.rglob('*') if path.is_file() and path.suffix in ('.json','.jsonl','.md','.png','.svg','.html'))
    with tarfile.open(str(archive),'x:gz') as tar:
        for path in files:
            write_path(path)
            tar.add(str(path),arcname=folder.name+'/'+str(path.relative_to(folder)),recursive=False)
        log=folder.with_name(folder.name+'.log')
        if log.is_file():
            write_path(log)
            tar.add(str(log),arcname=folder.name+'/run.log',recursive=False)
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    print('Verified records (no weights/data): {}\nSHA256 {}'.format(archive,digest))


if __name__=='__main__': main()
