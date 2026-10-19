"""Reuse unchanged actual V22 source/FEC/modulation code, extend only noise grid."""
from pathlib import Path
import numpy as np
from PIL import Image
from docci_pretrained_checks_v22.digital import PROFILES,encode,receive,source_encode,reconstruct,llrs
from docci_evidence_v23.core import (prepared,digital_observation,case_name,sha,
    line,atomic,complete,metric)

def run(a,root,budget):
    index,data,examples=prepared(a.prepared)
    (root/'images').mkdir();(root/'tx').mkdir();payloads={}
    for eid,ex in examples.items():
        with Image.open(Path(a.prepared)/ex['source_image']) as im: source=im.convert('RGB')
        for profile in PROFILES:
            budget()
            payload,choice=source_encode(source,profile,'jpeg',root,budget=budget)
            signal,allocation,private=encode(payload,0,profile)
            check=receive(signal,profile)
            if not check['packet_crc'] or check['payload']!=payload: raise RuntimeError('Actual clean packet roundtrip failed')
            (root/'tx'/(eid+'__'+profile+'.jpg')).write_bytes(payload)
            payloads[eid,profile]=(payload,choice,signal,allocation,private)
    with (root/'rows.jsonl').open('x',encoding='utf-8') as f:
        for i,r in enumerate(data):
            for profile in PROFILES:
                budget();payload,choice,signal,allocation,private=payloads[r['example_id'],profile]
                y=digital_observation(signal,index,r);rx=receive(y,profile)
                decoded=rx.pop('decoded_bits');value=rx.pop('payload')
                errors=None if decoded is None else int(np.count_nonzero(decoded!=private['packet']))
                if rx['packet_crc'] and value!=payload: raise RuntimeError('Undetected wrong packet; preserve output')
                image=reconstruct(value,0) if rx['packet_crc'] else Image.new('RGB',(512,512),(128,128,128))
                path=root/'images'/(case_name(r['example_id'],r['snr_db'],r['repeat'])+'__'+profile+'.png')
                image.save(path)
                hard=np.concatenate(((llrs(y[:680],1.,2)<0).astype(np.uint8),
                    (llrs(y[680:],1.,allocation['bps'],allocation['body_gain'])<0).astype(np.uint8)))
                with Image.open(Path(a.prepared)/examples[r['example_id']]['source_image']) as im:truth=np.asarray(im.convert('RGB'))
                scores=metric(truth,np.asarray(image),examples[r['example_id']]['rectangle'])
                line(f,dict(example_id=r['example_id'],snr_db=r['snr_db'],repeat=r['repeat'],control='jpeg_'+profile,
                    image_file=path.relative_to(root).as_posix(),image_sha256=sha(path),source_coding=choice,
                    allocation=allocation,rx=rx,pre_fec_errors=int(np.count_nonzero(hard!=private['bits'])),
                    pre_fec_bits=len(hard),post_fec_errors=errors,post_fec_bits=0 if errors is None else len(private['packet']),
                    source_image_at_RX=False,source_text_at_RX=False,post_noise_renormalization=False,
                    actual_average_power=float(np.mean(abs(signal)**2)),failure_fill='constant RGB128',**scores))
            print('V23 DIGITAL {}/{} (all four MCS)'.format(i+1,len(data)),flush=True)
    atomic(root/'identity.json',dict(prepared_sha256=sha(Path(a.prepared)/'COMPLETE.json'),
        codec='jpeg',JPEG2000=False,BPG=False,image_total_symbols=4128,ARQ=False,
        same_image_frame_energy=True,noise_family=index['noise_family'],
        JPEG_is_low_rate_reference_not_strongest_digital=True,adaptive_MCS_selection=False))
    complete(root,'digital',dict(rows=len(data)*4,training=False))
