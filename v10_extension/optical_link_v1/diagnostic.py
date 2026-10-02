"""Bounded NumPy bit-link diagnostic. Does NOT measure image/text recovery."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT))
import numpy as np
from business_link.engine import identity_seed
from optical_link_v1.plots import constellation_svg
from optical_link_v1 import VERSION
from optical_link_v1.config import load, TEMPLATE
from optical_link_v1.transport import OpticalTransport
from optical_link_v1.report import write_report
from v10.paths import write_path


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--optical-config', default=str(TEMPLATE))
    p.add_argument('--frames', type=int, default=4)
    p.add_argument('--payload-bytes', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if not 1 <= args.frames <= 64 or not 1 <= args.payload_bytes <= 4096:
        raise ValueError('Diagnostic budget: frames1..64, bytes1..4096')
    specification, cases, digest = load(args.optical_config)
    directory = write_path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'constellations').mkdir()
    started, rows = time.monotonic(), []
    def save(name, value):
        with (directory/name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
    identity = dict(version=VERSION, status='running', scope='random_bytes_numpy_link_only_no_neural_model',
                    optical_config_sha256=digest, python=sys.executable, argv=sys.argv,
                    optical_config=specification, frames=args.frames, payload_bytes=args.payload_bytes,
                    expected_rows=len(cases)*args.frames)
    save('design.json', identity)
    try:
        with (directory/'samples.jsonl').open('x', encoding='utf-8') as samples, (directory/'packets.jsonl').open('x', encoding='utf-8') as packets:
            for name, config in cases.items():
                for repeat in range(args.frames):
                    original = np.random.default_rng(identity_seed(args.seed, repeat, 'bytes')).integers(0,256,args.payload_bytes,dtype=np.uint8).tobytes()
                    transport = OpticalTransport(config, identity_seed(args.seed, repeat, 'field'), identity_seed(args.seed, repeat, 'noise'))
                    transport.probe()
                    recovered, ok, packet_record = transport.transfer(original, 'features', config['waveform']['policy'])
                    if ok and recovered != original:
                        raise RuntimeError('Undetected wire error in known diagnostic bytes')
                    row = dict(transport.totals(), case=name, repeat=repeat, object_accepted=ok,
                        feature_object_crc_accepted=ok, semantic_body_bytes=len(original),
                        payload_sha256=hashlib.sha256(original).hexdigest(), feature_packets=packet_record)
                    samples.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                    samples.flush()
                    for event in transport.events:
                        packets.write(json.dumps(dict(event,case=name,repeat=repeat), allow_nan=False)+'\n')
                    packets.flush()
                    if repeat == 0:
                        constellation_svg(transport.constellations, directory/'constellations'/(name+'.svg'))
                    rows.append(row)
                    physical_snr='undefined' if row['physical_symbol_snr_db'] is None else '{:.2f}dB'.format(row['physical_symbol_snr_db'])
                    print('LINK {} {}/{} SNR={} estimate={:.2f}dB CRC={} BER={:.4g}'.format(name,repeat+1,args.frames,physical_snr,row['estimated_snr_db'],ok,row['pre_fec_ber']), flush=True)
        if len(rows) != len(cases)*args.frames:
            raise RuntimeError('Incomplete link diagnostic')
        write_report(directory, rows, '随机字节链路诊断；未执行图文模型')
        save('COMPLETE.json', dict(version=VERSION, rows=len(rows), expected_rows=len(cases)*args.frames,
            wall_seconds=time.monotonic()-started, scope='random_bytes_link_only'))
    except BaseException as exc:
        save('FAILED.json', dict(error=str(exc), exception=type(exc).__name__, completed_rows=len(rows)))
        raise


if __name__ == '__main__':
    main()
