"""Verify completed lattice and make a new small records archive, no weights."""
import argparse
import gzip
import io
import json
from pathlib import Path
import sys
import tarfile
EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT))
from v10.paths import write_path
from business_link.report import make_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output')
    p.add_argument('archive')
    args = p.parse_args()
    output, archive = write_path(args.output), write_path(args.archive)
    design = json.loads((output/'design.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (output/'samples.jsonl').read_text(encoding='utf-8').splitlines() if line]
    report = make_report(rows, design)
    done = json.loads((output/'COMPLETE.json').read_text())
    if done['rows'] != report['rows'] or done['expected_rows'] != report['expected_rows']:
        raise ValueError('Incomplete completion evidence')
    if archive.exists():
        raise FileExistsError('Existing archive preserved')
    selected = ['business_summary.json', 'business_table.csv', 'REPORT.md', 'business_manifest.json',
                'design.json', 'samples.jsonl', 'packets.jsonl', 'run_identity.json',
                'checkpoint_load.json', 'COMPLETE.json']
    files = [output/name for name in selected]
    files += sorted((output/'images').glob('*.png'))
    files += sorted((output/'constellations').glob('*.svg'))
    files += sorted((output/'constellations').glob('*.json'))
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise ValueError('Missing/linked record file')
    with archive.open('xb') as stream:
        with gzip.GzipFile(filename='', fileobj=stream, mode='wb', mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w') as tar:
                for path in files:
                    data = path.read_bytes()
                    info = tarfile.TarInfo(output.name+'/'+path.relative_to(output).as_posix())
                    info.size, info.mode = len(data), 0o644
                    tar.addfile(info, io.BytesIO(data))
    print('Verified {} rows; records saved: {}'.format(report['rows'], archive))


if __name__ == '__main__':
    main()
