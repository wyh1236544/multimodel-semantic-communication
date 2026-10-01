"""Validate and collect small records/images/logs; never include checkpoints or data."""
import argparse
import json
import sys
import tarfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v10.paths import write_path
from token_guidance.report import analyze


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output')
    p.add_argument('archive')
    args = p.parse_args()
    folder, archive = write_path(args.output), write_path(args.archive)
    if archive.exists():
        raise FileExistsError('Existing archive preserved')
    complete = json.loads((folder/'COMPLETE.json').read_text())
    design = json.loads((folder/'design.json').read_text())
    rows = [json.loads(x) for x in (folder/'samples.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    result = analyze(rows, design)
    if complete['rows'] != result['rows'] or complete['expected_rows'] != result['expected_rows']:
        raise ValueError('Completion does not match independently validated matrix')
    for v in 'ABCD':
        evidence = json.loads((folder/('train_'+v)/'COMPLETE.json').read_text())
        if evidence['steps'] != design['train_samples']*design['epochs']:
            raise ValueError('Incomplete training: '+v)
    files = sorted(x for x in folder.rglob('*') if x.is_file() and x.suffix in ('.json', '.jsonl', '.md', '.svg', '.png'))
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(archive), 'x:gz') as tar:
        for path in files:
            if path.is_symlink():
                raise ValueError('No linked records allowed')
            tar.add(str(path), arcname=folder.name+'/'+str(path.relative_to(folder)), recursive=False)
        log = folder.with_name(folder.name+'.log')
        if log.is_file() and not log.is_symlink():
            tar.add(str(log), arcname=folder.name+'/run.log', recursive=False)
    print('Verified records, no .pth/data: '+str(archive))


if __name__ == '__main__':
    main()
