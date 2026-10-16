"""Explicit compatibility entry; no source/result edits or skipped validation."""
import argparse
from pathlib import Path
import sys
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_rx_bridge_v16_fix1.compat import installed, write_receipt


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Use report ROOT | run LEGACY_RUN_ARGUMENTS | collect ROOT DESTINATION')
    action, argv = sys.argv[1], sys.argv[2:]
    with installed() as verify:
        if action == 'run':
            from docci_rx_bridge_v16 import run
            sys.argv = [str(EXT/'docci_rx_bridge_v16/run.py')]+argv
            run.main()
        elif action == 'report':
            p = argparse.ArgumentParser()
            p.add_argument('root')
            p.add_argument('--max-minutes', type=float, default=60)
            a = p.parse_args(argv)
            result = verify(a.root, max_minutes=a.max_minutes)
            write_receipt(a.root, result)
            print('FIX1 CPU VERIFIED; original baseline/index/cache/source bytes preserved', flush=True)
        elif action == 'collect':
            from docci_rx_bridge_v16 import collect
            p = argparse.ArgumentParser()
            p.add_argument('root')
            p.add_argument('destination')
            p.add_argument('--max-minutes', type=float, default=90)
            a = p.parse_args(argv)
            collect.collect(a.root, a.destination, a.max_minutes)
        else:
            raise ValueError('Unknown compatibility action')


if __name__ == '__main__':
    main()
