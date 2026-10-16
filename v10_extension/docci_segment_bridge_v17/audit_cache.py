"""Read-only, CPU validation of existing full cache, including excluded large arrays."""
import argparse
from pathlib import Path
import sys
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_segment_bridge_v17.core import CACHE_INDEX, sha, atomic, check_source


def main():
    p = argparse.ArgumentParser(); p.add_argument('cache'); p.add_argument('receipt'); a = p.parse_args()
    check_source()
    cache = Path(a.cache).resolve(strict=True)
    if sha(cache/'index.json') != CACHE_INDEX:
        raise ValueError('Not the approved v16_cache_01')
    from docci_rx_bridge_v16_fix1.compat import installed
    with installed() as verify:
        result = verify(cache, arrays=True, max_minutes=60)
    if result['stage'] != 'cache' or result['smoke'] or result['pairs'] != 9747:
        raise ValueError('Full9747 pair cache required; no relabelled smoke')
    atomic(Path(a.receipt), result)
    print('Existing original cache independently verified read-only; no cache regeneration', flush=True)


if __name__ == '__main__':
    main()
