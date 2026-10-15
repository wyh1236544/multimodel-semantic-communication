"""Explicit local-only online download into NEW directory; never install dependencies."""
import argparse
from pathlib import Path

from core import FOLDER, file_hash, fresh, load, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-network', action='store_true', required=True)
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    model_id = load(FOLDER / 'protocol.json')['bertscore']['model_id']
    info = HfApi().model_info(model_id)
    available = {s.rfilename for s in info.siblings}
    weight = 'model.safetensors' if 'model.safetensors' in available else 'pytorch_model.bin'
    if weight not in available:
        raise ValueError('Expected official single-file weights unavailable; review before downloading')
    names = {weight, 'config.json', 'tokenizer_config.json', 'special_tokens_map.json',
             'vocab.json', 'merges.txt', 'tokenizer.json', 'README.md'} & available
    if not {'config.json', 'vocab.json', 'merges.txt'}.issubset(names):
        raise ValueError('Incomplete official tokenizer inventory')
    root = fresh(args.output)
    snapshot_download(repo_id=model_id, revision=info.sha, local_dir=str(root), allow_patterns=sorted(names))
    save(root / 'DOWNLOAD_RECEIPT.json', dict(model_id=model_id, resolved_revision=info.sha,
         files={name: file_hash(root / name) for name in sorted(names)},
         scope='Independent evaluation weights only, no project training weights; downloaded from approved official model repository'))
    print('Evaluator saved at ' + str(root) + '; revision=' + info.sha)


if __name__ == '__main__':
    main()
