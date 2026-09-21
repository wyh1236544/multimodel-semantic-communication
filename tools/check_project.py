from __future__ import annotations

import argparse
import json
from pathlib import Path


def count_images(path: Path) -> int:
    return sum(1 for _ in path.glob("*.jpg")) if path.is_dir() else 0


def main():
    parser = argparse.ArgumentParser(description="Check V6.1 server files before training")
    parser.add_argument("--root", default="/data1/zw/WYH/multimodel")
    args = parser.parse_args()
    root = Path(args.root)
    coco = root / "COCO2017"
    required = {
        "train images": coco / "train2017",
        "val images": coco / "val2017",
        "train captions": coco / "annotations" / "captions_train2017.json",
        "val captions": coco / "annotations" / "captions_val2017.json",
        "bandwidth config": root / "models" / "bandwidth_config.json",
        "model entry": root / "models" / "joint_image_text_model.py",
    }
    failed = False
    for name, path in required.items():
        ok = path.is_dir() if "images" in name else path.is_file()
        print(f"{'OK' if ok else 'MISSING':7} {name:20} {path}")
        failed |= not ok
    print("train image count:", count_images(coco / "train2017"))
    print("val image count:", count_images(coco / "val2017"))
    weights = sorted((root / "V6.1_model_weights").glob("seed*.pth"))
    print("weights:", [path.name for path in weights])
    if (root / "models" / "bandwidth_config.json").is_file():
        with (root / "models" / "bandwidth_config.json").open(encoding="utf-8") as handle:
            bandwidth = json.load(handle)
        print("active preset:", bandwidth.get("active_preset"))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
