from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path


def check_import(name: str):
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"OK      {name:15} {version}")
        return module
    except Exception as exc:
        print(f"MISSING {name:15} {exc}")
        return None


def main():
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    torch = check_import("torch")
    check_import("numpy")
    check_import("PIL")
    check_import("yaml")
    check_import("transformers")
    check_import("diffusers")
    check_import("safetensors")
    if torch is not None:
        print("CUDA available:", torch.cuda.is_available())
        print("PyTorch CUDA:", torch.version.cuda)
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU memory: {total:.1f} GiB")
    root = Path(__file__).resolve().parent
    print("Project root:", root)
    for relative in (
        "models/joint_image_text_model.py",
        "models/bandwidth_config.json",
        "configs/v61_bandwidth.yaml",
        "COCO2017/train2017",
        "COCO2017/val2017",
        "COCO2017/annotations/captions_train2017.json",
        "COCO2017/annotations/captions_val2017.json",
        "V6.1_model_weights/seed42.pth",
        "V6.1_model_weights/seed43.pth",
        "V6.1_model_weights/seed44.pth",
    ):
        path = root / relative
        print(f"{'OK' if path.exists() else 'MISSING':7} {relative}")


if __name__ == "__main__":
    main()

