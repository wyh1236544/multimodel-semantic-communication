from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Collection, Dict, List, Optional

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset


class COCOCaptionDataset(Dataset):
    """COCO 2017 image-caption pairs using the official captions JSON file."""

    def __init__(
        self,
        image_root: str | Path,
        annotation_file: str | Path,
        image_size: int = 128,
        train: bool = False,
        caption_mode: str = "shortest",
        max_samples: Optional[int] = None,
        seed: int = 42,
        exclude_image_ids: Optional[Collection[int]] = None,
    ):
        self.image_root = Path(image_root)
        self.annotation_file = Path(annotation_file)
        self.train = bool(train)
        self.caption_mode = str(caption_mode).lower()
        self.seed = int(seed)
        if self.caption_mode not in {"first", "shortest", "random"}:
            raise ValueError("caption_mode must be first, shortest, or random")
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_root}")
        if not self.annotation_file.is_file():
            raise FileNotFoundError(f"Caption annotation not found: {self.annotation_file}")

        with self.annotation_file.open("r", encoding="utf-8") as handle:
            coco = json.load(handle)
        filenames = {int(item["id"]): item["file_name"] for item in coco["images"]}
        captions: Dict[int, List[str]] = defaultdict(list)
        for item in coco["annotations"]:
            caption = " ".join(str(item["caption"]).strip().split())
            if caption:
                captions[int(item["image_id"])].append(caption)
        records = [
            {"image_id": image_id, "file_name": file_name, "captions": captions[image_id]}
            for image_id, file_name in sorted(filenames.items())
            if captions.get(image_id) and (self.image_root / file_name).is_file()
        ]
        if exclude_image_ids:
            excluded = {int(image_id) for image_id in exclude_image_ids}
            records = [
                record
                for record in records
                if int(record["image_id"]) not in excluded
            ]
        if max_samples is not None and int(max_samples) > 0:
            generator = random.Random(self.seed)
            generator.shuffle(records)
            records = records[: int(max_samples)]
        if not records:
            raise RuntimeError("No valid COCO image-caption pairs were found")
        self.records = records

        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.records)

    def _caption(self, captions: List[str], index: int) -> str:
        if self.caption_mode == "first":
            return captions[0]
        if self.caption_mode == "shortest":
            return min(captions, key=lambda text: (len(text.split()), len(text)))
        # Worker-local Python RNG is seeded by DataLoader; this varies captions
        # during training while remaining reproducible for a fixed worker seed.
        return random.choice(captions)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        path = self.image_root / record["file_name"]
        with Image.open(path) as image:
            image = ImageOps.fit(
                image.convert("RGB"),
                (self.image_size, self.image_size),
                method=Image.Resampling.BICUBIC,
            )
            if self.train and random.random() < 0.5:
                image = ImageOps.mirror(image)
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)
        return {
            "image": tensor,
            "text": self._caption(record["captions"], index),
            "image_id": int(record["image_id"]),
            "path": str(path),
        }


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": torch.stack([item["image"] for item in batch]),
        "texts": [item["text"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
        "paths": [item["path"] for item in batch],
    }


def build_dataloader(
    image_root: str | Path,
    annotation_file: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    train: bool,
    caption_mode: str,
    max_samples: Optional[int],
    seed: int,
    exclude_image_ids: Optional[Collection[int]] = None,
) -> DataLoader:
    dataset = COCOCaptionDataset(
        image_root=image_root,
        annotation_file=annotation_file,
        image_size=image_size,
        train=train,
        caption_mode=caption_mode,
        max_samples=max_samples,
        seed=seed,
        exclude_image_ids=exclude_image_ids,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=_collate,
        generator=generator,
        drop_last=train and len(dataset) >= batch_size,
    )
