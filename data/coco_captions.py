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
    """COCO 2017 images paired with COCO JSON or Localized Narratives JSONL."""

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
        if self.caption_mode not in {"first", "shortest", "longest", "random"}:
            raise ValueError(
                "caption_mode must be first, shortest, longest, or random"
            )
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_root}")
        if not self.annotation_file.is_file():
            raise FileNotFoundError(f"Caption annotation not found: {self.annotation_file}")

        filenames, captions = self._load_annotations()
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

    def _load_annotations(self) -> tuple[Dict[int, str], Dict[int, List[str]]]:
        suffix = self.annotation_file.suffix.lower()
        if suffix == ".jsonl":
            return self._load_localized_narratives()
        if suffix == ".json":
            return self._load_coco_captions()
        raise ValueError(
            "Caption annotations must be official COCO .json or "
            "Localized Narratives .jsonl"
        )

    def _load_coco_captions(self) -> tuple[Dict[int, str], Dict[int, List[str]]]:
        with self.annotation_file.open("r", encoding="utf-8") as handle:
            coco = json.load(handle)
        filenames = {int(item["id"]): item["file_name"] for item in coco["images"]}
        captions: Dict[int, List[str]] = defaultdict(list)
        for item in coco["annotations"]:
            caption = self._normalize_caption(item.get("caption"))
            if caption:
                captions[int(item["image_id"])].append(caption)
        return filenames, captions

    def _load_localized_narratives(
        self,
    ) -> tuple[Dict[int, str], Dict[int, List[str]]]:
        expected_dataset_id = f"mscoco_{self.image_root.name}"
        filenames: Dict[int, str] = {}
        captions: Dict[int, List[str]] = defaultdict(list)
        seen_pairs = set()
        with self.annotation_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at {self.annotation_file}:{line_number}"
                    ) from exc
                dataset_id = str(item.get("dataset_id", ""))
                if dataset_id and dataset_id != expected_dataset_id:
                    raise ValueError(
                        f"Unexpected dataset_id={dataset_id!r} at line {line_number}; "
                        f"expected {expected_dataset_id!r} for {self.image_root}"
                    )
                try:
                    image_id = int(item["image_id"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid image_id at {self.annotation_file}:{line_number}"
                    ) from exc
                caption = self._normalize_caption(item.get("caption"))
                if not caption:
                    continue
                pair = (image_id, caption)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                filenames[image_id] = f"{image_id:012d}.jpg"
                captions[image_id].append(caption)
        return filenames, captions

    @staticmethod
    def _normalize_caption(value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).strip().split())

    def __len__(self) -> int:
        return len(self.records)

    def _caption(self, captions: List[str], index: int) -> str:
        if self.caption_mode == "first":
            return captions[0]
        if self.caption_mode == "shortest":
            return min(captions, key=lambda text: (len(text.split()), len(text)))
        if self.caption_mode == "longest":
            return max(captions, key=lambda text: (len(text.split()), len(text)))
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
