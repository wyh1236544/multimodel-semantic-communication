from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


EXPECTED_RECORDS = {"train": 134272, "val": 8573}
EXPECTED_DATASET_IDS = {
    "train": "mscoco_train2017",
    "val": "mscoco_val2017",
}


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_tokenizer(model_name_or_path: Optional[str]):
    if not model_name_or_path:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required when --tokenizer is provided"
        ) from exc
    return AutoTokenizer.from_pretrained(model_name_or_path)


def tokenizer_lengths(tokenizer, captions: Iterable[str]) -> list[int]:
    lengths: list[int] = []
    batch: list[str] = []
    for caption in captions:
        batch.append(caption)
        if len(batch) == 256:
            encoded = tokenizer(batch, add_special_tokens=True, truncation=False)
            lengths.extend(len(ids) for ids in encoded["input_ids"])
            batch.clear()
    if batch:
        encoded = tokenizer(batch, add_special_tokens=True, truncation=False)
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return lengths


def inspect_split(
    split: str,
    annotation_file: Path,
    image_root: Path,
    tokenizer=None,
    max_length: int = 77,
    skip_image_check: bool = False,
) -> Dict[str, Any]:
    expected_dataset_id = EXPECTED_DATASET_IDS[split]
    captions = []
    word_lengths = []
    image_counts: Counter[int] = Counter()
    invalid = []
    missing_images = []
    missing_image_count = 0

    with annotation_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                item = json.loads(line)
                image_id = int(item["image_id"])
                caption = " ".join(str(item["caption"]).strip().split())
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                invalid.append({"line": line_number, "error": str(exc)})
                continue
            if item.get("dataset_id") != expected_dataset_id or not caption:
                invalid.append(
                    {
                        "line": line_number,
                        "error": (
                            f"dataset_id={item.get('dataset_id')!r}, "
                            f"caption_empty={not bool(caption)}"
                        ),
                    }
                )
                continue
            image_counts[image_id] += 1
            captions.append(caption)
            word_lengths.append(len(caption.split()))
            if not skip_image_check:
                image_path = image_root / f"{image_id:012d}.jpg"
                if not image_path.is_file():
                    missing_image_count += 1
                    if len(missing_images) < 100:
                        missing_images.append(str(image_path))

    token_lengths = tokenizer_lengths(tokenizer, captions) if tokenizer else []
    report: Dict[str, Any] = {
        "split": split,
        "annotation_file": str(annotation_file),
        "image_root": str(image_root),
        "records": len(captions),
        "expected_records": EXPECTED_RECORDS[split],
        "unique_images": len(image_counts),
        "images_with_multiple_narratives": sum(
            count > 1 for count in image_counts.values()
        ),
        "invalid_records": len(invalid),
        "invalid_examples": invalid[:10],
        "missing_images": missing_image_count,
        "missing_image_examples": missing_images[:10],
        "word_length": {
            "mean": statistics.fmean(word_lengths) if word_lengths else 0.0,
            "median": statistics.median(word_lengths) if word_lengths else 0.0,
            "p90": percentile(word_lengths, 0.90),
            "p95": percentile(word_lengths, 0.95),
            "p99": percentile(word_lengths, 0.99),
            "max": max(word_lengths, default=0),
        },
    }
    if token_lengths:
        report["token_length"] = {
            "tokenizer": str(getattr(tokenizer, "name_or_path", "unknown")),
            "max_length": max_length,
            "mean": statistics.fmean(token_lengths),
            "median": statistics.median(token_lengths),
            "p90": percentile(token_lengths, 0.90),
            "p95": percentile(token_lengths, 0.95),
            "p99": percentile(token_lengths, 0.99),
            "max": max(token_lengths, default=0),
            "truncated_records": sum(length > max_length for length in token_lengths),
            "truncated_fraction": (
                sum(length > max_length for length in token_lengths)
                / max(len(token_lengths), 1)
            ),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate COCO Localized Narratives and report caption lengths"
    )
    parser.add_argument(
        "--data-root",
        default="/data1/zw/WYH/multimodel/COCO2017",
    )
    parser.add_argument("--tokenizer", help="CLIP tokenizer path or model ID")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--skip-image-check", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    root = Path(args.data_root)
    annotation_root = root / "annotations" / "localized_narratives"
    tokenizer = load_tokenizer(args.tokenizer)
    reports = []
    for split, image_dir, filename in (
        ("train", "train2017", "coco_train_captions.jsonl"),
        ("val", "val2017", "coco_val_captions.jsonl"),
    ):
        annotation_file = annotation_root / filename
        if not annotation_file.is_file():
            raise FileNotFoundError(f"Missing annotation: {annotation_file}")
        report = inspect_split(
            split=split,
            annotation_file=annotation_file,
            image_root=root / image_dir,
            tokenizer=tokenizer,
            max_length=args.max_length,
            skip_image_check=args.skip_image_check,
        )
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False, indent=2))

    failures = []
    for report in reports:
        if report["records"] != report["expected_records"]:
            failures.append(
                f"{report['split']} record count {report['records']} != "
                f"{report['expected_records']}"
            )
        if report["invalid_records"]:
            failures.append(f"{report['split']} has invalid records")
        if not args.skip_image_check and report["missing_images"]:
            failures.append(f"{report['split']} has missing images")
    payload = {"reports": reports, "failures": failures}
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
