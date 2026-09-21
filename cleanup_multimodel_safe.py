#!/usr/bin/env python3
"""Exact-list cleanup for /data1/zw/WYH/multimodel. Python 3.8+, stdlib only.
Default: preview. --delete: interactive, permanent unlink of specified files only.
No recursive deletion; no source code, real dataset or formal checkpoint targets.
"""
import argparse
import datetime
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import zipfile
import zlib

PROJECT = Path("/data1/zw/WYH/multimodel")
MANIFEST = [
  [
    "archives",
    "COCO2017/val2017.zip",
    815585330,
    "2026-07-17 17:13"
  ],
  [
    "archives",
    "COCO2017/train2017.zip",
    19336861798,
    "2026-07-17 23:17"
  ],
  [
    "archives",
    "COCO2017/annotations_trainval2017.zip",
    252907541,
    "2026-07-17 17:04"
  ],
  [
    "archives",
    "COCO2017/annotations/localized_narratives/localized_narratives_coco_captions.zip",
    8450442,
    "2026-08-22 15:53"
  ],
  [
    "test",
    "COCO2017/test2017.zip",
    6646970404,
    "2026-07-17 18:14"
  ],
  [
    "smoke",
    "experiment_results/ln_v7a_shared32_3stage_smoke/seed42.pth",
    564869745,
    "2026-08-22 19:44"
  ],
  [
    "smoke",
    "experiment_results/token_budget_smoke/budget_208/seed42.pth",
    564802865,
    "2026-07-24 12:22"
  ],
  [
    "smoke",
    "experiment_results/ln_v9a_feedback_smoke/seed42.pth",
    635282929,
    "2026-08-24 16:19"
  ],
  [
    "smoke",
    "experiment_results/high_6912_smoke/seed42.pth",
    564843761,
    "2026-07-23 13:33"
  ],
  [
    "smoke",
    "experiment_results/ln_v61_smoke/seed42.pth",
    564802993,
    "2026-08-22 16:29"
  ],
  [
    "smoke",
    "experiment_results/ln_v7a_shared32_smoke/seed42.pth",
    564869745,
    "2026-08-22 19:04"
  ],
  [
    "smoke",
    "experiment_results/ln_v8a_common_base_smoke/seed42.pth",
    635256305,
    "2026-08-23 15:19"
  ],
  [
    "smoke",
    "experiment_results/ln_v7b_regression_balanced_smoke/seed42.pth",
    572544359,
    "2026-08-23 12:43"
  ],
  [
    "incomplete",
    "pretrained/stable-diffusion-v1-5/.cache/huggingface/download/unet/qvL24ak4oZi668_a9cUr1J0xW-w=.42bc8b8f3af32866db3c7bb5bcf591ab04438296c2712246d7a640bde5a5ddc1.78a276ac.incomplete",
    0,
    "2026-08-16 18:34"
  ],
  [
    "incomplete",
    "pretrained/stable-diffusion-v1-5/.cache/huggingface/download/unet/ubm5m3bnxno3jFS723tbWO9jMbk=.c83908253f9a64d08c25fc90874c9c8aef9a329ce1ca5fb909d73b0c83d1ea21.9b56bc15.incomplete",
    0,
    "2026-08-16 18:32"
  ],
  [
    "incomplete",
    "pretrained/stable-diffusion-v1-5/.cache/huggingface/download/unet/qTcGw6_SfHk2KzLWka1IiYIORvY=.cd1b6db09a81cb1d39fbd245a89c1e3db9da9fe8eba5e8f9098ea6c4994221d3.7c3bedf6.incomplete",
    0,
    "2026-08-16 18:34"
  ],
  [
    "incomplete",
    "pretrained/stable-diffusion-v1-5/.cache/huggingface/download/unet/QGKUr3yaD9JD-RaWEuvjvgKhPHs=.c7da0e21ba7ea50637bee26e81c220844defdf01aafca02b2c42ecdadb813de4.fb455a13.incomplete",
    937519175,
    "2026-08-16 18:39"
  ],
  [
    "incomplete",
    "pretrained/stable-diffusion-v1-5/.cache/huggingface/download/unet/XvpF6D3IOdMzSRYQ9R6x9V0lSjo=.30eb3dc47c90e4a55476332b284b2331774c530edbbb83b70cacdd9e7b91af92.9d6c0c51.incomplete",
    39522,
    "2026-08-16 18:39"
  ],
  [
    "incomplete",
    "pretrained/stable-diffusion-v1-5/.cache/huggingface/download/vae/4SfAgk9U607e8pVunEB9nSiU10k=.a2b5134f4dbc140d9c11f11cba3233099e00af40f262f136c691fb7d38d2194c.ce8b8499.incomplete",
    199229440,
    "2026-08-16 18:39"
  ],
  [
    "incomplete",
    "pretrained/stable-diffusion-v1-5/.cache/huggingface/download/I435WywFzQd8laV1YV_K0UjQrvU=.e1441589a6f3c5a53f5f54d0975a18a7feb7cdf0b0dee276dfc3331ae376a053.2a954e69.incomplete",
    0,
    "2026-08-16 18:39"
  ],
  [
    "incomplete",
    "pretrained/stable-diffusion-v1-5/.cache/huggingface/download/Ov23847ez9V5KQfE83BsxwzJ9Ug=.cc6cb27103417325ff94f52b7a5d2dde45a7515b25c255d8e396c90014281516.857103e9.incomplete",
    0,
    "2026-08-16 18:34"
  ],
  [
    "incomplete",
    "pretrained/stable-diffusion-v1-5/.cache/huggingface/download/3DaBvY67ce_xicjc_iyzZfZKomY=.6ce0161689b3853acaa03779ec93eafe75a02f4ced659bee03f50797806fa2fa.ab24f1b3.incomplete",
    0,
    "2026-08-16 18:35"
  ],
  [
    "clip",
    "pretrained/clip-vit-base-patch32/tf_model.h5",
    605551040,
    "2026-07-23 11:03"
  ],
  [
    "clip",
    "pretrained/clip-vit-base-patch32/flax_model.msgpack",
    605123003,
    "2026-07-23 11:03"
  ]
]
GROUPS = ("archives", "incomplete", "smoke", "clip", "test")
NOTICES = {
    "archives": "Verify archive integrity and extracted data before deleting ZIPs.",
    "incomplete": "Confirm no download/resume is running and completed SD loads offline. Partial downloads cannot resume after deletion.",
    "smoke": "Confirm these smoke checkpoints are not needed for inference or resuming training. Formal checkpoints and metrics are retained.",
    "clip": "Confirm PyTorch CLIP loads offline and TensorFlow/Flax weights are not needed.",
    "test": "test2017.zip has NO verified extracted copy. Confirm it is not needed or you have a usable backup.",
}


def signature(info):
    return (info.st_dev, info.st_ino, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def regular_file(root, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts or "\\" in relative or not rel.parts:
        raise ValueError("Unsafe relative path: " + relative)
    path = root
    for part in rel.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Symlink refused: " + str(path))
    # Resolve both containment and the root before touching a target.
    resolved = path.resolve(strict=True)
    resolved.relative_to(root.resolve(strict=True))
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Not a regular file: " + str(path))
    if info.st_nlink != 1:
        raise ValueError("Hardlinked file refused: " + str(path))
    return path, info


def candidate(root, row):
    group, relative, expected_size, expected_modified = row
    path, info = regular_file(root, relative)
    if info.st_size != expected_size:
        raise ValueError("Size changed since inventory: " + str(path))
    modified = datetime.datetime.fromtimestamp(info.st_mtime).strftime("%Y-%m-%d %H:%M")
    if modified != expected_modified:
        raise ValueError("Modification time differs from inventory (including timezone differences): " + str(path))
    return path, info


def archive_layout(relative):
    layouts = {
        "COCO2017/train2017.zip": ("COCO2017", "train2017"),
        "COCO2017/val2017.zip": ("COCO2017", "val2017"),
        "COCO2017/annotations_trainval2017.zip": ("COCO2017", "annotations"),
        "COCO2017/annotations/localized_narratives/localized_narratives_coco_captions.zip":
            ("COCO2017/annotations/localized_narratives", None),
    }
    if relative not in layouts:
        raise ValueError("Not an approved redundant archive")
    return layouts[relative]


def verify_archive(root, relative, archive_path):
    base, prefix = archive_layout(relative)
    snapshots = []
    names = set()
    print("  Checking ZIP integrity (large archives can take time)...", flush=True)
    with zipfile.ZipFile(str(archive_path)) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError("ZIP CRC failure: " + bad)
        files = [item for item in archive.infolist() if not item.is_dir()]
        if not files:
            raise ValueError("ZIP has no files")
        print("  Comparing {} extracted files...".format(len(files)), flush=True)
        for index, item in enumerate(files, 1):
            member = PurePosixPath(item.filename)
            if (member.is_absolute() or ".." in member.parts
                    or "\\" in item.filename or not member.parts
                    or item.filename in names):
                raise ValueError("Unsafe or duplicate ZIP member: " + item.filename)
            names.add(item.filename)
            if prefix and member.parts[0] != prefix:
                raise ValueError("Unexpected ZIP layout: " + item.filename)
            if prefix is None and item.filename not in (
                    "coco_train_captions.jsonl", "coco_val_captions.jsonl"):
                raise ValueError("Unexpected LN archive member: " + item.filename)
            relative_file = str(PurePosixPath(base) / member)
            target, before = regular_file(root, relative_file)
            if before.st_size != item.file_size:
                raise ValueError("Extracted size mismatch: " + str(target))
            crc = 0
            with target.open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    crc = zlib.crc32(block, crc)
            if (crc & 0xffffffff) != item.CRC:
                raise ValueError("Extracted CRC mismatch: " + str(target))
            _, after = regular_file(root, relative_file)
            if signature(before) != signature(after):
                raise ValueError("Extracted file changed during verification: " + str(target))
            snapshots.append((relative_file, signature(after)))
            if index % 10000 == 0:
                print("  Verified {}/{}".format(index, len(files)), flush=True)
    if prefix is None and names != {"coco_train_captions.jsonl", "coco_val_captions.jsonl"}:
        raise ValueError("LN archive is missing a required annotation file")
    # Recheck every extracted file before authorizing the ZIP unlink.
    for relative_file, saved_signature in snapshots:
        _, current = regular_file(root, relative_file)
        if signature(current) != saved_signature:
            raise ValueError("Extracted file changed after verification: " + relative_file)
    print("  ZIP and extracted contents match.", flush=True)


def check_completed_models(root, group):
    if group == "clip":
        regular_file(root, "pretrained/clip-vit-base-patch32/config.json")
        _, info = regular_file(root, "pretrained/clip-vit-base-patch32/pytorch_model.bin")
        if info.st_size != 605247071:
            raise ValueError("Retained PyTorch CLIP weight differs from inventory")
    if group == "incomplete":
        for rel in (
            "pretrained/stable-diffusion-v1-5/model_index.json",
            "pretrained/stable-diffusion-v1-5/unet/diffusion_pytorch_model.safetensors",
            "pretrained/stable-diffusion-v1-5/vae/diffusion_pytorch_model.safetensors",
        ):
            _, info = regular_file(root, rel)
            if not info.st_size:
                raise ValueError("Completed SD file is empty: " + rel)
    # Existence is NOT proof of model loading. Operator must confirm offline use.


def remove_one(root, row, previous_signature):
    path, current = candidate(root, row)
    if signature(current) != previous_signature:
        raise ValueError("Candidate changed after preview: " + str(path))
    if row[0] == "archives":
        verify_archive(root, row[1], path)
    path, current = candidate(root, row)
    if signature(current) != previous_signature:
        raise ValueError("Candidate changed during checks: " + str(path))
    path.unlink()  # Exact approved regular file only. Never recursive.
    print("DELETED: " + str(path), flush=True)
    return current.st_size


def ask(message):
    try:
        return input(message).strip() == "DELETE"
    except EOFError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delete", action="store_true", help="Enable interactive permanent deletion")
    parser.add_argument("--groups", nargs="+", choices=GROUPS,
                        default=["archives", "incomplete", "smoke", "clip"],
                        help="Default excludes test2017.zip")
    args = parser.parse_args()
    if not PROJECT.is_dir() or PROJECT.resolve() != PROJECT:
        parser.error("Expected a real, non-symlink project directory: " + str(PROJECT))
    if args.delete and not sys.stdin.isatty():
        parser.error("--delete requires an interactive terminal; do not pipe 'yes' into this script")
    selected = [group for group in GROUPS if group in args.groups]
    ready = {}
    issues = 0
    for group in selected:
        print("\n[{}] {}".format(group, NOTICES[group]), flush=True)
        eligible = []
        for row in MANIFEST:
            if row[0] != group:
                continue
            try:
                path, info = candidate(PROJECT, row)
                print("  {:8.3f} GiB  {}".format(info.st_size / 2**30, path), flush=True)
                eligible.append((row, signature(info)))
            except FileNotFoundError:
                print("  ABSENT, skipped: " + row[1], flush=True)
            except (ValueError, OSError) as error:
                print("  SKIPPED: " + str(error), flush=True)
                issues += 1
        ready[group] = eligible
    planned = sum(row[2] for group in selected for row, sig in ready[group])
    print("\nCurrent eligible file sizes: {:.3f} GiB.".format(planned / 2**30), flush=True)
    if not args.delete:
        print("PREVIEW ONLY: no files deleted. Archive content checks run only with --delete.")
        print("Candidate sizes/mtime must match the supplied inventory; mismatches are never forced.")
        return 0
    print("\nPERMANENT DELETE, not recycle-bin. No automatic recovery.", flush=True)
    if not ask("Confirm no training/evaluation/download jobs use these files. Type DELETE to continue: "):
        print("Cancelled: no deletion.")
        return 0
    deleted_count = 0
    deleted_bytes = 0
    for group in selected:
        if not ready[group]:
            continue
        print("\n[{}] {}".format(group, NOTICES[group]), flush=True)
        print("Only this group's previously listed files will be processed.", flush=True)
        if not ask("Confirm these conditions and delete this group? Type DELETE (Enter skips): "):
            print("Group retained.", flush=True)
            continue
        try:
            check_completed_models(PROJECT, group)
        except (ValueError, OSError) as error:
            print("GROUP SKIPPED: " + str(error), flush=True)
            issues += 1
            continue
        for row, saved_signature in ready[group]:
            try:
                deleted_bytes += remove_one(PROJECT, row, saved_signature)
                deleted_count += 1
            except (ValueError, OSError, RuntimeError, zipfile.BadZipFile) as error:
                print("SKIPPED: {}: {}".format(row[1], error), flush=True)
                issues += 1
    print("\nDeleted {} files, {:.3f} GiB logical size. No automatic recovery.".format(
        deleted_count, deleted_bytes / 2**30), flush=True)
    print("Skipped/failed checks: {}. Actual freed disk space may differ.".format(issues), flush=True)
    print("Formal checkpoints, source code, extracted data and evaluation results were not deletion targets.")
    return 1 if issues else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Earlier completed deletions are not rolled back.", flush=True)
        sys.exit(130)

