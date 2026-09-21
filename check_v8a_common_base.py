"""Runtime contract check for the V8-A common semantic base layer."""
from __future__ import annotations

import argparse

import torch

from utils.checkpoint import load_checkpoint, load_state_dict_flexible
from utils.config import load_config, prepare_config
from utils.experiment import create_model, seed_everything, select_device


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    seed_everything(42)
    device = select_device(args.device)
    config = prepare_config(
        load_config(args.config),
        preset="original_4608",
        vae_path=args.vae_path,
        text_model_path=args.text_model_path,
        data_root=args.data_root,
    )
    model = create_model(config, device)
    if args.checkpoint:
        report = load_state_dict_flexible(
            model,
            load_checkpoint(args.checkpoint),
            allow_shape_expansion=False,
        )
        print(
            f"checkpoint exact={report['exact_count']} "
            f"missing={len(report['missing'])}"
        )
    model.eval()
    images = torch.rand(2, 3, 128, 128, device=device)
    texts = [
        "A person is standing beside a motorcycle on a street.",
        "Two dogs are playing together in a grassy park.",
    ]
    common = model(
        images,
        texts,
        snr_db=0.0,
        semantic_ablation="common_only",
        disable_modality_dropout=True,
    )
    full = model(
        images,
        texts,
        snr_db=0.0,
        semantic_ablation="full",
        disable_modality_dropout=True,
    )
    assert tuple(common["z_transmitted"].shape[1:]) == (32, 16)
    assert int(common["channel_uses"].item()) == 512
    assert float(common["allocated_image_tokens"].max()) == 0.0
    assert float(common["allocated_text_tokens"].max()) == 0.0
    assert int(common["physical_common_only"].item()) == 1
    assert tuple(full["z_transmitted"].shape[1:]) == (288, 16)
    assert int(full["channel_uses"].item()) == 4608
    assert tuple(common["common_base_text_logits"].shape[:2]) == (2, 24)
    assert tuple(common["common_receiver_semantic"].shape) == (2, 256)
    assert tuple(common["common_coarse_latents"].shape) == (2, 4, 16, 16)
    coarse_images = model.vae_wrapper.decode_latents(
        common["common_coarse_latents"]
    )
    assert tuple(coarse_images.shape) == (2, 3, 128, 128)
    captions = model.decode_text_logits(common["common_base_text_logits"])
    print("V8-A contract check passed")
    print("common payload: C32 x 16 = 512 real channel uses")
    print("full payload: C32 + I224 + T32 = 4608 real channel uses")
    print("untrained/restored common captions:", captions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v8a_common_base.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--text-model-path", required=True)
    parser.add_argument("--device", default="auto")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
