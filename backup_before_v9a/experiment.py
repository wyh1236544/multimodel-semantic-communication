from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

from data import build_dataloader
from losses import MultimodalCommunicationLoss
from models import JointImageTextSemanticCommunicationModel
from utils.checkpoint import (
    load_checkpoint,
    load_state_dict_flexible,
    save_checkpoint,
)
from utils.config import save_config
from utils.metrics import (
    MetricAccumulator,
    batch_image_metrics,
    corpus_bleu2,
    token_accuracy,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _build_training_schedule(train_cfg: Dict[str, Any], total_epochs: int):
    """Expand fractional V7 stages to exactly ``total_epochs`` entries."""
    stages = train_cfg.get("stages")
    if not stages:
        return [
            {"name": "joint", "semantic_ablation": "full", "loss_weights": {}}
            for _ in range(total_epochs)
        ]
    if not isinstance(stages, list) or not stages:
        raise ValueError("train.stages must be a non-empty list")
    fractions = [max(float(stage.get("fraction", 0.0)), 0.0) for stage in stages]
    if sum(fractions) <= 0:
        raise ValueError("train.stages fractions must have a positive sum")
    raw = [total_epochs * value / sum(fractions) for value in fractions]
    counts = [int(value) for value in raw]
    # At normal run lengths, every declared stage should be exercised once.
    if total_epochs >= len(stages):
        counts = [max(value, 1) for value in counts]
    while sum(counts) > total_epochs:
        candidates = [i for i, value in enumerate(counts) if value > 1]
        counts[candidates[-1] if candidates else len(counts) - 1] -= 1
    order = sorted(range(len(raw)), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    cursor = 0
    while sum(counts) < total_epochs:
        counts[order[cursor % len(order)]] += 1
        cursor += 1
    schedule = []
    for stage, count in zip(stages, counts):
        schedule.extend([dict(stage)] * count)
    return schedule


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _apply_stage_trainability(
    model: torch.nn.Module,
    initially_trainable: set[str],
    stage: Dict[str, Any],
) -> None:
    """Optionally freeze all but explicitly selected modules for one stage."""
    prefixes = tuple(stage.get("trainable_modules", []))
    for name, parameter in model.named_parameters():
        allowed = name in initially_trainable
        if prefixes:
            allowed = allowed and name.startswith(prefixes)
        parameter.requires_grad_(allowed)


def create_model(config: Dict[str, Any], device: torch.device):
    model_config = dict(config["model"])
    if device.type != "cuda":
        model_config["vae_use_fp16"] = False
    model = JointImageTextSemanticCommunicationModel(
        model_config,
        modality_dropout=config.get("modality_dropout", {}),
    )
    model.channel.configure(config.get("channel", {}).get("type", "awgn"))
    model.to(device)
    return model


def build_loaders(
    config: Dict[str, Any],
    seed: int,
    train_samples: Optional[int],
    val_samples: Optional[int],
):
    data = config["data"]
    model = config["model"]
    train_cfg = config["train"]
    test_cfg = config["test"]
    train_loader = build_dataloader(
        image_root=data["train_image_root"],
        annotation_file=data["train_annotation_file"],
        image_size=int(model["image_size"]),
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg.get("num_workers", 4)),
        train=True,
        caption_mode=data.get("train_caption_mode", "random"),
        max_samples=train_samples,
        seed=seed,
    )
    val_loader = build_dataloader(
        image_root=data["val_image_root"],
        annotation_file=data["val_annotation_file"],
        image_size=int(model["image_size"]),
        batch_size=int(test_cfg.get("batch_size", train_cfg["batch_size"])),
        num_workers=int(test_cfg.get("num_workers", train_cfg.get("num_workers", 4))),
        train=False,
        caption_mode=data.get("val_caption_mode", "shortest"),
        max_samples=val_samples,
        seed=seed,
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    snr_db: float,
    amp: bool,
    semantic_ablation: str = "full",
    utility_weight: float = 0.5,
) -> Dict[str, Any]:
    model.eval()
    meter = MetricAccumulator()
    references = []
    hypotheses = []
    correct_tokens = 0.0
    valid_tokens = 0
    common_image_embeddings = []
    common_text_embeddings = []
    receiver_common_embeddings = []
    receiver_base_semantics = []
    source_text_semantics = []
    common_caption_references = []
    common_caption_hypotheses = []
    common_caption_correct = 0.0
    common_caption_valid = 0
    autocast_enabled = bool(amp and device.type == "cuda")
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        texts = batch["texts"]
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            outputs = model(
                images,
                texts,
                snr_db=snr_db,
                semantic_ablation=semantic_ablation,
                disable_modality_dropout=True,
                utility_weight=utility_weight,
            )
        metrics = batch_image_metrics(outputs["image"], images)
        for name, values in metrics.items():
            meter.add(name, values)
        correct, count = token_accuracy(
            outputs["text_logits"],
            outputs["text_input_ids"],
            outputs["text_attention_mask"],
        )
        correct_tokens += correct
        valid_tokens += count
        references.extend(texts)
        hypotheses.extend(model.decode_text_logits(outputs["text_logits"]))
        meter.add("allocated_image_tokens", outputs["allocated_image_tokens"])
        meter.add("allocated_text_tokens", outputs["allocated_text_tokens"])
        meter.add("channel_uses", outputs["channel_uses"].reshape(1))
        meter.add(
            "text_channel_reliability",
            outputs.get(
                "text_channel_reliability",
                outputs["allocated_text_tokens"].new_ones(
                    outputs["allocated_text_tokens"].shape
                ),
            ),
        )
        if outputs.get("common_coarse_latents") is not None:
            kernel = int(outputs["common_coarse_blur_kernel"].item())
            target_coarse = torch.nn.functional.avg_pool2d(
                outputs["vae_latents"].detach().float(),
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2,
            )
            coarse_mse = (
                outputs["common_coarse_latents"].float() - target_coarse
            ).square().mean(dim=(1, 2, 3))
            meter.add("common_coarse_latent_mse", coarse_mse)
        common_image_embeddings.append(
            torch.nn.functional.normalize(
                outputs["common_image_tokens"].float().mean(1), dim=-1
            ).cpu()
        )
        common_text_embeddings.append(
            torch.nn.functional.normalize(
                outputs["common_text_tokens"].float().mean(1), dim=-1
            ).cpu()
        )
        receiver_common_embeddings.append(
            torch.nn.functional.normalize(
                outputs["decoded_common_tokens"].float().mean(1), dim=-1
            ).cpu()
        )
        if outputs.get("common_receiver_semantic") is not None:
            receiver_base_semantics.append(
                outputs["common_receiver_semantic"].detach().float().cpu()
            )
            source_text_semantics.append(
                outputs["common_text_semantic_target"].detach().float().cpu()
            )
        common_logits = outputs.get("common_base_text_logits")
        if common_logits is not None:
            common_length = min(
                int(outputs["common_caption_max_tokens"].item()),
                common_logits.size(1),
                outputs["text_input_ids"].size(1),
            )
            common_correct, common_count = token_accuracy(
                common_logits[:, :common_length],
                outputs["text_input_ids"][:, :common_length],
                outputs["text_attention_mask"][:, :common_length],
            )
            common_caption_correct += common_correct
            common_caption_valid += common_count
            common_caption_references.extend(texts)
            common_caption_hypotheses.extend(
                model.decode_text_logits(common_logits)
            )
    result = meter.as_dict()
    common_image = torch.cat(common_image_embeddings, dim=0)
    common_text = torch.cat(common_text_embeddings, dim=0)
    receiver_common = torch.cat(receiver_common_embeddings, dim=0)
    labels = torch.arange(common_image.size(0))

    def retrieval_recall(query, target, k):
        k = min(int(k), target.size(0))
        topk = (query @ target.transpose(0, 1)).topk(k, dim=1).indices
        return float(topk.eq(labels[:, None]).any(dim=1).float().mean())

    result.update(
        {
            "token_accuracy": correct_tokens / max(valid_tokens, 1),
            "bleu2": corpus_bleu2(references, hypotheses),
            "sample_count": len(references),
            "snr_db": float(snr_db),
            "semantic_ablation": semantic_ablation,
            "utility_weight": float(utility_weight),
            "allocated_common_tokens": int(model.num_common_tokens),
            "channel_uses": int(round(result.get("channel_uses", model.num_channel_uses()))),
            "bandwidth": model.get_bandwidth_info(),
            "common_image_text_cosine": float(
                (common_image * common_text).sum(-1).mean()
            ),
            "common_i2t_top1": retrieval_recall(common_image, common_text, 1),
            "common_t2i_top1": retrieval_recall(common_text, common_image, 1),
            "receiver_common_image_top1": retrieval_recall(
                receiver_common, common_image
                , 1
            ),
            "receiver_common_text_top1": retrieval_recall(
                receiver_common, common_text
                , 1
            ),
            "common_embedding_std": float(
                torch.cat([common_image, common_text], dim=0).std(dim=0).mean()
            ),
        }
    )
    if receiver_base_semantics:
        receiver_base = torch.cat(receiver_base_semantics, dim=0)
        source_text = torch.cat(source_text_semantics, dim=0)
        result.update(
            {
                "common_semantic_text_cosine": float(
                    (receiver_base * source_text).sum(-1).mean()
                ),
                "common_semantic_text_r1": retrieval_recall(
                    receiver_base, source_text, 1
                ),
                "common_semantic_text_r5": retrieval_recall(
                    receiver_base, source_text, 5
                ),
                "common_semantic_text_r10": retrieval_recall(
                    receiver_base, source_text, 10
                ),
                "common_caption_token_accuracy": (
                    common_caption_correct / max(common_caption_valid, 1)
                ),
                "common_caption_bleu2": corpus_bleu2(
                    common_caption_references,
                    common_caption_hypotheses,
                ),
            }
        )
    return result


def train_one_seed(
    config: Dict[str, Any],
    output_dir: str | Path,
    seed: int,
    init_checkpoint: Optional[str] = None,
    allow_shape_expansion: bool = False,
    train_samples: Optional[int] = None,
    val_samples: Optional[int] = 500,
    epochs: Optional[int] = None,
    device_name: str = "auto",
    lr_override: Optional[float] = None,
) -> Dict[str, Any]:
    seed_everything(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(device_name)
    print(f"[seed {seed}] device={device}")
    model = create_model(config, device)
    print(f"[seed {seed}] bandwidth={model.get_bandwidth_info()}")

    load_report = None
    if init_checkpoint:
        checkpoint = load_checkpoint(init_checkpoint)
        load_report = load_state_dict_flexible(
            model,
            checkpoint,
            allow_shape_expansion=allow_shape_expansion,
        )
        del checkpoint
        print(
            f"[seed {seed}] checkpoint exact={load_report['exact_count']} "
            f"expanded={len(load_report['expanded'])} "
            f"missing={len(load_report['missing'])}"
        )
        if load_report["expanded"]:
            for item in load_report["expanded"]:
                print("  expanded", item)
        if any(
            name.startswith("common_base_")
            for name in load_report["missing"]
        ):
            model.initialize_common_base_from_existing()
            print("[seed %d] initialized V8-A common base from V7 receiver" % seed)

    train_loader, val_loader = build_loaders(
        config, seed, train_samples=train_samples, val_samples=val_samples
    )
    criterion = MultimodalCommunicationLoss(config.get("loss", {})).to(device)
    train_cfg = config["train"]
    named_trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable = [parameter for _, parameter in named_trainable]
    initially_trainable = {name for name, _ in named_trainable}
    if not trainable:
        raise RuntimeError("No trainable model parameters")
    parameter_count = sum(parameter.numel() for parameter in trainable)
    print(f"[seed {seed}] trainable_parameters={parameter_count:,}")
    base_lr = float(lr_override or train_cfg.get("lr", 1e-6))
    high_lr = float(lr_override or train_cfg.get("high_lr", 2e-5))
    high_prefixes = tuple(train_cfg.get("high_lr_modules", []))
    high_parameters = [
        parameter
        for name, parameter in named_trainable
        if high_prefixes and name.startswith(high_prefixes)
    ]
    high_ids = {id(parameter) for parameter in high_parameters}
    base_parameters = [parameter for parameter in trainable if id(parameter) not in high_ids]
    parameter_groups = []
    if base_parameters:
        parameter_groups.append({"params": base_parameters, "lr": base_lr})
    if high_parameters:
        parameter_groups.append({"params": high_parameters, "lr": high_lr})
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )
    print(
        f"[seed {seed}] optimizer base_lr={base_lr:g} "
        f"high_lr={high_lr:g} high_lr_tensors={len(high_parameters)}"
    )
    amp = bool(train_cfg.get("amp", True) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    grad_accumulation = max(int(train_cfg.get("grad_accumulation_steps", 1)), 1)
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    total_epochs = int(epochs or train_cfg.get("epochs", 8))
    training_schedule = _build_training_schedule(train_cfg, total_epochs)
    final_stage_name = str(training_schedule[-1].get("name", "joint"))
    snr_choices = list(config.get("channel", {}).get("train_snr_list", [10]))
    utility_choices = list(train_cfg.get("utility_weight_choices", [0.5]))
    log_interval = max(int(train_cfg.get("log_interval", 20)), 1)
    best_score = -float("inf")
    history = []
    best_path = output_dir / f"seed{seed}.pth"
    save_config(config, output_dir / f"config_seed{seed}.yaml")
    if load_report is not None:
        with (output_dir / f"load_report_seed{seed}.json").open("w", encoding="utf-8") as handle:
            json.dump(load_report, handle, ensure_ascii=False, indent=2)

    for epoch in range(1, total_epochs + 1):
        stage = training_schedule[epoch - 1]
        stage_name = str(stage.get("name", "joint"))
        stage_ablation = str(stage.get("semantic_ablation", "full"))
        _apply_stage_trainability(model, initially_trainable, stage)
        criterion.set_weight_overrides(stage.get("loss_weights", {}))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = MetricAccumulator()
        started = time.time()
        for step, batch in enumerate(train_loader, start=1):
            images = batch["images"].to(device, non_blocking=True)
            texts = batch["texts"]
            snr_db = random.choice(snr_choices)
            utility_weight = random.choice(utility_choices)
            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model(
                    images,
                    texts,
                    snr_db=snr_db,
                    utility_weight=utility_weight,
                    semantic_ablation=stage_ablation,
                )
                loss, terms = criterion(outputs, images)
                scaled_loss = loss / grad_accumulation
            scaler.scale(scaled_loss).backward()
            if step % grad_accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            for name, value in terms.items():
                running.add(name, float(value.detach().item()))
            if step % log_interval == 0 or step == len(train_loader):
                print(
                    f"[seed {seed}] epoch={epoch}/{total_epochs} "
                    f"stage={stage_name} "
                    f"step={step}/{len(train_loader)} "
                    f"loss={running.mean('total'):.5f} snr={snr_db}"
                )

        # Use identical VAE sampling/channel noise each epoch, while restoring
        # the training RNG stream afterwards. Otherwise checkpoint selection is
        # partly driven by validation noise rather than model quality.
        rng_state = _capture_rng_state()
        seed_everything(seed * 10000 + 777)
        try:
            validation = evaluate_model(
                model,
                val_loader,
                device=device,
                snr_db=20.0,
                amp=amp,
            )
        finally:
            _restore_rng_state(rng_state)
        # PSNR is primary; token accuracy prevents a completely image-only solution.
        score = float(validation["psnr"] + 0.5 * validation["token_accuracy"])
        epoch_record = {
            "epoch": epoch,
            "stage": stage_name,
            "training_semantic_ablation": stage_ablation,
            "loss_weights": dict(criterion.weights),
            "train": running.as_dict(),
            "validation_20db": validation,
            "score": score,
            "seconds": time.time() - started,
        }
        history.append(epoch_record)
        print(
            f"[seed {seed}] validation psnr={validation['psnr']:.4f} "
            f"ssim={validation['ssim']:.4f} "
            f"token_acc={validation['token_accuracy']:.4f}"
        )
        eligible_for_best = stage_name == final_stage_name
        if eligible_for_best and score > best_score:
            best_score = score
            save_checkpoint(
                best_path,
                model,
                config=config,
                epoch=epoch,
                metrics=validation,
            )
            print(f"[seed {seed}] saved best checkpoint: {best_path}")
        with (output_dir / f"history_seed{seed}.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)

    return {
        "seed": seed,
        "checkpoint": str(best_path),
        "best_score": best_score,
        "history": history,
        "bandwidth": model.get_bandwidth_info(),
    }
