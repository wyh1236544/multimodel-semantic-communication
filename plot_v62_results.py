"""Plot publication-style V6.2 curves from v62_quality_adaptive_eval.py."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["snr_db"] != "ALL"]
    if not rows:
        raise RuntimeError("No per-SNR rows found in %s" % path)
    rows.sort(key=lambda row: float(row["snr_db"]))
    return rows


def _values(rows, name):
    if name not in rows[0] or rows[0][name] == "":
        return None
    return [float(row[name]) for row in rows]


def _plot_available(axis, snr, rows, specifications):
    for column, label, style in specifications:
        values = _values(rows, column)
        if values is not None:
            axis.plot(snr, values, style, linewidth=2.0, markersize=5, label=label)
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)


def main(args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required only for plotting. Install it with "
            "`pip install matplotlib` in the torch environment."
        ) from exc

    input_path = Path(args.summary)
    output_path = Path(args.output) if args.output else input_path.with_name("v62_summary.png")
    rows = _read_rows(input_path)
    snr = [float(row["snr_db"]) for row in rows]
    overlap = sum(int(float(row.get("policy_data_overlap_count", 0))) for row in rows)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    _plot_available(
        axes[0, 0],
        snr,
        rows,
        [
            ("probe_psnr", "Fixed regression", "o--"),
            ("snr_baseline_psnr", "SNR-only SD", "s--"),
            ("adaptive_psnr", "Text-adaptive regression", "^-"),
            ("quality_prompt_psnr", "Quality + prompt-only", "d-"),
            ("selected_psnr", "Full V6.2", "o-"),
        ],
    )
    axes[0, 0].set(title="Image fidelity", xlabel="SNR (dB)", ylabel="PSNR (dB)")

    _plot_available(
        axes[0, 1],
        snr,
        rows,
        [
            ("probe_ssim", "Fixed regression", "o--"),
            ("snr_baseline_ssim", "SNR-only SD", "s--"),
            ("adaptive_ssim", "Text-adaptive regression", "^-"),
            ("quality_prompt_ssim", "Quality + prompt-only", "d-"),
            ("selected_ssim", "Full V6.2", "o-"),
        ],
    )
    axes[0, 1].set(title="Structural fidelity", xlabel="SNR (dB)", ylabel="SSIM")

    _plot_available(
        axes[1, 0],
        snr,
        rows,
        [
            ("probe_text_accuracy", "Probe text accuracy", "o--"),
            ("adaptive_text_accuracy", "Adaptive text accuracy", "o-"),
        ],
    )
    axes[1, 0].set(
        title="Text reliability and protection",
        xlabel="SNR (dB)",
        ylabel="Token accuracy",
        ylim=(0.0, 1.02),
    )
    repetition_axis = axes[1, 0].twinx()
    repetition = _values(rows, "mean_text_repetition_factor")
    if repetition is not None:
        repetition_axis.plot(snr, repetition, "k:^", linewidth=1.5, label="Text repetition")
        repetition_axis.set_ylabel("Mean repetition factor")

    _plot_available(
        axes[1, 1],
        snr,
        rows,
        [
            ("generation_rate", "Quality generation rate", "o-"),
            ("snr_baseline_generation_rate", "SNR-only generation rate", "s--"),
            ("feedback_rate", "Text feedback rate", "^-"),
        ],
    )
    axes[1, 1].set(
        title="Routing, feedback and latency",
        xlabel="SNR (dB)",
        ylabel="Rate",
        ylim=(0.0, 1.02),
    )
    latency_axis = axes[1, 1].twinx()
    latency = _values(rows, "selected_diffusion_latency_ms")
    if latency is not None:
        latency_axis.plot(snr, latency, "k:d", linewidth=1.5, label="Mean SD latency")
        latency_axis.set_ylabel("Latency per sample (ms)")

    status = "STRICT HOLDOUT: PASS" if overlap == 0 else "DATA OVERLAP: %d" % overlap
    figure.suptitle("V6.2 Quality-Adaptive Semantic Communication — %s" % status, fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), dpi=args.dpi)
    plt.close(figure)
    print("Saved %s" % output_path)
    print("policy_data_overlap_count=%d" % overlap)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot V6.2 evaluation summary")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output")
    parser.add_argument("--dpi", type=int, default=180)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
