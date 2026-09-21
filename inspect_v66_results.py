"""Compact decision report for a V6.6 structure-side-information pilot."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main(args: argparse.Namespace) -> None:
    path = Path(args.summary)
    rows = list(csv.DictReader(path.open("r", newline="", encoding="utf-8")))
    overall = {
        row["structure_mode"]: row for row in rows if row["snr_db"] == "ALL"
    }
    if "receiver" not in overall or "oracle" not in overall:
        raise RuntimeError("summary must contain ALL receiver and oracle rows")
    receiver = overall["receiver"]
    oracle = overall["oracle"]

    def delta(field: str) -> float:
        return float(oracle[field]) - float(receiver[field])

    print("V6.6 paired development diagnostic")
    print("samples_per_mode=%s" % oracle["samples"])
    print("oracle_minus_receiver_utility=%+.6f" % delta("candidate_utility"))
    print("oracle_minus_receiver_clip=%+.6f" % delta("candidate_clip_image"))
    print("oracle_minus_receiver_psnr=%+.6f" % delta("candidate_psnr"))
    print("oracle_minus_receiver_ssim=%+.6f" % delta("candidate_ssim"))
    print(
        "oracle_minus_receiver_structure_f1=%+.6f"
        % delta("candidate_source_structure_f1")
    )
    print(
        "metric_safe_positive_rate receiver=%.3f oracle=%.3f"
        % (
            float(receiver["metric_safe_positive_rate"]),
            float(oracle["metric_safe_positive_rate"]),
        )
    )
    utility_ok = delta("candidate_utility") > args.min_utility_gain
    structure_ok = delta("candidate_source_structure_f1") > args.min_structure_gain
    safe_ok = (
        float(oracle["metric_safe_positive_rate"])
        >= float(receiver["metric_safe_positive_rate"])
    )
    if utility_ok and structure_ok and safe_ok:
        decision = "GO_TO_NOISY_STRUCTURE_CHANNEL"
    elif sum((utility_ok, structure_ok, safe_ok)) >= 2:
        decision = "PROMISING_REVIEW_PANELS_BEFORE_CHANNEL"
    else:
        decision = "STOP_OR_REDESIGN_STRUCTURE_CONTROL"
    print("pilot_decision=%s" % decision)
    print("This is a development gate, not a final statistical claim.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--min-utility-gain", type=float, default=0.002)
    parser.add_argument("--min-structure-gain", type=float, default=0.01)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
