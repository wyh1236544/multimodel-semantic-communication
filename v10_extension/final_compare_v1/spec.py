"""Strict public plan, exact AWG resource accounting, no channel truth."""
import copy
import hashlib
import json
import math
from pathlib import Path
from business_link.engine import WIRE
from physical_link.digital import fec_size

PLAN = Path(__file__).with_name("plan.json")
SMOKE_CONDITIONS = ("good_p09", "moderate_p15", "stress_p09")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def load_plan(path=PLAN):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    template = json.loads(PLAN.read_text(encoding="utf-8"))
    if set(value) != set(template) or value["schema"] != "final-compare-v1":
        raise ValueError("Unknown/missing comparison-plan key")
    for key in ("slot_microseconds", "laser_max_dbm", "image_psnr_target_db",
                "normalized_wer_target"):
        v = value[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError("Finite public parameter required: " + key)
    if not .5 <= value["slot_microseconds"] <= 20 or not -20 <= value["laser_max_dbm"] <= 30:
        raise ValueError("Unsupported time/laser cap")
    if not 0 <= value["normalized_wer_target"] <= 1 or not 0 < value["image_psnr_target_db"] <= 60:
        raise ValueError("Invalid task threshold")
    for key, lo, hi in (("jpeg_quality_min", 1, 95), ("jpeg_quality_max", 1, 95),
                        ("bootstrap_resamples", 100, 20000)):
        if type(value[key]) is not int or not lo <= value[key] <= hi:
            raise ValueError("Invalid integer: " + key)
    if value["jpeg_quality_min"] > value["jpeg_quality_max"]:
        raise ValueError("Reversed JPEG quality range")
    if value["digital_policy"] not in ("16qam_none", "16qam_conv", "16qam_repeat2", "qpsk_conv"):
        raise ValueError("Unsupported digital policy")
    if type(value["jpeg_subsampling"]) is not int or value["jpeg_subsampling"] not in (0, 1, 2):
        raise ValueError("Invalid JPEG subsampling")
    if value["analog_equalizer"] not in ("zf", "lmmse"):
        raise ValueError("Unsupported continuous equalizer")
    if not isinstance(value["conditions"], list) or not 1 <= len(value["conditions"]) <= 24:
        raise ValueError("Need a bounded condition list")
    ids = set()
    for c in value["conditions"]:
        if set(c) != {"id", "case", "launch_signal_dbm"}:
            raise ValueError("Incomplete/unknown condition")
        if not c["id"] or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for ch in c["id"]):
            raise ValueError("Unsafe condition ID")
        p = c["launch_signal_dbm"]
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not -60 <= p <= 0:
            raise ValueError("Unsupported launch power")
        if c["id"] in ids or not isinstance(c["case"], str):
            raise ValueError("Duplicate/invalid condition")
        ids.add(c["id"])
    return value, canonical_hash(value)


def sampled_count(data_symbols, waveform):
    # Pulse.shape: (symbol_count-1)*sps + span*sps + 1; integer SPS.
    return ((256 + int(data_symbols) - 1) * waveform["sps"]
            + waveform["span"] * waveform["sps"] + 1 + waveform["guard_samples"])


def packet_symbols(waveform, policy):
    uncoded = (WIRE.size + waveform["packet_bytes"] + 4) * 8
    coded = uncoded if policy.endswith("_none") else fec_size(uncoded)
    wire = coded * (2 if policy == "16qam_repeat2" else 1)
    return math.ceil(wire / (2 if policy == "qpsk_conv" else 4))


def digital_samples(byte_lengths, waveform, policy):
    packets = sum(max(1, math.ceil(n / waveform["packet_bytes"])) for n in byte_lengths)
    return sampled_count(0, waveform) + packets * sampled_count(packet_symbols(waveform, policy), waveform)


def resources(optical, plan):
    w = optical["waveform"]
    fs = w["fs_hz"]
    if fs != 64e9 or w["sps"] != 4:
        raise ValueError("This matched comparison fixes 64GSa/s and integer 4 SPS")
    if optical["receiver"]["carrier_recovery"] != "pilot_constant":
        raise ValueError("Use shared pilot_constant DSP; QAM DD phase loop is not an analog loop")
    if optical["laser"]["linewidth_hz"] or optical["receiver"]["lo_linewidth_hz"] or optical["receiver"]["frequency_offset_hz"]:
        raise ValueError("This frozen comparison assumes zero linewidth/CFO; declare, do not silently disable")
    if optical["tx_edfa"]["control"] != "gain":
        raise ValueError("Public launch-power regulator currently requires TX EDFA gain mode")
    limit = int(math.floor(plan["slot_microseconds"] * 1e-6 * fs + 1e-6))
    probe = sampled_count(0, w)
    digital = digital_samples([4622], w, plan["digital_policy"])
    if digital > limit:
        raise ValueError("Semantic digital frame exceeds common slot; enlarge PUBLIC slot or change public coding")
    burst_overhead = sampled_count(0, w)
    repeats = (limit - probe - burst_overhead) // (2304 * w["sps"])
    if repeats < 1:
        raise ValueError("Continuous payload cannot fit common slot")
    if limit / fs > optical["turbulence"]["coherence_time_s"]:
        raise ValueError("Common slot exceeds slow-fading coherence time")
    return dict(slot_samples=limit, slot_seconds=limit/fs, probe_samples=probe,
                semantic_digital_samples=digital, analog_repeats=int(repeats),
                analog_samples=probe + sampled_count(2304*repeats, w),
                symbol_rate_baud=fs/w["sps"], occupied_complex_bandwidth_hz=(1+w["rolloff"])*fs/w["sps"],
                gross_digital_bps=(2 if plan["digital_policy"] == "qpsk_conv" else 4)*fs/w["sps"])


def effective_conditions(cases, plan, smoke=False):
    conditions = copy.deepcopy(plan["conditions"])
    if smoke:
        conditions = [c for c in conditions if c["id"] in SMOKE_CONDITIONS]
        if not conditions:
            conditions = copy.deepcopy(plan["conditions"][:min(3, len(plan["conditions"]))])
    for c in conditions:
        if c["case"] not in cases:
            raise ValueError("Comparison case absent from physical configuration: " + c["case"])
        resources(cases[c["case"]], plan)
    return conditions

