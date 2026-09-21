"""Receiver-only prompt and candidate features for V9-B selective generation."""
from __future__ import annotations

import re
from collections import Counter
from typing import List, Sequence

import torch
from torch import Tensor


PROMPT_FEATURE_NAMES: List[str] = [
    "prompt_nonempty",
    "prompt_length",
    "unique_token_ratio",
    "inverse_dominant_ratio",
    "inverse_repeat_ratio",
    "inverse_max_run",
    "alphabetic_ratio",
    "content_token_ratio",
]

_TOKEN_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|\d+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can", "for", "from",
    "here", "i", "in", "is", "it", "of", "on", "see", "the", "there",
    "this", "to", "we", "with",
}


def _tokens(text: str) -> List[str]:
    return [value.lower() for value in _TOKEN_RE.findall(str(text))]


def sanitize_received_caption(text: str, max_tokens: int = 60) -> str:
    """Collapse decoder repetitions without adding source-side information."""
    tokens = _tokens(text)
    if not tokens:
        return "a natural photograph"
    result: List[str] = []
    counts: Counter[str] = Counter()
    previous = None
    run = 0
    for token in tokens:
        run = run + 1 if token == previous else 1
        previous = token
        limit = 2 if token in _STOPWORDS else 3
        if run > 2 or counts[token] >= limit:
            continue
        result.append(token)
        counts[token] += 1
        if len(result) >= int(max_tokens):
            break
    return " ".join(result) if result else "a natural photograph"


def prompt_quality_features(
    texts: Sequence[str], device: torch.device | None = None
) -> Tensor:
    rows = []
    for text in texts:
        tokens = _tokens(text)
        count = len(tokens)
        if count == 0:
            rows.append([0.0] * len(PROMPT_FEATURE_NAMES))
            continue
        frequencies = Counter(tokens)
        repeats = sum(max(value - 1, 0) for value in frequencies.values())
        max_run = 1
        run = 1
        for index in range(1, count):
            run = run + 1 if tokens[index] == tokens[index - 1] else 1
            max_run = max(max_run, run)
        alphabetic = sum(token.isalpha() for token in tokens) / count
        content = sum(token not in _STOPWORDS for token in tokens) / count
        rows.append(
            [
                1.0,
                min(count / 60.0, 1.0),
                len(frequencies) / count,
                1.0 - max(frequencies.values()) / count,
                1.0 - repeats / count,
                1.0 / max_run,
                alphabetic,
                content,
            ]
        )
    return torch.tensor(rows, dtype=torch.float32, device=device).clamp(0.0, 1.0)


def pre_generation_reliability(route_features: Tensor, prompt_features: Tensor) -> Tensor:
    """Transparent pre-gate score; all indices are receiver-side quantities."""
    route = route_features.float()
    prompt = prompt_features.float()
    decoder_confidence = (
        0.35 * route[:, 1]
        + 0.25 * route[:, 2]
        + 0.15 * route[:, 3]
        + 0.15 * route[:, 4]
        + 0.10 * route[:, 5]
    )
    lexical_quality = (
        0.25 * prompt[:, 2]
        + 0.20 * prompt[:, 3]
        + 0.20 * prompt[:, 4]
        + 0.15 * prompt[:, 5]
        + 0.20 * prompt[:, 7]
    )
    return (0.60 * decoder_confidence + 0.40 * lexical_quality).clamp(0.0, 1.0)
