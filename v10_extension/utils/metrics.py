from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, Sequence

import torch
import torch.nn.functional as F


def ssim_per_image(x: torch.Tensor, y: torch.Tensor, window: int = 11) -> torch.Tensor:
    x = x.float().clamp(0, 1)
    y = y.float().clamp(0, 1)
    padding = window // 2
    mu_x = F.avg_pool2d(x, window, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, window, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(x * x, window, 1, padding) - mu_x.square()
    sigma_y = F.avg_pool2d(y * y, window, 1, padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(x * y, window, 1, padding) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2) + 1e-12
    )
    return score.flatten(1).mean(1)


def batch_image_metrics(reconstructed: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    mse = (reconstructed.float() - target.float()).square().flatten(1).mean(1)
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    return {"mse": mse, "psnr": psnr, "ssim": ssim_per_image(reconstructed, target)}


def token_accuracy(logits: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor) -> tuple[float, int]:
    length = min(logits.size(1), ids.size(1))
    predictions = logits[:, :length].argmax(-1)
    valid = mask[:, :length].bool()
    correct = ((predictions == ids[:, :length]) & valid).sum().item()
    count = valid.sum().item()
    return float(correct), int(count)


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(len(tokens) - n + 1, 0)))


def sentence_bleu2(reference: str, hypothesis: str) -> float:
    ref = reference.lower().strip().split()
    hyp = hypothesis.lower().strip().split()
    if not ref or not hyp:
        return 0.0
    precisions = []
    for n in (1, 2):
        h = _ngrams(hyp, n)
        r = _ngrams(ref, n)
        total = sum(h.values())
        clipped = sum(min(count, r[gram]) for gram, count in h.items())
        precisions.append((clipped + 1.0) / (total + 1.0))
    brevity = 1.0 if len(hyp) >= len(ref) else math.exp(1.0 - len(ref) / len(hyp))
    return brevity * math.exp(0.5 * sum(math.log(p) for p in precisions))


def corpus_bleu2(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    scores = [sentence_bleu2(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    return float(sum(scores) / max(len(scores), 1))


class MetricAccumulator:
    def __init__(self):
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)

    def add(self, name: str, values: torch.Tensor | float, count: int | None = None) -> None:
        if isinstance(values, torch.Tensor):
            values = values.detach().float()
            self.sums[name] += values.sum().item()
            self.counts[name] += values.numel()
        else:
            self.sums[name] += float(values)
            self.counts[name] += int(1 if count is None else count)

    def mean(self, name: str) -> float:
        return self.sums[name] / max(self.counts[name], 1)

    def as_dict(self) -> Dict[str, float]:
        return {name: self.mean(name) for name in sorted(self.sums)}
