"""Conservative observable-only combination; not a certified semantic error bound."""
import numpy as np


def combine(copies, pilot_variance, block_error_floor, locked, equalizer):
    copies=np.asarray(copies, complex)
    if copies.ndim!=2 or not len(copies) or not copies.shape[1] or not np.isfinite(copies).all():
        raise ValueError("Nonempty finite repeated observations required")
    if equalizer not in ("zf","lmmse"): raise ValueError("Unknown equalizer")
    if not np.isfinite(pilot_variance) or not np.isfinite(block_error_floor) or min(pilot_variance,block_error_floor)<0:
        raise ValueError("Nonnegative finite pilot estimates required")
    repeats=len(copies); average=copies.mean(0)
    white=float(pilot_variance/repeats)
    split=None
    if repeats>1 and locked:
        a,b=copies[::2],copies[1::2]
        split=float(np.mean(abs(a.mean(0)-b.mean(0))**2)/(1/len(a)+1/len(b))/repeats)
    effective=max(white,float(block_error_floor),split or 0.)
    reliability=float(1/(1+effective)) if locked else 0.
    if equalizer=="lmmse": average*=reliability
    if not locked: average[:]=0
    return average,dict(analog_white_combined_variance=white,analog_split_half_variance=split,
        analog_correlated_error_floor=float(block_error_floor),
        analog_effective_combined_variance=effective,analog_reliability=reliability)

