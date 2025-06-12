try:
    import cupy as _xp
except ModuleNotFoundError:
    import numpy as _xp


def _adjust(val):
    if _xp.abs(val) > 4.0:
        return val * 0.5
    if _xp.abs(val) < 0.1:
        return val * 2.0
    return val


def apply_stochastic_controls(x):
    if _xp.isscalar(x):
        return _adjust(x)
    vec = _xp.asarray(x)
    hi = _xp.abs(vec) > 4
    lo = _xp.abs(vec) < 0.1
    vec[hi] *= 0.5
    vec[lo] *= 2.0
    return vec
