try:
    import cupy as _xp
except ModuleNotFoundError:
    import numpy as _xp


def apply_stochastic_controls(increments):
    increments[_xp.abs(increments) > 4.0] *= 0.5
    increments[_xp.abs(increments) < 0.1] *= 2.0
    return increments
