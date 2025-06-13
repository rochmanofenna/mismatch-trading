import numpy as _np
try:
    import cupy as _cp
    _ = _cp.cuda.runtime.getDeviceCount()        # probe driver
    _xp = _cp
except Exception:                                # no CuPy or bad driver
    _xp = _np

import numpy as _np
try:
    import cupy as _cp
    _ = _cp.cuda.runtime.getDeviceCount()
    _xp = _cp
except Exception:
    _xp = _np


def apply_stochastic_controls(x):
    if _xp.isscalar(x):
        return x * 0.5 if abs(x) > 4 else x * 2.0 if abs(x) < 0.1 else x
    arr = _xp.asarray(x)
    arr[_xp.abs(arr) > 4] *= 0.5
    arr[_xp.abs(arr) < 0.1] *= 2.0
    return arr
