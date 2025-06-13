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


def simulate_batch(T, n_steps, n_paths, xp):
    dt = T / n_steps
    inc = xp.random.normal(0, 1, (n_paths, n_steps)).astype(xp.float32)
    inc *= xp.sqrt(dt)
    # one cumsum per path – all on GPU
    paths = xp.concatenate(
        [xp.zeros((n_paths, 1), dtype=inc.dtype),
         xp.cumsum(inc, axis=1)], axis=1)
    return paths
