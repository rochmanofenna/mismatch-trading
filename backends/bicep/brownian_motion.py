import os

# ---------------------------------------------------------
# pick array backend once, based on the environment switch
# ---------------------------------------------------------
USE_CUPY = os.getenv("DISABLE_CUPY") != "1"

if USE_CUPY:                 # GPU branch
    import cupy as _xp
else:                        # forced-CPU branch
    import numpy as _xp

import psutil
from .stochastic_control import apply_stochastic_controls


def detect_system_resources():
    mem = psutil.virtual_memory().available / 2**30
    cpu = psutil.cpu_count(logical=False)
    try:
        import cupy as cp
        gpu = True
        gmem = cp.cuda.Device(0).mem_info[1] / 2**30
    except ModuleNotFoundError:
        gpu, gmem = False, 0
    return mem, cpu, gpu, gmem


def calculate_optimal_parameters(n_paths, n_steps, mem, cpu, gpu, gmem):
    est = n_steps * 4 / 2**20               # MB per path
    batch = min(max(1, int(mem*1024//est)), n_paths, cpu*100)
    gpu_thr = 1000 if gpu and gmem >= 8 else n_paths + 1
    return batch, 2000, gpu_thr


def simulate_batch(T, n_steps, n_paths, xp, dir_bias=0.0, var_adj=None):
    """
    Simulate *n_paths* Brownian paths of length *n_steps* on the given
    array module (xp = numpy | cupy).  Returns (n_paths, n_steps+1).
    Heavy lifting happens in a single RNG + cumsum call.
    """
    dt = T / n_steps
    # one kernel launch – draw all increments
    inc = xp.random.normal(dir_bias, 1.0, size=(n_paths, n_steps)) * xp.sqrt(dt)

    if var_adj is not None:        # optional variance schedule
        scale = xp.asarray(var_adj(xp.linspace(dt, T, n_steps)))  # (n_steps,)
        inc *= scale[None, :]

    paths = xp.empty((n_paths, n_steps + 1), dtype=inc.dtype)
    paths[:, 0] = 0.0
    paths[:, 1:] = xp.cumsum(inc, axis=1)
    return paths


def brownian_motion_paths(
    T, n_steps, *,                      # make them keyword-only
    initial_value=0.0,
    n_paths=10,
    directional_bias=0.0,
    variance_adjustment=None,
    batch=2_000                         # how many paths per GPU copy-out
):
    try:
        import cupy as cp
        xp = cp
        gpu = True
    except ImportError:
        import numpy as np
        xp = np
        gpu = False

    if n_paths == 0:
        return xp.linspace(0, T, n_steps + 1), xp.empty((0, n_steps + 1))

    time_grid = xp.linspace(0, T, n_steps + 1)

    if gpu:                                       # stay on device
        full = simulate_batch(
            T, n_steps, n_paths, xp,
            dir_bias=directional_bias,
            var_adj=variance_adjustment
        )
        return time_grid, full + initial_value

    # --- CPU fallback in manageable chunks --------------------------------
    out = []
    for start in range(0, n_paths, batch):
        n = min(batch, n_paths - start)
        out.append(
            simulate_batch(T, n_steps, n, xp,
                           dir_bias=directional_bias,
                           var_adj=variance_adjustment)
            + initial_value
        )
    return time_grid, xp.concatenate(out, axis=0)
