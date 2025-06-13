import os
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

# pick array backend
USE_CUPY = cp is not None and os.getenv("DISABLE_CUPY") != "1"
_xp = cp if USE_CUPY else np

import psutil
from .stochastic_control import apply_stochastic_controls

def detect_system_resources():
    """Returns (mem_gb, cpu_cores, gpu_available, gpu_mem_gb)."""
    mem = psutil.virtual_memory().available / 2**30
    cpu = psutil.cpu_count(logical=False)
    try:
        import cupy as _cp
        gpu = True
        gmem = _cp.cuda.Device(0).mem_info[1] / 2**30
    except Exception:
        gpu, gmem = False, 0
    return mem, cpu, gpu, gmem

def calculate_optimal_parameters(n_paths, n_steps, mem, cpu, gpu, gmem):
    """
    Returns (batch_size, save_interval, gpu_threshold).
    """
    # estimate MB per path
    est_mb = n_steps * 4 / 2**20
    # on CPU we want no more than cpu*100 paths in one batch
    batch = min(max(1, int(mem*1024//est_mb)), n_paths, cpu*100)
    gpu_thr = 1000 if gpu and gmem >= 8 else n_paths + 1
    return batch, 2000, gpu_thr

def simulate_single_path(T, n_steps, initial_value,
                         dt, directional_bias, variance_adjustment, xp, apply_ctrl):
    """
    Generate one path (n_steps+1) by cumsum + stochastic controls.
    """
    # draw increments
    inc = xp.random.normal(directional_bias, 1.0, size=n_steps) * xp.sqrt(dt)
    # apply your custom controls
    for i in range(n_steps):
        inc[i] = apply_ctrl(inc[i], None, None, None, None)
    # build path
    path = xp.empty(n_steps+1, dtype=inc.dtype)
    path[0] = initial_value
    path[1:] = xp.cumsum(inc) + initial_value
    return path

def brownian_motion_paths(
    T, n_steps,
    *,
    initial_value=0.0,
    n_paths=10,
    directional_bias=0.0,
    variance_adjustment=None,
    batch=None
):
    """
    Returns (time_grid, paths) where paths is (n_paths, n_steps+1).
    Raises ValueError if T<=0 or n_steps<=0.
    """
    if T <= 0 or n_steps <= 0:
        raise ValueError("T and n_steps must be positive")

    dt = T / n_steps
    # build time grid on whichever backend
    time_grid = _xp.linspace(0, T, n_steps+1)

    # single-path implementation used in the unit-tests
    def _single(i):
        return simulate_single_path(
            T, n_steps, initial_value, dt,
            directional_bias, variance_adjustment or (lambda t:1.0),
            _xp, apply_stochastic_controls
        )

    # GPU-only: do all paths in one big stack if small enough
    if USE_CUPY and batch is None:
        stacked = _xp.stack([_single(i) for i in range(n_paths)], axis=0)
        return time_grid, stacked

    # CPU fallback or batched GPU
    mem, cpu, gpu_avail, gmem = detect_system_resources()
    batch_size, _, gpu_thr = calculate_optimal_parameters(
        n_paths, n_steps, mem, cpu, gpu_avail, gmem
    )
    # if they passed an explicit batch override, use it
    bs = batch or batch_size

    pieces = []
    for start in range(0, n_paths, bs):
        end = min(start + bs, n_paths)
        pieces.append(_xp.stack([_single(i) for i in range(start, end)], axis=0))
    return time_grid, _xp.concatenate(pieces, axis=0)
