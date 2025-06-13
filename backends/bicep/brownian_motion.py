import numpy as _np
try:
    import cupy as _cp
    _ = _cp.cuda.runtime.getDeviceCount()          # probe driver
    _xp = _cp
except Exception:                                  # no CuPy or bad driver
    _xp = _np

import psutil
from .stochastic_control import apply_stochastic_controls


# -----------------------------------------------------------------
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


# -----------------------------------------------------------------
def calculate_optimal_parameters(n_paths, n_steps, mem, cpu, gpu, gmem):
    est = n_steps * 4 / 2**20               # MB per path
    batch = min(max(1, int(mem*1024//est)), n_paths, cpu*100)
    gpu_thr = 1000 if gpu and gmem >= 8 else n_paths + 1
    return batch, 2000, gpu_thr


# -----------------------------------------------------------------
def simulate_single_path(T, n_steps, x0, dt, dir_bias=0.0, var_adj=None,
                         xp=_xp, apply_controls=apply_stochastic_controls):
    inc = xp.random.normal(dir_bias, 1.0, n_steps) * xp.sqrt(dt)
    for i in range(n_steps):
        inc[i] = apply_controls(inc[i])
    start = xp.asarray([x0])  
    return xp.concatenate((start, xp.cumsum(inc) + x0))


# -----------------------------------------------------------------
def brownian_motion_paths(T, n_steps, initial_value, n_paths,
                          directional_bias=0.0, variance_adjustment=None):
    if T <= 0 or n_steps <= 0 or n_paths < 0:
        raise ValueError("invalid arguments")
    dt = T / n_steps
    time_grid = _xp.linspace(0, T, n_steps + 1)
    if n_paths == 0:
        return time_grid, _xp.empty((0, n_steps + 1))

    mem, cpu, gpu, gmem = detect_system_resources()
    batch, _, _ = calculate_optimal_parameters(n_paths, n_steps, mem, cpu, gpu, gmem)

    pieces = []
    for _ in range(0, n_paths, batch):
        bs = min(batch, n_paths - len(pieces)*batch)
        paths = [_xp.asarray(simulate_single_path(
                    T, n_steps, initial_value, dt,
                    directional_bias, variance_adjustment))
                 for _ in range(bs)]
        pieces.append(_xp.stack(paths))
    return time_grid, _xp.concatenate(pieces, axis=0)
