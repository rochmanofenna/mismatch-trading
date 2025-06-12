cat > backends/bicep/brownian_motion.py <<'PY'
try:
    import cupy as _xp
except ModuleNotFoundError:
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
        gpu = False
        gmem = 0
    return mem, cpu, gpu, gmem


def simulate_single_path(T, n_steps, x0, dt, dir_bias, var_adj, xp=_xp,
                         apply_controls=apply_stochastic_controls):
    inc = xp.random.normal(dir_bias, 1.0, n_steps) * xp.sqrt(dt)
    for i in range(n_steps):
        inc[i] = apply_controls(inc[i])
    return xp.concatenate(([x0], xp.cumsum(inc) + x0))


def brownian_motion_paths(T, n_steps, initial_value, n_paths,
                          directional_bias=0.0, variance_adjustment=None):
    if T <= 0:
        raise ValueError("T must be > 0")
    if n_steps <= 0:
        raise ValueError("n_steps must be > 0")
    if n_paths < 0:
        raise ValueError("n_paths must be >= 0")
    dt = T / n_steps
    time_grid = _xp.linspace(0, T, n_steps + 1)
    if n_paths == 0:
        return time_grid, _xp.empty((0, n_steps + 1))

    mem, cpu, gpu, gmem = detect_system_resources()
    est = n_steps * 4 / 2**20
    batch = min(max(1, int(mem * 1024 // est)), n_paths, cpu * 100)

    batches = []
    for start in range(0, n_paths, batch):
        bs = min(batch, n_paths - start)
        batch_paths = [_xp.asarray(
            simulate_single_path(T, n_steps, initial_value, dt,
                                 directional_bias, variance_adjustment))
                       for _ in range(bs)]
        batches.append(_xp.stack(batch_paths))
    all_paths = _xp.concatenate(batches, axis=0)
    return time_grid, all_paths
