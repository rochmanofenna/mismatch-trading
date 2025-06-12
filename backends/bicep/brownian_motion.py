try:
    import cupy as _xp
except ModuleNotFoundError:
    import numpy as _xp

import psutil
from distributed import Client, LocalCluster
from .stochastic_control import apply_stochastic_controls


def detect_system_resources():
    available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
    cpu_count = psutil.cpu_count(logical=False)
    try:
        import cupy as cp
        gpu_available = True
        gpu_memory_gb = cp.cuda.Device(0).mem_info[1] / (1024 ** 3)
    except ModuleNotFoundError:
        gpu_available = False
        gpu_memory_gb = 0
    return available_memory_gb, cpu_count, gpu_available, gpu_memory_gb


def calculate_optimal_parameters(n_paths, n_steps, available_memory_gb, cpu_count, gpu_available, gpu_memory_gb):
    estimated_path_memory = n_steps * 4 / (1024 ** 2)
    max_batch_size = int(available_memory_gb * 1024 / estimated_path_memory)
    batch_size = min(max_batch_size, n_paths, cpu_count * 100)
    gpu_threshold = 1000 if gpu_available and gpu_memory_gb >= 8 else n_paths + 1
    save_interval = max(2000, batch_size * 2)
    return batch_size, save_interval, gpu_threshold


def setup_dask_cluster(n_paths, large_scale_threshold=5000):
    if n_paths < large_scale_threshold:
        return None
    cluster = LocalCluster(n_workers=2, threads_per_worker=2, memory_limit="2GB")
    return Client(cluster)


def simulate_single_path(T, n_steps, initial_value, dt, directional_bias, variance_adjustment, _xp=_xp, apply_controls=apply_stochastic_controls):
    state_visit_count = 0.5
    feedback_value = 0.5
    time = _xp.linspace(0, T, n_steps + 1)
    increments = _xp.random.normal(0, 1, n_steps)
    if variance_adjustment:
        variance_scale = variance_adjustment(time[1:])
        increments *= _xp.sqrt(variance_scale * dt)
    else:
        increments *= _xp.sqrt(dt)
    for i in range(len(increments)):
        increments[i] = apply_controls(increments[i])
    path = initial_value + _xp.cumsum(increments)
    return path


def brownian_motion_paths(n_paths, n_steps, initial_value=0.0, dt=0.01, directional_bias=0.0, variance_adjustment=None):
    mem_gb, cpu_cnt, gpu_ok, gpu_mem = detect_system_resources()
    batch, save_int, gpu_thr = calculate_optimal_parameters(n_paths, n_steps, mem_gb, cpu_cnt, gpu_ok, gpu_mem)
    client = setup_dask_cluster(n_paths)
    paths = []
    for start in range(0, n_paths, batch):
        end = min(start + batch, n_paths)
        batch_paths = [
            simulate_single_path(
                T=n_steps * dt,
                n_steps=n_steps,
                initial_value=initial_value,
                dt=dt,
                directional_bias=directional_bias,
                variance_adjustment=variance_adjustment,
            )
            for _ in range(start, end)
        ]
        paths.append(_xp.stack(batch_paths))
        if len(paths) * batch >= gpu_thr and not gpu_ok:
            break
    if client:
        client.close()
    return _xp.concatenate(paths, axis=0)
