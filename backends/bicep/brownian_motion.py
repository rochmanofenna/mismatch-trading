import os
import time
import numpy as _np
import psutil
import logging

from dask import delayed, compute
from dask.distributed import Client, LocalCluster
from .stochastic_control import apply_stochastic_controls

# ——— Logging setup ———
LOG_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir, os.pardir,
    "results", "logs", "simulation.log"
)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def detect_system_resources():
    mem = psutil.virtual_memory().available / (1024**3)
    cpu = psutil.cpu_count(logical=False)
    try:
        import cupy as cp
        gpu = True
        gmem = cp.cuda.Device(0).mem_info[1] / (1024**3)
    except Exception:
        gpu, gmem = False, 0
    logging.info(f"Resources: mem={mem:.2f}GB cpu={cpu} gpu={gpu} gmem={gmem:.2f}GB")
    return mem, cpu, gpu, gmem

def calculate_optimal_parameters(n_paths, n_steps, mem, cpu, gpu, gmem):
    est_mb = n_steps * 4 / 2**20
    batch = min(max(1, int(mem*1024//est_mb)), n_paths, cpu*100)
    gpu_thr = 1000 if gpu and gmem>=8 else n_paths+1
    save_int = max(2000, batch*2)
    logging.info(f"Params: batch={batch} gpu_thr={gpu_thr} save_int={save_int}")
    return batch, save_int, gpu_thr

def setup_dask_cluster():
    cluster = LocalCluster(n_workers=2, threads_per_worker=2, memory_limit="2GB")
    client = Client(cluster)
    logging.info("Dask cluster started")
    return client

def simulate_single_path(T, n_steps, initial_value, dt,
                         directional_bias, variance_adjustment,
                         xp, apply_ctrl):
    state_count = 1
    feedback = 0.5
    t0 = 0.1
    total = n_steps

    times = xp.linspace(0, T, n_steps+1)
    inc = xp.random.normal(directional_bias, 1.0, size=n_steps)
    if variance_adjustment:
        var_scale = variance_adjustment(times[1:])
        inc *= xp.sqrt(var_scale * dt)
    else:
        inc *= xp.sqrt(dt)

    for i in range(n_steps):
        inc[i] = apply_ctrl(inc[i], state_count, feedback, t0, total)

    path = xp.empty(n_steps+1, dtype=inc.dtype)
    path[0]= initial_value
    path[1:]= xp.cumsum(inc) + initial_value
    return path

def brownian_motion_paths(
    T=1.0,
    n_steps=100,
    initial_value=0.0,
    n_paths=10,
    directional_bias=0.0,
    variance_adjustment=None
):
    # -- validate --
    if T<=0 or n_steps<=0:
        raise ValueError("T and n_steps must be > 0")
    if n_paths<0:
        raise ValueError("n_paths must be >= 0")
    if n_paths==0:
        return _np.linspace(0, T, n_steps+1), _np.empty((0, n_steps+1))

    mem, cpu, gpu, gmem = detect_system_resources()
    batch, save_int, gpu_thr = calculate_optimal_parameters(n_paths, n_steps, mem, cpu, gpu, gmem)
    client = setup_dask_cluster()

    try:
        if n_paths >= gpu_thr:
            import cupy as cp
            xp = cp; use_gpu = True
        else:
            xp = _np; use_gpu = False
    except ImportError:
        xp = _np; use_gpu = False

    dt = T / n_steps
    tg = xp.linspace(0, T, n_steps+1)
    if use_gpu:
        tg = tg.get()

    # memmap to avoid OOM
    fname = f"brownian_paths_{int(time.time()*1e3)}.dat"
    paths = _np.memmap(fname, dtype="float32", mode="w+",
                       shape=(n_paths, n_steps+1))
    logging.info(f"Writing memmap → {fname}")

    for i in range(0, n_paths, batch):
        end = min(i+batch, n_paths)
        tasks = [
            delayed(simulate_single_path)(
                T, n_steps, initial_value, dt,
                directional_bias, variance_adjustment,
                xp, apply_stochastic_controls
            )
            for _ in range(i, end)
        ]
        batch_res = compute(*tasks)
        for j, p in enumerate(batch_res):
            paths[i+j] = p.get() if use_gpu else p

        if end % save_int == 0 or end==n_paths:
            paths.flush()
            logging.info(f"Flushed up to {end}/{n_paths}")

    return tg, paths
