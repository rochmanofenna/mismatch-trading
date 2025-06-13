# ── brownian_motion.py ────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from dask import delayed, compute
from dask.distributed import Client, LocalCluster
import psutil, logging, os, time

from backends.bicep.stochastic_control import apply_stochastic_controls

# ---------------------------------------------------------------------
#  resource helpers
# ---------------------------------------------------------------------
def detect_system_resources():
    mem = psutil.virtual_memory().available / 2**30
    cpu = psutil.cpu_count(logical=False)
    try:
        import cupy as cp
        gpu, gmem = True, cp.cuda.Device(0).mem_info[1] / 2**30
    except ModuleNotFoundError:
        gpu, gmem = False, 0.0
    return mem, cpu, gpu, gmem


def calculate_optimal_parameters(n, m, mem, cpu, gpu, gmem):
    est = m * 4 / 2**20
    bs = min(int(mem * 1024 // est), n, cpu * 100)
    gthr = 1000 if gpu and gmem >= 8 else n + 1
    return bs or 1, 2000, gthr


def setup_dask_cluster(n, thresh=5000):
    if n < thresh:
        return None
    c = Client(LocalCluster(n_workers=2, threads_per_worker=2, memory_limit="2GB"))
    return c


# ---------------------------------------------------------------------
#  Brownian helpers
# ---------------------------------------------------------------------
def simulate_single_path(
    T,
    n_steps,
    x0,
    dt,
    directional_bias=0.0,
    variance_adjustment=None,
    xp=np,
    apply_controls=apply_stochastic_controls,
):
    inc = xp.random.normal(directional_bias, 1.0, n_steps) * xp.sqrt(dt)
    if variance_adjustment is not None:
        inc *= xp.sqrt(variance_adjustment(xp.linspace(0, T, n_steps)))
    for i in range(n_steps):
        inc[i] = apply_controls(inc[i])
    return xp.concatenate(([x0], x0 + xp.cumsum(inc)))


# ---------------------------------------------------------------------
#  public API
# ---------------------------------------------------------------------
def brownian_motion_paths(
    T=1,
    n_steps=100,
    initial_value=0.0,
    n_paths=10,
    directional_bias=0.0,
    variance_adjustment=None,
):
    if T <= 0 or n_steps <= 0 or n_paths < 0:
        raise ValueError("Invalid arguments")

    mem, cpu, gpu, gmem = detect_system_resources()
    bs, save_int, gthr = calculate_optimal_parameters(n_paths, n_steps, mem, cpu, gpu, gmem)
    client = setup_dask_cluster(n_paths)

    xp = np
    if gpu and n_paths >= gthr:
        try:
            import cupy as cp

            xp = cp
        except ModuleNotFoundError:
            pass

    paths = []
    dt = T / n_steps
    for start in range(0, n_paths, bs):
        end = min(start + bs, n_paths)
        batch = [
            delayed(simulate_single_path)(
                T,
                n_steps,
                initial_value,
                dt,
                directional_bias,
                variance_adjustment,
                xp,
                apply_stochastic_controls,
            )
            for _ in range(start, end)
        ]
        paths.extend(compute(*batch))
    if client:
        client.close()
    return xp.linspace(0, T, n_steps + 1), xp.asarray(paths)


# ---------------------------------------------------------------------
#  demo when run directly
# ---------------------------------------------------------------------
if __name__ == "__main__":
    time_grid, paths = brownian_motion_paths(T=1, n_steps=100, n_paths=10)
    for p in paths[:10]:
        plt.plot(time_grid, p)
    plt.title("Brownian Motion Paths")
    plt.xlabel("t")
    plt.ylabel("W(t)")
    plt.show()
