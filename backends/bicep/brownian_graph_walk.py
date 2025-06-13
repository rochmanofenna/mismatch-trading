# ── brownian_graph_walk.py ─────────────────────────────────────────────
import networkx as nx
import numpy as np
from backends.bicep.brownian_motion import simulate_single_path
from backends.bicep.stochastic_control import apply_stochastic_controls

import json
import time
import logging
from functools import lru_cache
from dask import delayed, compute
import matplotlib.pyplot as plt
import psutil

# ---------------------------------------------------------------------
#  graph setup
# ---------------------------------------------------------------------
G = nx.DiGraph()
bit_states = ["00", "01", "10", "11"]
G.add_nodes_from(bit_states)

transitions = {
    ("00", "01"): 1.0,
    ("00", "10"): 1.0,
    ("01", "11"): 0.5,
    ("10", "11"): 0.5,
    ("11", "00"): 0.2,
}
for (src, dst), w in transitions.items():
    G.add_edge(src, dst, weight=w)

# ---------------------------------------------------------------------
#  optional per-user tweaks
# ---------------------------------------------------------------------
def load_user_config(path: str = "user_config.json") -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.warning("config %s not found/parse-err (%s); using defaults", path, e)
        return {}


user_config: dict = load_user_config()

# ---------------------------------------------------------------------
#  helper utilities
# ---------------------------------------------------------------------
@lru_cache(maxsize=128)
def _weight(G, cur, nbr, inc):
    return G[cur][nbr]["weight"] * (1 + inc)


def _next_state(cur, inc):
    nbrs = list(G.neighbors(cur))
    if not nbrs:
        return cur
    w = np.array([_weight(G, cur, n, inc) for n in nbrs], dtype=float)
    w /= w.sum()
    return np.random.choice(nbrs, p=w)


def _update_weights(hist):
    visits = {n: hist.count(n) for n in G.nodes()}
    for src in G:
        tot = sum(visits[n] for n in G.neighbors(src)) or 1
        for dst in G.neighbors(src):
            G[src][dst]["weight"] = visits[dst] / tot


# ---------------------------------------------------------------------
#  public API
# ---------------------------------------------------------------------
def simulate_graph_walk(
    initial_state: str = "00",
    T: float = 1.0,
    n_steps: int = 20,
    directional_bias: float = 0.0,
    variance_adjustment=None,
):
    path = [initial_state]
    bm = simulate_single_path(
        T,
        n_steps,
        0.0,
        T / n_steps,
        directional_bias,
        variance_adjustment,
        np,
        apply_stochastic_controls,
    )

    cur = initial_state
    for i in range(n_steps):
        inc = bm[i + 1] - bm[i]
        cur = _next_state(cur, inc)
        path.append(cur)

    _update_weights(path)
    return path


def run_parallel_walks(
    initial_state="00", T=1.0, n_steps=20, num_paths=10, parallel_threshold=50
):
    if num_paths > parallel_threshold:
        futs = [delayed(simulate_graph_walk)(initial_state, T, n_steps) for _ in range(num_paths)]
        return compute(*futs)
    return [simulate_graph_walk(initial_state, T, n_steps) for _ in range(num_paths)]


def visualize_graph(state_history):
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)
    sizes = [state_history.count(n) * 100 for n in G]
    nx.draw(G, pos, node_size=sizes, with_labels=True, font_size=12)
    plt.show()


# ---------------------------------------------------------------------
#  demo when run directly
# ---------------------------------------------------------------------
if __name__ == "__main__":
    pth = simulate_graph_walk()
    print("Final walk:", pth)
    visualize_graph(pth)
