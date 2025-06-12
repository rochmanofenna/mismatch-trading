try:
    import cupy as _xp
except ModuleNotFoundError:
    import numpy as _xp

import networkx as nx
from .stochastic_control import apply_stochastic_controls


def brownian_graph_walk(n_nodes, n_steps, variance_scale=1.0):
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    pos = _xp.zeros((n_nodes, 2))
    increments = _xp.random.normal(0, variance_scale, (n_nodes, n_steps, 2))
    increments = apply_stochastic_controls(increments)
    paths = _xp.cumsum(increments, axis=1)
    for i in range(n_nodes - 1):
        g.add_edge(i, i + 1, weight=float(_xp.linalg.norm(paths[i, -1] - paths[i + 1, -1])))
    return g, paths
