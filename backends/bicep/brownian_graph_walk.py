import numpy as _np
try:
    import cupy as _cp
    _ = _cp.cuda.runtime.getDeviceCount()          # probe driver
    _xp = _cp
except Exception:                                  # no CuPy or bad driver
    _xp = _np
# ---------------------------------------------------------------------

import networkx as nx
from .stochastic_control import apply_stochastic_controls


def brownian_graph_walk(n_nodes, n_steps):
    """
    Minimal implementation used only by tests:
    • Generates n_nodes × n_steps 2-D Brownian increments
    • Applies the same stochastic controls
    • Returns (graph, paths) like the original API
    """
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))

    inc = _xp.random.normal(0, 1, (n_nodes, n_steps, 2))
    inc = apply_stochastic_controls(inc)
    paths = _xp.cumsum(inc, axis=1)

    for i in range(n_nodes - 1):
        w = float(_xp.linalg.norm(paths[i, -1] - paths[i + 1, -1]))
        g.add_edge(i, i + 1, weight=w)

    return g, paths
