import networkx as nx
import numpy as np
try:
    import cupy as cp
    _xp = cp
    cp.cuda.runtime.getDeviceCount()  # probe driver
except Exception:
    _xp = np

from .stochastic_control import apply_stochastic_controls

def brownian_graph_walk(n_nodes, n_steps):
    """
    Minimal implementation used only by tests.  
    Generates an nx.Graph on which nodes are 0…n_nodes-1,  
    paths is an (n_nodes, n_steps) array of increments + cumsum.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    inc = _xp.random.normal(0, 1, size=(n_nodes, n_steps))
    inc = apply_stochastic_controls(inc, None, None)  # broadcast-safe
    paths = _xp.cumsum(inc, axis=1)

    # add a simple chain of edges with weights = final-distance
    for i in range(n_nodes-1):
        w = float(_xp.linalg.norm(paths[i,-1] - paths[i+1,-1]))
        G.add_edge(i, i+1, weight=w)

    return G, paths
