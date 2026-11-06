"""Fill vertices that no color camera observed, from colored neighbours on the mesh (numpy + scipy).

Small uncolored patches (seen by depth but outside the color keyframes) would
otherwise render as flat gray. Colors diffuse along mesh edges for a few rings;
anything still unreached keeps its placeholder.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def fill_uncolored(vertices: np.ndarray, triangles: np.ndarray, colors: np.ndarray, valid: np.ndarray, rings: int = 3) -> np.ndarray:
    n = len(vertices)
    if n == 0 or valid.all():
        return colors
    t = np.asarray(triangles, dtype=np.int64)
    rows = np.concatenate([t[:, 0], t[:, 1], t[:, 2], t[:, 1], t[:, 2], t[:, 0]])
    cols = np.concatenate([t[:, 1], t[:, 2], t[:, 0], t[:, 0], t[:, 1], t[:, 2]])
    adjacency = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))
    adjacency.data[:] = 1.0
    filled = np.asarray(colors, np.float32).copy()
    known = np.asarray(valid, bool).copy()
    for _ in range(rings):
        weights = adjacency @ known.astype(np.float32)
        frontier = (~known) & (weights > 0)
        if not frontier.any():
            break
        sums = adjacency @ (filled * known[:, None])
        filled[frontier] = sums[frontier] / weights[frontier, None]
        known |= frontier
    return filled
