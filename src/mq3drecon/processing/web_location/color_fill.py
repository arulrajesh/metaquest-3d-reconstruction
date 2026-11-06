"""Fill vertices that no color camera observed, from colored neighbours on the mesh (numpy + scipy).

Small uncolored patches (seen by depth but outside the color keyframes) would
otherwise render as flat gray. Colors diffuse along mesh edges for a few rings.
With ``harmonic=True`` the remaining unseen vertices are then solved as a
harmonic (Laplacian) interpolation of the colors around them: a smooth blend
across large unseen areas such as an unscanned ceiling, instead of flat gray.
Pieces with no observed vertex at all keep their placeholder.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def fill_uncolored(
    vertices: np.ndarray, triangles: np.ndarray, colors: np.ndarray, valid: np.ndarray, rings: int = 3, *, harmonic: bool = False,
) -> np.ndarray:
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
    if harmonic and (~known).any() and known.any():
        filled = _harmonic_fill(adjacency, filled, known)
    return filled


def _harmonic_fill(adjacency: sparse.csr_matrix, colors: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Solve L_uu x_u = -L_uk x_k (graph Laplacian, unknown vertices u, known k) for each color channel.

    Unknown vertices in a piece that touches no known vertex make the system singular there; a tiny
    diagonal term keeps it solvable and leaves those at (near) zero, so they are restored afterwards.
    """
    from scipy.sparse.linalg import spsolve

    unknown = ~known
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = (sparse.diags(degree) - adjacency).tocsr()
    l_uu = laplacian[unknown][:, unknown] + sparse.identity(int(unknown.sum()), format="csr") * 1e-6
    l_uk = laplacian[unknown][:, known]
    rhs = -(l_uk @ colors[known])
    solved = spsolve(l_uu.tocsc(), rhs)
    filled = colors.copy()
    # Keep the placeholder where no known color exists in the piece (the solution is ~0 there).
    target = np.flatnonzero(unknown)
    good = _reaches_known(adjacency, known)[target]
    filled[target[good]] = np.clip(np.asarray(solved).reshape(-1, colors.shape[1])[good], 0.0, 1.0)
    return filled


def _reaches_known(adjacency: sparse.csr_matrix, known: np.ndarray) -> np.ndarray:
    """Vertices connected (through any path) to at least one known vertex."""
    from scipy.sparse.csgraph import connected_components

    _, labels = connected_components(adjacency, directed=False)
    has_known = np.zeros(labels.max() + 1, bool)
    has_known[labels[known]] = True
    return has_known[labels]
