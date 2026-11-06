"""UV unwrapping into a single texture atlas (Open3D's UVAtlas, parallel partitions).

xatlas was measured at ~500 s for 150k triangles; UVAtlas with parallel
partitions takes ~5 s, so it is used here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UvMesh:
    vertices: np.ndarray  # (N, 3) float32 (vertices on UV seams are duplicated)
    triangles: np.ndarray  # (M, 3) int64
    uvs: np.ndarray  # (N, 2) float32 in [0, 1]; glTF convention: (0, 0) is the image's top-left
    source_vertex: np.ndarray  # (N,) index of the original vertex each new vertex came from


def orient_consistently(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Consistent winding across an edge-manifold mesh: returns ``(triangles, conflict_mask)``.

    Breadth-first over triangle neighbours, each triangle is flipped (or not) to
    agree with the one it was reached from; a triangle that is reached again with
    the opposite requirement closes a non-orientable loop and is flagged for
    removal. Each connected piece keeps the winding most of its triangles had,
    so marching-cubes "outward" normals survive (the texture bake uses them).
    """
    from collections import deque

    tris = np.asarray(triangles, np.int64)
    m = len(tris)
    starts = np.concatenate([tris[:, 0], tris[:, 1], tris[:, 2]])
    ends = np.concatenate([tris[:, 1], tris[:, 2], tris[:, 0]])
    owner = np.tile(np.arange(m), 3)
    n_vertices = int(tris.max()) + 1 if m else 0
    keys = np.minimum(starts, ends) * n_vertices + np.maximum(starts, ends)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    pair = np.flatnonzero(sorted_keys[1:] == sorted_keys[:-1])  # i, i+1 share an edge (manifold: at most pairs)
    first, second = order[pair], order[pair + 1]
    same_direction = starts[first] == starts[second]  # same direction => their windings disagree
    a, b = owner[first], owner[second]
    # CSR adjacency: neighbour + whether crossing the edge flips the parity.
    src = np.concatenate([a, b])
    dst = np.concatenate([b, a])
    flip_across = np.concatenate([same_direction, same_direction]).astype(np.int8)
    sort = np.argsort(src, kind="stable")
    src, dst, flip_across = src[sort], dst[sort], flip_across[sort]
    offsets = np.searchsorted(src, np.arange(m + 1))
    dst_list, flip_list, offsets_list = dst.tolist(), flip_across.tolist(), offsets.tolist()

    parity = [-1] * m
    conflict = np.zeros(m, bool)
    component = np.full(m, -1, np.int64)
    for seed in range(m):
        if parity[seed] != -1:
            continue
        parity[seed] = 0
        component[seed] = seed
        queue = deque([seed])
        while queue:
            t = queue.popleft()
            if conflict[t]:
                continue
            for k in range(offsets_list[t], offsets_list[t + 1]):
                n = dst_list[k]
                want = parity[t] ^ flip_list[k]
                if parity[n] == -1:
                    parity[n] = want
                    component[n] = seed
                    queue.append(n)
                elif parity[n] != want and not conflict[n]:
                    conflict[n] = True
    parity_arr = np.asarray(parity, np.int8)
    # Keep each piece's majority winding.
    flips_per_component = np.bincount(component, weights=parity_arr, minlength=m)
    size_per_component = np.bincount(component, minlength=m)
    invert = flips_per_component[component] * 2 > size_per_component[component]
    flip = (parity_arr == 1) ^ invert
    oriented = tris.copy()
    oriented[flip] = oriented[flip][:, [0, 2, 1]]
    return oriented, conflict


def same_vertex_set_mask(triangles: np.ndarray) -> np.ndarray:
    """Triangles whose three vertices are also used, in any order or winding, by another triangle.

    Open3D's ``remove_duplicated_triangles`` only treats rotations (0-1-2 vs 1-2-0) as duplicates. A
    mirrored pair (0-1-2 and 1-0-2) is a closed two-faced "pillow" that passes Open3D's manifold and
    orientability checks but makes UVAtlas fail with "Non-manifold mesh" (seen on a real capture).
    Both triangles go: a zero-volume pillow carries no surface.
    """
    tris = np.asarray(triangles, np.int64)
    if len(tris) == 0:
        return np.zeros(0, bool)
    _, inverse, counts = np.unique(np.sort(tris, axis=1), axis=0, return_inverse=True, return_counts=True)
    return counts[inverse.reshape(-1)] > 1


def make_manifold(mesh, *, max_passes: int = 8):
    """Clean a legacy Open3D mesh in place so UVAtlas accepts it; returns ``(mesh, stats)``.

    Real scans (marching cubes + quadric decimation) contain a few duplicate or
    degenerate triangles, edges shared by 3+ triangles, "bowtie" vertices where
    two fans touch at one point, and (after decimation) triangles whose winding
    disagrees with their neighbours; UVAtlas rejects all of them ("Non-manifold
    mesh") or crashes. Non-manifold edges are cut by Open3D, triangles around
    bowties and winding conflicts are dropped, and winding is made consistent,
    repeating until clean (dropping triangles can create new bowties). Vertex
    colors stay attached.
    """
    import open3d as o3d

    triangles_before = len(mesh.triangles)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    passes = 0
    clean = False
    for passes in range(1, max_passes + 1):
        pillows = same_vertex_set_mask(np.asarray(mesh.triangles))
        if pillows.any():
            mesh.remove_triangles_by_mask(pillows)
            mesh.remove_unreferenced_vertices()
        mesh.remove_non_manifold_edges()
        bowties = np.asarray(mesh.get_non_manifold_vertices(), np.int64)
        if len(bowties):
            mesh.remove_triangles_by_mask(np.isin(np.asarray(mesh.triangles), bowties).any(axis=1))
            mesh.remove_unreferenced_vertices()
            continue
        if len(mesh.get_non_manifold_edges()):
            continue
        oriented, conflict = orient_consistently(np.asarray(mesh.triangles))
        mesh.triangles = o3d.utility.Vector3iVector(oriented.astype(np.int32))
        if conflict.any():
            mesh.remove_triangles_by_mask(conflict)
            mesh.remove_unreferenced_vertices()
            continue
        clean = True
        break
    mesh.remove_unreferenced_vertices()
    removed = triangles_before - len(mesh.triangles)
    return mesh, {"triangles_removed": int(removed), "passes": passes, "manifold": bool(clean and mesh.is_orientable())}


def unwrap(vertices: np.ndarray, triangles: np.ndarray, *, size: int = 4096, gutter: float = 2.0, partitions: int = 16) -> UvMesh:
    import open3d as o3d

    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex.positions = o3d.core.Tensor(np.ascontiguousarray(vertices, np.float32))
    mesh.triangle.indices = o3d.core.Tensor(np.ascontiguousarray(triangles, np.int64))
    mesh.compute_uvatlas(size=int(size), gutter=float(gutter), parallel_partitions=int(partitions), nthreads=0)
    corner_uvs = mesh.triangle.texture_uvs.numpy().reshape(-1, 2).astype(np.float32)  # (3M, 2)
    corner_vertex = np.asarray(triangles, np.int64).reshape(-1)  # (3M,)

    # Merge corners that share a vertex and a UV (inside a chart); keep seams split.
    quantized = np.round(corner_uvs * (size * 4)).astype(np.int64)
    keys = np.stack([corner_vertex, quantized[:, 0], quantized[:, 1]], axis=1)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    source = corner_vertex[first]
    return UvMesh(
        vertices=np.asarray(vertices, np.float32)[source],
        triangles=inverse.reshape(-1, 3).astype(np.int64),
        uvs=np.clip(corner_uvs[first], 0.0, 1.0),
        source_vertex=source,
    )
