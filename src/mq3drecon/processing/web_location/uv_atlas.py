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
