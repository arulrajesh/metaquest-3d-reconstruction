"""Coarse surface mesh for the headset overlay (needs Open3D).

The live mesh is simplified by vertex clustering (fast, topology-agnostic) to a
size that streams comfortably over Wi-Fi. Each vertex carries its color and a
confidence in [0, 1] derived from the Kalman posterior sigma, so the headset can
show which surfaces are solid (green) and which still need looking at.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from mq3drecon.processing.live.gpu_tsdf import MeshArrays


def confidence_from_sigma(sigma_m: np.ndarray, voxel_size_m: float) -> np.ndarray:
    """1 at a quarter voxel or better, 0 at one voxel or worse."""
    good, bad = 0.25 * voxel_size_m, 1.0 * voxel_size_m
    return np.clip((bad - np.asarray(sigma_m, np.float64)) / (bad - good), 0.0, 1.0)


def _cluster(vertices: np.ndarray, triangles: np.ndarray, colors: np.ndarray, voxel_m: float) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh.simplify_vertex_clustering(voxel_size=voxel_m, contraction=o3d.geometry.SimplificationContraction.Average)


def coarse_mesh(
    mesh: MeshArrays, *, voxel_size_m: float, cluster_m: float = 0.08, max_triangles: int = 40_000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (vertices, triangles, rgb, confidence) of a simplified mesh in the same world frame."""
    if not len(mesh.triangles):
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 3)), np.zeros(0)
    vertices = mesh.vertices.astype(np.float64)
    triangles = mesh.triangles.astype(np.int32)
    sigma = mesh.sigma_m if mesh.sigma_m is not None else np.zeros(len(vertices))
    confidence = confidence_from_sigma(sigma, voxel_size_m)
    size = cluster_m
    for _ in range(6):
        colored = _cluster(vertices, triangles, mesh.colors.astype(np.float64), size)
        if len(colored.triangles) <= max_triangles:
            break
        size *= float(np.sqrt(len(colored.triangles) / max_triangles)) * 1.05
    graded = _cluster(vertices, triangles, np.repeat(confidence[:, None], 3, axis=1), size)
    return (
        np.asarray(colored.vertices),
        np.asarray(colored.triangles, dtype=np.int64),
        np.asarray(colored.vertex_colors),
        np.asarray(graded.vertex_colors)[:, 0],
    )
