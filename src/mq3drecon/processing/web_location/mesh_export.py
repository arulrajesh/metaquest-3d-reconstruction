"""Mesh cleanup, decimation, floor/spawn estimation, and GLB export for web locations (needs Open3D).

Coordinates stay in MQ3DRecon's Open3D world frame, which is also glTF's
convention (Y-up, right-handed), in meters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from mq3drecon.processing.live.gpu_tsdf import MeshArrays


@dataclass(frozen=True)
class Spawn:
    position: tuple[float, float, float]  # on the floor
    yaw_deg: float  # rotation about +Y applied to the -Z forward direction (three.js convention)


def to_open3d(mesh: MeshArrays) -> o3d.geometry.TriangleMesh:
    result = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices.astype(np.float64)),
        o3d.utility.Vector3iVector(mesh.triangles.astype(np.int32)),
    )
    result.vertex_colors = o3d.utility.Vector3dVector(mesh.colors.astype(np.float64))
    return result


def remove_debris(mesh: o3d.geometry.TriangleMesh, min_fraction: float = 0.01) -> o3d.geometry.TriangleMesh:
    """Drop connected pieces smaller than ``min_fraction`` of all triangles (floating noise)."""
    clusters, counts, _ = mesh.cluster_connected_triangles()
    clusters, counts = np.asarray(clusters), np.asarray(counts)
    if len(counts) <= 1:
        return mesh
    small = counts[clusters] < min_fraction * len(clusters)
    cleaned = o3d.geometry.TriangleMesh(mesh)
    cleaned.remove_triangles_by_mask(small)
    cleaned.remove_unreferenced_vertices()
    return cleaned


def decimate(mesh: o3d.geometry.TriangleMesh, target_triangles: int) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) <= target_triangles:
        return mesh
    simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=int(target_triangles))
    simplified.remove_degenerate_triangles()
    simplified.remove_unreferenced_vertices()
    return simplified


FLOOR_PRIOR_TOLERANCE_M = 0.25
STANDING_EYE_HEIGHT_RANGE_M = (1.2, 2.1)


def _horizontal_heights(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
    return np.asarray(mesh.vertices)[np.abs(normals[:, 1]) > 0.9, 1]


def lowest_substantial_surface(heights: np.ndarray, bin_m: float = 0.02) -> float | None:
    if len(heights) < 50:
        return None
    bins = np.floor(heights / bin_m).astype(np.int64)
    values, counts = np.unique(bins, return_counts=True)
    floor_bin = values[counts >= 0.05 * counts.max()].min()
    return float(np.median(heights[bins == floor_bin]))


def estimate_floor_y(mesh: o3d.geometry.TriangleMesh, head_heights: np.ndarray | None = None) -> tuple[float, str]:
    """Floor height and where it came from.

    Quest tracking uses a floor-level origin, so with a standing operator (median eye height 1.2–2.1 m)
    the floor is y = 0. A captured horizontal surface within 0.25 m of that refines it; anything
    higher is furniture (a bed or sofa top), not floor. Without a usable prior, the lowest
    substantial horizontal surface is used.
    """
    heights = _horizontal_heights(mesh)
    if head_heights is not None and len(head_heights):
        eye = float(np.median(head_heights))
        if STANDING_EYE_HEIGHT_RANGE_M[0] <= eye <= STANDING_EYE_HEIGHT_RANGE_M[1]:
            near = heights[np.abs(heights) <= FLOOR_PRIOR_TOLERANCE_M]
            observed = lowest_substantial_surface(near) if len(near) >= 50 else None
            return (observed, "captured_floor") if observed is not None else (0.0, "tracking_origin")
    observed = lowest_substantial_surface(heights)
    if observed is not None:
        return observed, "lowest_horizontal_surface"
    return float(np.percentile(np.asarray(mesh.vertices)[:, 1], 2)), "lowest_geometry"


def spawn_from_camera(camera_to_world: np.ndarray, floor_y: float) -> Spawn:
    forward = camera_to_world[:3, 2]  # Open3D camera looks along +Z
    yaw = float(np.degrees(np.arctan2(-forward[0], -forward[2])))
    position = camera_to_world[:3, 3]
    return Spawn(position=(float(position[0]), float(floor_y), float(position[2])), yaw_deg=yaw)


def write_glb(mesh: o3d.geometry.TriangleMesh, path: Path, *, with_colors: bool) -> int:
    """Binary GLB (BIN chunk, uint8 RGBA colors); Open3D's writer embeds base64 and is ~2x larger."""
    from mq3drecon.processing.web_location.glb import write_glb as write_binary_glb

    colors = np.asarray(mesh.vertex_colors) if with_colors and mesh.has_vertex_colors() else None
    return write_binary_glb(path, np.asarray(mesh.vertices), np.asarray(mesh.triangles), colors)
