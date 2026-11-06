"""Operator exclusion zones: keep the capture operator's body out of the model.

Depth pixels whose 3D point falls inside a capsule are set to invalid (0)
before integration. The torso capsule hangs below the head (the depth camera
is head-mounted) along world -Y in the Open3D world frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mq3drecon.processing.live.native_depth import MetricDepth


@dataclass(frozen=True)
class Capsule:
    a: np.ndarray  # (3,) segment start, world
    b: np.ndarray  # (3,) segment end, world
    radius: float


def torso_capsule(head_position: np.ndarray, *, radius_m: float = 0.35, top_offset_m: float = 0.15, length_m: float = 1.75) -> Capsule:
    head = np.asarray(head_position, dtype=np.float64)
    top = head - np.array([0.0, top_offset_m, 0.0])
    return Capsule(a=top, b=top - np.array([0.0, length_m, 0.0]), radius=radius_m)


def _inside(points: np.ndarray, capsule: Capsule) -> np.ndarray:
    ab = capsule.b - capsule.a
    t = np.clip(((points - capsule.a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    closest = capsule.a + t[:, None] * ab
    return np.einsum("ij,ij->i", points - closest, points - closest) < capsule.radius ** 2


def apply_exclusion(depth: MetricDepth, capsules: list[Capsule]) -> tuple[MetricDepth, int]:
    """Return depth with excluded pixels zeroed, and the number of pixels removed."""
    if not capsules:
        return depth, 0
    z = depth.depth_m
    v, u = np.nonzero(z > 0)
    if len(v) == 0:
        return depth, 0
    k = depth.view.intrinsic
    zs = z[v, u].astype(np.float64)
    cam = np.stack([(u - k[0, 2]) / k[0, 0] * zs, (v - k[1, 2]) / k[1, 1] * zs, zs], axis=1)
    world = cam @ depth.view.camera_to_world[:3, :3].T + depth.view.camera_to_world[:3, 3]
    mask = np.zeros(len(v), dtype=bool)
    for capsule in capsules:
        mask |= _inside(world, capsule)
    if not mask.any():
        return depth, 0
    cleaned = z.copy()
    cleaned[v[mask], u[mask]] = 0.0
    return MetricDepth(view=depth.view, depth_m=cleaned), int(mask.sum())
