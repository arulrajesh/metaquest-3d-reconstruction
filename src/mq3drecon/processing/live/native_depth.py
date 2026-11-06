"""Per-frame conversion of streamed Quest depth and color into integration inputs.

Uses exactly the offline pipeline's helpers so live and offline results agree:
``convert_depth_to_linear`` and ``compute_depth_camera_params`` for depth,
``mirror_principal_point_x`` for Open3D-convention intrinsics, and
``Transforms.convert_coordinate_system(OPEN3D, is_camera=True)`` for poses.
Lightweight: numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mq3drecon.models.side import Side
from mq3drecon.models.transforms import CoordinateSystem, Transforms
from mq3drecon.streaming.messages import ColorFrame, DepthFrame, Pose
from mq3drecon.streaming.payloads import decode_color_rgba, decode_depth_ndc
from mq3drecon.utils.depth_utils import (
    compute_depth_camera_params,
    convert_depth_to_linear,
    is_ndc_depth_map_valid,
    mirror_principal_point_x,
)


@dataclass(frozen=True)
class CameraView:
    """An image with Open3D-convention intrinsics and pose (Open3D world)."""

    side: Side
    timestamp_us: int
    intrinsic: np.ndarray  # (3, 3) float64
    camera_to_world: np.ndarray  # (4, 4) float64
    world_to_camera: np.ndarray  # (4, 4) float64

    @property
    def position(self) -> np.ndarray:
        return self.camera_to_world[:3, 3]


@dataclass(frozen=True)
class MetricDepth:
    view: CameraView
    depth_m: np.ndarray  # (H, W) float32; 0 marks invalid pixels


@dataclass(frozen=True)
class ColorImage:
    view: CameraView
    rgb: np.ndarray  # (H, W, 3) uint8, top-down


def unity_pose_to_open3d(pose: Pose) -> tuple[np.ndarray, np.ndarray]:
    """Return (camera_to_world, world_to_camera) 4x4 matrices in the Open3D world/camera convention."""
    unity = Transforms(
        coordinate_system=CoordinateSystem.UNITY,
        positions=np.asarray([pose.position], dtype=np.float64),
        rotations=np.asarray([pose.rotation], dtype=np.float64),
    )
    converted = unity.convert_coordinate_system(CoordinateSystem.OPEN3D, is_camera=True)
    cw = converted.extrinsics_cw[0].astype(np.float64)
    return cw, np.linalg.inv(cw)


def _intrinsic(fx: float, fy: float, cx_mirrored: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx_mirrored], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def depth_intrinsic(frame: DepthFrame) -> np.ndarray:
    fx, fy, cx, cy = compute_depth_camera_params(
        frame.fov_left, frame.fov_right, frame.fov_top, frame.fov_down, frame.width, frame.height
    )
    return _intrinsic(fx, fy, mirror_principal_point_x(cx, frame.width), cy)


def color_intrinsic(intrinsics: dict[str, Any], width: int, height: int) -> np.ndarray:
    """MRUK intrinsics JSON object to an Open3D-convention matrix, scaled if the streamed frame is resized."""
    resolution = intrinsics.get("resolution", {})
    sx = width / float(resolution.get("width", width) or width)
    sy = height / float(resolution.get("height", height) or height)
    fx = float(intrinsics["focalLength"]["x"]) * sx
    fy = float(intrinsics["focalLength"]["y"]) * sy
    cx = float(intrinsics["principalPoint"]["x"]) * sx
    cy = float(intrinsics["principalPoint"]["y"]) * sy
    return _intrinsic(fx, fy, mirror_principal_point_x(cx, width), cy)


def depth_to_metric(frame: DepthFrame, *, depth_min_m: float, depth_max_m: float) -> MetricDepth | None:
    """Convert a DEPTH message to metric depth. Returns ``None`` for buffers the offline pipeline would skip."""
    ndc = decode_depth_ndc(frame)
    if not is_ndc_depth_map_valid(ndc):
        return None
    depth = convert_depth_to_linear(ndc, frame.near_z, frame.far_z)
    depth = np.where(np.isfinite(depth) & (depth >= depth_min_m) & (depth <= depth_max_m), depth, 0.0)
    cw, wc = unity_pose_to_open3d(frame.create_pose)
    view = CameraView(frame.side, frame.timestamp_ms * 1000, depth_intrinsic(frame), cw, wc)
    return MetricDepth(view=view, depth_m=depth.astype(np.float32))


def color_to_image(frame: ColorFrame, intrinsics: dict[str, Any]) -> ColorImage:
    rgba = decode_color_rgba(frame)
    cw, wc = unity_pose_to_open3d(frame.pose)
    view = CameraView(frame.side, frame.timestamp_us, color_intrinsic(intrinsics, frame.width, frame.height), cw, wc)
    return ColorImage(view=view, rgb=np.ascontiguousarray(rgba[:, :, :3]))
