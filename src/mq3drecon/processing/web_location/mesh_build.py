"""Fuse a recorded (typically compact) session into a mesh at build resolution (needs Warp).

The volume covers the camera trajectory plus the depth range; the voxel size
grows automatically so the dense volume fits ``max_volume_bytes``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mq3drecon.models.side import Side
from mq3drecon.processing.live.exclusion import apply_exclusion, torso_capsule
from mq3drecon.processing.live.gpu_tsdf import GpuTsdfVolume, MeshArrays, VolumeSpec
from mq3drecon.processing.live.native_depth import CameraView, color_to_image, depth_to_metric
from mq3drecon.processing.live.pose_filter import PoseKalmanFilter
from mq3drecon.streaming.messages import ColorFrame, DepthFrame, SessionStart
from mq3drecon.streaming.replay import ReplayOptions, iter_project_messages

BYTES_PER_VOXEL = 28


@dataclass(frozen=True)
class BuildResult:
    mesh: MeshArrays
    voxel_size_m: float
    first_view: CameraView
    trajectory: np.ndarray  # (N, 3) Open3D world camera positions
    frames: int


def _volume_spec(positions: np.ndarray, voxel_size_m: float, depth_max_m: float, max_volume_bytes: int) -> VolumeSpec:
    lo = positions.min(axis=0) - np.array([depth_max_m, 2.5, depth_max_m])
    hi = positions.max(axis=0) + np.array([depth_max_m, 1.5, depth_max_m])
    extent = hi - lo
    voxel = voxel_size_m
    max_voxels = max_volume_bytes // BYTES_PER_VOXEL
    while np.prod(np.ceil(extent / voxel) + 1) > max_voxels:
        voxel *= 1.1
    return VolumeSpec.around((lo + hi) / 2.0, tuple(extent), voxel, 4.0 * voxel)  # type: ignore[arg-type]


def fuse_session(
    project_dir: Path,
    *,
    voxel_size_m: float = 0.02,
    depth_max_m: float = 4.0,
    max_volume_bytes: int = 4 * 1024 ** 3,
    device: str = "cuda:0",
    max_sigma_voxels: float = 1.0,
) -> BuildResult:
    """Fuse left depth + nearest left color of every frame in the session (segment of the first frame only)."""
    options = ReplayOptions(color_sides=(Side.LEFT,), depth_sides=(Side.LEFT,), depth_encoding="f32")
    intrinsics: dict = {}
    colors: dict[int, ColorFrame] = {}
    depths: list[DepthFrame] = []
    for _, message in iter_project_messages(project_dir, options):
        if isinstance(message, SessionStart):
            intrinsics = message.intrinsics
        elif isinstance(message, ColorFrame):
            colors[message.timestamp_us] = message
        elif isinstance(message, DepthFrame):
            depths.append(message)
    metric = []
    pose_filter = PoseKalmanFilter()
    for frame in depths:
        depth = depth_to_metric(frame, depth_min_m=0.3, depth_max_m=depth_max_m)
        if depth is None:
            continue
        motion = pose_filter.update(depth.view.timestamp_us / 1e6, depth.view.position, frame.create_pose.rotation)
        if motion.is_jump and metric:
            break  # later segments are in another coordinate frame (not merged without registration)
        metric.append(apply_exclusion(depth, [torso_capsule(depth.view.position)])[0])
    if not metric:
        raise ValueError(f"No usable depth frames in {project_dir}")
    trajectory = np.array([d.view.position for d in metric])
    volume = GpuTsdfVolume(_volume_spec(trajectory, voxel_size_m, depth_max_m, max_volume_bytes), device=device)
    color_times = np.array(sorted(colors), dtype=np.int64)
    for depth in metric:
        image = None
        if len(color_times) and Side.LEFT in intrinsics:
            t = int(color_times[np.argmin(np.abs(color_times - depth.view.timestamp_us))])
            if abs(t - depth.view.timestamp_us) <= 300_000:
                image = color_to_image(colors[t], intrinsics[Side.LEFT])
        if image is not None:
            volume.integrate(depth.depth_m, depth.view.intrinsic, depth.view.world_to_camera,
                             image.rgb, image.view.intrinsic, image.view.world_to_camera)
        else:
            volume.integrate(depth.depth_m, depth.view.intrinsic, depth.view.world_to_camera)
    mesh = volume.extract_mesh(max_sigma_m=max_sigma_voxels * volume.spec.voxel_size)
    return BuildResult(mesh=mesh, voxel_size_m=volume.spec.voxel_size, first_view=metric[0].view, trajectory=trajectory, frames=len(metric))
