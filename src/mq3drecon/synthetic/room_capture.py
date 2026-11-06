"""Synthetic MRUK capture of an analytic room, for testing without a headset.

The generator inverts the conventions the existing offline pipeline uses to
interpret real recordings:

* poses are Unity camera-to-world and converted with
  ``Transforms.convert_coordinate_system(OPEN3D, is_camera=True)``;
* Quest depth intrinsics come from ``compute_depth_camera_params`` and MRUK
  color intrinsics from the intrinsics JSON, both with the principal point x
  mirrored (``mirror_principal_point_x``) for Open3D;
* depth is stored as raw NDC float32 rows exactly as ``DepthDataIO`` reads them;
* ``.rgba`` color is stored bottom-up, ``.jpg`` top-down.

So any code that reconstructs this capture correctly using the shared helpers
also follows the offline pipeline's interpretation of real data.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from mq3drecon.models.transforms import CoordinateSystem, Transforms
from mq3drecon.utils.depth_utils import (
    compute_depth_camera_params,
    compute_ndc_to_linear_depth_params,
    mirror_principal_point_x,
)
from mq3drecon.dataio.qrc_csv_headers import COLOR_METADATA_HEADER, DEPTH_DESCRIPTOR_HEADER, STEREO_PAIR_HEADER

ROOM_MIN = np.array([-2.0, 0.0, -2.5])
ROOM_MAX = np.array([2.0, 2.6, 2.5])
TABLE_MIN = np.array([0.3, 0.0, 0.8])
TABLE_MAX = np.array([1.1, 0.75, 1.6])
DEPTH_FOV = {"left": 1.05, "right": 0.95, "top": 1.0, "down": 0.9}
UNITY_TO_OPEN3D_WORLD = np.diag([1.0, 1.0, -1.0])


@dataclass(frozen=True)
class SyntheticCaptureSpec:
    frames: int = 40
    fps: float = 10.0
    color_size: int = 320
    depth_size: int = 160
    color_format: str = "rgba"
    radius_m: float = 0.8
    eye_height_m: float = 1.6
    pitch_down_deg: float = 20.0
    near_z: float = 0.1
    far_z: float = float("inf")
    base_time_us: int = 1_781_849_907_000_000


def scene_color(points_unity: np.ndarray) -> np.ndarray:
    """Smooth, position-unique RGB in [0, 1] (mirror or offset errors show up as color errors)."""
    p = np.asarray(points_unity, dtype=np.float64)
    return np.stack(
        [
            0.5 + 0.35 * np.sin(1.9 * p[..., 0] + 0.4),
            0.5 + 0.35 * np.sin(2.3 * p[..., 1] + 1.3),
            0.5 + 0.35 * np.sin(1.7 * p[..., 2] + 2.1),
        ],
        axis=-1,
    )


def surface_distance(points_unity: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest room or table surface (Unity world, meters)."""
    p = np.asarray(points_unity, dtype=np.float64)
    room = np.min(np.concatenate([np.abs(p - ROOM_MIN), np.abs(ROOM_MAX - p)], axis=-1), axis=-1)
    outside = np.linalg.norm(np.maximum(np.maximum(TABLE_MIN - p, 0.0), p - TABLE_MAX), axis=-1)
    inside = np.min(np.concatenate([p - TABLE_MIN, TABLE_MAX - p], axis=-1), axis=-1)
    table = np.where(outside > 0, outside, np.abs(inside))
    return np.minimum(room, table)


def head_pose(t: float, spec: SyntheticCaptureSpec) -> tuple[np.ndarray, np.ndarray]:
    """Head position and camera-to-world rotation matrix (columns right, up, forward) in Unity world."""
    theta = 2.0 * np.pi * t / max(spec.frames / spec.fps, 1e-6)
    pitch = np.deg2rad(spec.pitch_down_deg)
    position = np.array([spec.radius_m * np.cos(theta), spec.eye_height_m, spec.radius_m * np.sin(theta)])
    forward = np.array([np.cos(theta) * np.cos(pitch), -np.sin(pitch), np.sin(theta) * np.cos(pitch)])
    right = np.cross([0.0, 1.0, 0.0], forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return position, np.stack([right, up, forward], axis=1)


def _camera_pose(t: float, spec: SyntheticCaptureSpec, offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position, rotation = head_pose(t, spec)
    return position + rotation @ offset, rotation


def _ray_hits(origins_u: np.ndarray, dirs_u: np.ndarray) -> np.ndarray:
    """Ray parameter of the first hit with the room (from inside) or the table."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs_u
        to_max, to_min = (ROOM_MAX - origins_u) * inv, (ROOM_MIN - origins_u) * inv
        t_room = np.nanmin(np.where(dirs_u > 0, to_max, np.where(dirs_u < 0, to_min, np.inf)), axis=-1)
        t1, t2 = (TABLE_MIN - origins_u) * inv, (TABLE_MAX - origins_u) * inv
        t_near = np.nanmax(np.minimum(t1, t2), axis=-1)
        t_far = np.nanmin(np.maximum(t1, t2), axis=-1)
    hit_table = (t_near <= t_far) & (t_near > 1e-6)
    return np.where(hit_table, np.minimum(t_near, t_room), t_room)


def render(position_u: np.ndarray, rotation_u: np.ndarray, k_open3d: np.ndarray, width: int, height: int):
    """Return (z_depth, hit_points_unity) for a camera, using the pipeline's Open3D interpretation."""
    quat = R.from_matrix(rotation_u).as_quat()
    unity = Transforms(CoordinateSystem.UNITY, position_u[None].astype(np.float64), quat[None])
    cw = unity.convert_coordinate_system(CoordinateSystem.OPEN3D, is_camera=True).extrinsics_cw[0].astype(np.float64)
    vs, us = np.mgrid[0:height, 0:width].astype(np.float64)
    dirs_cam = np.stack([(us - k_open3d[0, 2]) / k_open3d[0, 0], (vs - k_open3d[1, 2]) / k_open3d[1, 1], np.ones_like(us)], -1)
    dirs_world_o3d = dirs_cam @ cw[:3, :3].T
    origin_o3d = cw[:3, 3]
    dirs_u = dirs_world_o3d @ UNITY_TO_OPEN3D_WORLD.T
    origin_u = np.broadcast_to(UNITY_TO_OPEN3D_WORLD @ origin_o3d, dirs_u.shape)
    t = _ray_hits(origin_u, dirs_u)  # dirs_cam has z = 1, so t is the camera-frame depth
    return t.astype(np.float32), origin_u + dirs_u * t[..., None]


def _color_intrinsics(size: int) -> tuple[dict, np.ndarray]:
    fx = fy = 0.5 * size / np.tan(np.deg2rad(40.0))
    cx_o3d, cy = 0.52 * size, 0.47 * size
    cx_json = mirror_principal_point_x(cx_o3d, size)  # the JSON stores the un-mirrored value
    k = np.array([[fx, 0, cx_o3d], [0, fy, cy], [0, 0, 1]])
    obj = {
        "backend": "MRUK",
        "imageFormat": "RGBA32",
        "focalLength": {"x": fx, "y": fy},
        "principalPoint": {"x": cx_json, "y": cy},
        "resolution": {"width": size, "height": size},
    }
    return obj, k


def _depth_intrinsics(size: int) -> np.ndarray:
    fx, fy, cx, cy = compute_depth_camera_params(
        DEPTH_FOV["left"], DEPTH_FOV["right"], DEPTH_FOV["top"], DEPTH_FOV["down"], size, size
    )
    return np.array([[fx, 0, mirror_principal_point_x(cx, size)], [0, fy, cy], [0, 0, 1]])


def _ndc_from_depth(z: np.ndarray, near: float, far: float) -> np.ndarray:
    x, y = compute_ndc_to_linear_depth_params(near, far)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndc = x / z - y
    d = np.where(np.isfinite(z) & (z > 0), (ndc + 1.0) / 2.0, 1.0)
    return np.clip(d, 0.0, 1.0).astype(np.float32)


def _quat_fields(rotation: np.ndarray) -> list[float]:
    return [float(v) for v in R.from_matrix(rotation).as_quat()]


def write_synthetic_capture(output_dir: Path, spec: SyntheticCaptureSpec | None = None) -> Path:
    """Write a synthetic MRUK project. ``output_dir`` must not exist or must be empty."""
    spec = spec or SyntheticCaptureSpec()
    if spec.color_format not in ("rgba", "jpg"):
        raise ValueError("color_format must be 'rgba' or 'jpg'")
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "session_info.json").write_text(json.dumps({"sessionFormatVersion": 2, "captureBackend": "MRUK"}), encoding="utf-8")

    color_obj, k_color = _color_intrinsics(spec.color_size)
    k_depth = _depth_intrinsics(spec.depth_size)
    eyes = {"left": -0.032, "right": 0.032}
    pair_rows = []
    for side, eye_x in eyes.items():
        _write_color_side(out, side, eye_x, spec, color_obj, k_color, pair_rows)
        _write_depth_side(out, side, eye_x, spec, k_depth)

    with open(out / "mruk_stereo_pairs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(STEREO_PAIR_HEADER)
        for i in range(spec.frames):
            left_ts, right_ts = _color_ts(spec, i, "left"), _color_ts(spec, i, "right")
            ext = "jpg" if spec.color_format == "jpg" else "rgba"
            writer.writerow([i + 1, f"{left_ts}.{ext}", f"{right_ts}.{ext}", left_ts, right_ts, right_ts - left_ts, i, i])
    return out


def _color_ts(spec: SyntheticCaptureSpec, i: int, side: str) -> int:
    return spec.base_time_us + int(round(i / spec.fps * 1e6)) + (250 if side == "right" else 0)


def _write_color_side(out, side, eye_x, spec, color_obj, k_color, pair_rows) -> None:
    name = side.capitalize()
    (out / f"{side}_camera_mruk_intrinsics.json").write_text(json.dumps({**color_obj, "cameraPosition": name}), encoding="utf-8")
    frame_dir = out / f"{side}_camera_mruk_rgba"
    frame_dir.mkdir()
    size = spec.color_size
    with open(out / f"{side}_camera_mruk_frame_metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLOR_METADATA_HEADER)
        for i in range(spec.frames):
            ts = _color_ts(spec, i, side)
            position, rotation = _camera_pose((ts - spec.base_time_us) / 1e6, spec, np.array([eye_x, -0.02, 0.05]))
            _, hits = render(position, rotation, k_color, size, size)
            rgb = (scene_color(hits) * 255.0 + 0.5).astype(np.uint8)
            rgba = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], axis=-1)
            if spec.color_format == "jpg":
                file_name, fmt = f"{ts}.jpg", "JPEG"
                cv2.imwrite(str(frame_dir / file_name), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                file_name, fmt = f"{ts}.rgba", "RGBA32"
                np.ascontiguousarray(np.flipud(rgba)).tofile(frame_dir / file_name)
            pixels = size * size
            writer.writerow([
                i, file_name, ts // 1000, ts, size, size, fmt, name,
                *[float(v) for v in position], *_quat_fields(rotation),
                "true", "true", "true", size, size, pixels, pixels, pixels * 4, "true", "true", "",
            ])


def _write_depth_side(out, side, eye_x, spec, k_depth) -> None:
    depth_dir = out / f"{side}_depth"
    depth_dir.mkdir()
    size = spec.depth_size
    with open(out / f"{side}_depth_descriptors.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(DEPTH_DESCRIPTOR_HEADER)
        for i in range(spec.frames):
            t = i / spec.fps + 0.015
            ts_ms = spec.base_time_us // 1000 + int(round(t * 1000))
            position, rotation = _camera_pose(t, spec, np.array([eye_x * 0.9, 0.0, 0.03]))
            z, _ = render(position, rotation, k_depth, size, size)
            _ndc_from_depth(z, spec.near_z, spec.far_z).astype("<f4").tofile(depth_dir / f"{ts_ms}.raw")
            writer.writerow([
                ts_ms, t, *[float(v) for v in position], *_quat_fields(rotation),
                DEPTH_FOV["left"], DEPTH_FOV["right"], DEPTH_FOV["top"], DEPTH_FOV["down"],
                spec.near_z, "inf" if np.isinf(spec.far_z) else spec.far_z, size, size,
            ])
