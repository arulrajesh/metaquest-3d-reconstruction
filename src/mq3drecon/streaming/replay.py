"""Turn a recorded MRUK project into a live message stream (replay sender and offline builds)."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mq3drecon.config.project_path_config import ProjectPathConfig
from mq3drecon.dataio.mruk_frame_codec import RAW_SUFFIX, read_mruk_frame_rgba
from mq3drecon.models.side import Side
from mq3drecon.streaming.messages import ColorFrame, DepthFrame, Message, Pose, SessionEnd, SessionStart
from mq3drecon.streaming.payloads import encode_color_jpeg, encode_depth_ndc
from mq3drecon.streaming.sender import LiveSender


@dataclass(frozen=True)
class ReplayOptions:
    jpeg_quality: int = 90
    depth_encoding: str = "u16_ndc"
    color_sides: tuple[Side, ...] = (Side.LEFT, Side.RIGHT)
    depth_sides: tuple[Side, ...] = (Side.LEFT, Side.RIGHT)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in ("true", "1")


def _color_frames(paths: ProjectPathConfig, side: Side, options: ReplayOptions) -> Iterator[tuple[int, Message]]:
    metadata_path = paths.image.get_mruk_frame_metadata_csv_path(side)
    if not metadata_path.exists():
        return
    frame_dir = paths.image.get_mruk_rgba_dir(side)
    for row in pd.read_csv(metadata_path).itertuples(index=False):
        if not (_truthy(row.get_texture_ok) and _truthy(row.get_colors_ok)):
            continue
        path = frame_dir / str(row.file_name)
        if not path.exists():
            continue
        width, height = int(row.width), int(row.height)
        if path.suffix.lower() == RAW_SUFFIX:
            data = encode_color_jpeg(read_mruk_frame_rgba(path, width, height), options.jpeg_quality)
        else:
            data = path.read_bytes()
        timestamp_us = int(row.timestamp_us_realtime)
        yield timestamp_us, ColorFrame(
            side=side,
            frame_index=int(row.frame_index),
            timestamp_us=timestamp_us,
            unix_time_ms=int(row.unix_time_ms),
            width=width,
            height=height,
            encoding="jpeg",
            row_order="top_down",
            pose=Pose(
                position=(float(row.pose_pos_x), float(row.pose_pos_y), float(row.pose_pos_z)),
                rotation=(float(row.pose_rot_x), float(row.pose_rot_y), float(row.pose_rot_z), float(row.pose_rot_w)),
            ),
            pose_ok=_truthy(row.pose_ok),
            data=data,
        )


def _depth_frames(paths: ProjectPathConfig, side: Side, options: ReplayOptions) -> Iterator[tuple[int, Message]]:
    descriptor_path = paths.depth.get_depth_descriptor_path(side)
    if not descriptor_path.exists():
        return
    for row in pd.read_csv(descriptor_path).itertuples(index=False):
        timestamp_ms = int(row.timestamp_ms)
        path = paths.depth.get_depth_map_path(side=side, timestamp=timestamp_ms)
        if not path.exists():
            continue
        width, height = int(row.width), int(row.height)
        ndc = np.fromfile(path, dtype="<f4").reshape(height, width)
        yield timestamp_ms * 1000, DepthFrame(
            side=side,
            timestamp_ms=timestamp_ms,
            ovr_timestamp=float(row.ovr_timestamp),
            width=width,
            height=height,
            encoding=options.depth_encoding,
            create_pose=Pose(
                position=(float(row.create_pose_location_x), float(row.create_pose_location_y), float(row.create_pose_location_z)),
                rotation=(
                    float(row.create_pose_rotation_x),
                    float(row.create_pose_rotation_y),
                    float(row.create_pose_rotation_z),
                    float(row.create_pose_rotation_w),
                ),
            ),
            fov_left=float(row.fov_left_angle_tangent),
            fov_right=float(row.fov_right_angle_tangent),
            fov_top=float(row.fov_top_angle_tangent),
            fov_down=float(row.fov_down_angle_tangent),
            near_z=float(row.near_z),
            far_z=float(row.far_z),
            data=encode_depth_ndc(ndc, options.depth_encoding),
        )


def load_session_start(project_dir: Path, session_name: str | None = None) -> SessionStart:
    paths = ProjectPathConfig(project_dir)
    intrinsics = {}
    for side in Side:
        path = paths.image.get_mruk_intrinsics_json_path(side)
        if path.exists():
            intrinsics[side] = json.loads(path.read_text(encoding="utf-8"))
    return SessionStart(session_name=session_name or Path(project_dir).name, intrinsics=intrinsics)


def iter_project_messages(project_dir: Path, options: ReplayOptions | None = None) -> Iterator[tuple[int, Message]]:
    """Yield ``(timestamp_us, message)`` in capture-time order, framed by SESSION_START and SESSION_END."""
    options = options or ReplayOptions()
    paths = ProjectPathConfig(project_dir)
    timed: list[tuple[int, int, Message]] = []
    order = 0
    for side in options.color_sides:
        for timestamp_us, message in _color_frames(paths, side, options):
            timed.append((timestamp_us, order, message))
            order += 1
    for side in options.depth_sides:
        for timestamp_us, message in _depth_frames(paths, side, options):
            timed.append((timestamp_us, order, message))
            order += 1
    timed.sort(key=lambda item: (item[0], item[1]))
    first = timed[0][0] if timed else 0
    yield first, load_session_start(project_dir)
    for timestamp_us, _order, message in timed:
        yield timestamp_us, message
    yield (timed[-1][0] if timed else first), SessionEnd(reason="replay_complete")


def replay_project(
    project_dir: Path,
    host: str,
    port: int,
    *,
    speed: float = 1.0,
    options: ReplayOptions | None = None,
) -> int:
    """Stream a recorded project to a live receiver. ``speed <= 0`` sends as fast as possible. Returns bytes sent."""
    start_wall = time.monotonic()
    first_us: int | None = None
    with LiveSender(host, port) as sender:
        for timestamp_us, message in iter_project_messages(project_dir, options):
            if first_us is None:
                first_us = timestamp_us
            if speed > 0:
                due = start_wall + (timestamp_us - first_us) / 1e6 / speed
                delay = due - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            sender.send(message)
        return sender.bytes_sent
