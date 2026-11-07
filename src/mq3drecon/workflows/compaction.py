"""Session compaction: keep only the frames worth keeping (Kalman information gain + blur).

Pass 1 scores every left depth frame with a Kalman TSDF and the pose filter and
selects keyframes; for each keyframe the sharpest color pair nearby is chosen.
Pass 2 writes a new compact session (JPEG color, 16-bit PNG left depth). The
source is left untouched unless ``delete_source`` is set. Needs Warp.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from mq3drecon.config.project_path_config import ProjectPathConfig
from mq3drecon.models.side import Side
from mq3drecon.processing.live.keyframes import KeyframeDecision, KeyframePolicy, KeyframeSelector, select_color_frames
from mq3drecon.processing.live.pose_filter import PoseKalmanFilter
from mq3drecon.streaming.messages import ColorFrame, DepthFrame, SessionEnd, SessionStart
from mq3drecon.streaming.replay import ReplayOptions, iter_project_messages
from mq3drecon.streaming.session_recorder import RecordingOptions, SessionRecorder

KEYFRAMES_CSV = "keyframes.csv"
COMPACTION_JSON = "compaction.json"
_COPIED_FILES = ("hmd_poses.csv", "left_controller_poses.csv", "right_controller_poses.csv", "room_model.json", "markers.json")


def _detail_windows(project_dir: Path) -> list[tuple[int, int]]:
    """(start_us, end_us) of the operator's "detail here" markers: every color frame inside is kept."""
    path = Path(project_dir) / "markers.json"
    if not path.exists():
        return []
    windows = []
    for marker in json.loads(path.read_text(encoding="utf-8")):
        if marker.get("kind") == "detail" and "start_us" in marker and "end_us" in marker:
            windows.append((int(marker["start_us"]), int(marker["end_us"])))
    return windows


@dataclass
class CompactionReport:
    source: str
    output: str
    frames_before: dict[str, int] = field(default_factory=dict)
    frames_after: dict[str, int] = field(default_factory=dict)
    bytes_before: int = 0
    bytes_after: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    @property
    def ratio(self) -> float:
        return self.bytes_before / self.bytes_after if self.bytes_after else float("inf")


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def score_depth_keyframes(
    project_dir: Path, policy: KeyframePolicy, *, voxel_size_m: float = 0.04, device: str = "cuda:0"
) -> list[KeyframeDecision]:
    """Run every left depth frame through a Kalman TSDF and the pose filter; return keyframe decisions."""
    from mq3drecon.processing.live.gpu_tsdf import GpuTsdfVolume, VolumeSpec
    from mq3drecon.processing.live.native_depth import depth_to_metric

    from mq3drecon.processing.live.native_depth import color_intrinsic, unity_pose_to_open3d

    selector, pose_filter, volume = KeyframeSelector(policy), PoseKalmanFilter(), None
    options = ReplayOptions(color_sides=(), depth_sides=(Side.LEFT,), depth_encoding="f32")
    paths = ProjectPathConfig(project_dir)
    meta = pd.read_csv(paths.image.get_mruk_frame_metadata_csv_path(Side.LEFT)).sort_values("timestamp_us_realtime")
    intrinsics = json.loads(paths.image.get_mruk_intrinsics_json_path(Side.LEFT).read_text(encoding="utf-8"))
    color_times = meta.timestamp_us_realtime.to_numpy()
    for _, message in iter_project_messages(project_dir, options):
        if not isinstance(message, DepthFrame):
            continue
        depth = depth_to_metric(message, depth_min_m=0.3, depth_max_m=5.0)
        if depth is None:
            continue
        view = depth.view
        motion = pose_filter.update(view.timestamp_us / 1e6, view.position, message.create_pose.rotation)
        if volume is None or motion.is_jump:
            center = view.position + np.array([0.0, -0.8, 0.0])
            volume = GpuTsdfVolume(VolumeSpec.around(center, (16.0, 8.0, 16.0), voxel_size_m, 4 * voxel_size_m), device=device)
        # Causal: score against the frames kept so far (dry run), integrate only what is kept.
        row = meta.iloc[int(np.argmin(np.abs(color_times - view.timestamp_us)))]
        size = (int(row.height), int(row.width))
        k_color = color_intrinsic(intrinsics, size[1], size[0])
        _, wc_color = unity_pose_to_open3d(_row_pose(row))
        score = volume.score(depth.depth_m, view.intrinsic, view.world_to_camera, size, k_color, wc_color)
        decision = selector.decide(
            view.timestamp_us, score.novelty, score.info_gain, motion,
            focal_px=float(k_color[0, 0]), color_novelty=score.color_novelty,
        )
        if decision.depth_kept:
            color_args = (np.asarray(size), k_color, wc_color) if decision.color_wanted else (None, None, None)
            volume.integrate(depth.depth_m, view.intrinsic, view.world_to_camera, *color_args)
    return selector.decisions


def _row_pose(row):
    from mq3drecon.streaming.messages import Pose

    return Pose(
        (float(row.pose_pos_x), float(row.pose_pos_y), float(row.pose_pos_z)),
        (float(row.pose_rot_x), float(row.pose_rot_y), float(row.pose_rot_z), float(row.pose_rot_w)),
    )


def _color_sharpness(project_dir: Path, side: Side) -> list[float]:
    """Measured sharpness of every color frame, at 1/4 resolution (fast JPEG reduced decode)."""
    import cv2

    from mq3drecon.dataio.mruk_frame_codec import RAW_SUFFIX, read_mruk_frame_rgba
    from mq3drecon.processing.live.keyframes import image_sharpness

    paths = ProjectPathConfig(project_dir)
    meta = pd.read_csv(paths.image.get_mruk_frame_metadata_csv_path(side)).sort_values("timestamp_us_realtime")
    frame_dir = paths.image.get_mruk_rgba_dir(side)
    values = []
    for row in meta.itertuples(index=False):
        path = frame_dir / str(row.file_name)
        if path.suffix.lower() == RAW_SUFFIX:
            rgba = read_mruk_frame_rgba(path, int(row.width), int(row.height))
            gray = cv2.cvtColor(cv2.resize(rgba[:, :, :3], None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
        values.append(image_sharpness(gray) if gray is not None else 0.0)
    return values


def _color_blur(project_dir: Path, side: Side, policy: KeyframePolicy) -> tuple[list[int], list[float]]:
    paths = ProjectPathConfig(project_dir)
    meta = pd.read_csv(paths.image.get_mruk_frame_metadata_csv_path(side)).sort_values("timestamp_us_realtime")
    intrinsics = json.loads(paths.image.get_mruk_intrinsics_json_path(side).read_text(encoding="utf-8"))
    from mq3drecon.dataio.mruk_intrinsics import image_pinhole

    focal = image_pinhole(intrinsics)[0]  # in image pixels (MRUK reports sensor pixels)
    pose_filter = PoseKalmanFilter()
    times, blur = [], []
    for row in meta.itertuples(index=False):
        t = int(row.timestamp_us_realtime)
        motion = pose_filter.update(
            t / 1e6,
            (row.pose_pos_x, row.pose_pos_y, row.pose_pos_z),
            (row.pose_rot_x, row.pose_rot_y, row.pose_rot_z, row.pose_rot_w),
        )
        times.append(t)
        blur.append(motion.predicted_blur_px(focal, policy.exposure_s))
    return times, blur


def _frame_counts(root: Path) -> dict[str, int]:
    def count(directory: Path) -> int:
        return len([p for p in directory.iterdir() if p.is_file()]) if directory.exists() else 0

    return {
        "color_left": count(root / "left_camera_mruk_rgba"),
        "color_right": count(root / "right_camera_mruk_rgba"),
        "depth_left": count(root / "left_depth"),
        "depth_right": count(root / "right_depth"),
    }


def compact_session(
    project_dir: Path,
    output_dir: Path,
    *,
    policy: KeyframePolicy | None = None,
    device: str = "cuda:0",
    jpeg_quality: int = 90,
    delete_source: bool = False,
) -> CompactionReport:
    project_dir, output_dir = Path(project_dir), Path(output_dir)
    policy = policy or KeyframePolicy()
    decisions = score_depth_keyframes(project_dir, policy, device=device)
    kept_depth_us = [d.timestamp_us for d in decisions if d.depth_kept]
    color_keyframes_us = [d.timestamp_us for d in decisions if d.color_wanted]
    left_times, left_blur = _color_blur(project_dir, Side.LEFT, policy)
    targets_us = list(color_keyframes_us)
    if policy.splat_color_gap_s and left_times:
        # Splat views: the sharpest frame around every gap step, on top of the TSDF color keyframes.
        step = max(int(policy.splat_color_gap_s * 1e6), 1)
        targets_us += list(range(min(left_times), max(left_times) + 1, step))
    chosen_left = set(select_color_frames(
        targets_us, left_times, left_blur, policy, sharpness=_color_sharpness(project_dir, Side.LEFT)
    ))
    for start_us, end_us in _detail_windows(project_dir):
        chosen_left |= {t for t in left_times if start_us <= t <= end_us}
    right_meta = pd.read_csv(ProjectPathConfig(project_dir).image.get_mruk_frame_metadata_csv_path(Side.RIGHT))
    right_times = sorted(int(t) for t in right_meta.timestamp_us_realtime)
    chosen_right = {min(right_times, key=lambda r: abs(r - t)) for t in chosen_left} if right_times else set()

    options = ReplayOptions(
        jpeg_quality=jpeg_quality,
        depth_encoding="f32",
        depth_sides=(Side.LEFT,),
        color_timestamps_us=frozenset(chosen_left | chosen_right),
        depth_timestamps_ms=frozenset(t // 1000 for t in kept_depth_us),
    )
    recorder: SessionRecorder | None = None
    for _, message in iter_project_messages(project_dir, options):
        if isinstance(message, SessionStart):
            recorder = SessionRecorder(output_dir, SessionStart(output_dir.name, message.intrinsics), RecordingOptions())
        elif recorder is not None and isinstance(message, ColorFrame):
            if message.timestamp_us in (chosen_left if message.side == Side.LEFT else chosen_right):
                recorder.write_color(message)
        elif recorder is not None and isinstance(message, DepthFrame):
            recorder.write_depth(message)
        elif recorder is not None and isinstance(message, SessionEnd):
            recorder.finalize("compacted")
    for name in _COPIED_FILES:
        if (project_dir / name).exists():
            shutil.copy2(project_dir / name, output_dir / name)
    pd.DataFrame([asdict(d) for d in decisions]).to_csv(output_dir / KEYFRAMES_CSV, index=False)

    report = CompactionReport(source=str(project_dir), output=str(output_dir))
    report.frames_before, report.frames_after = _frame_counts(project_dir), _frame_counts(output_dir)
    for decision in decisions:
        report.reasons[decision.reason] = report.reasons.get(decision.reason, 0) + 1
    report.bytes_before, report.bytes_after = _dir_bytes(project_dir), _dir_bytes(output_dir)
    (output_dir / COMPACTION_JSON).write_text(json.dumps({**asdict(report), "policy": asdict(policy)}, indent=2), encoding="utf-8")
    if delete_source:
        shutil.rmtree(project_dir)
    return report
