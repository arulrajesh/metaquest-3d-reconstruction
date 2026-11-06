"""Background live reconstruction worker: streamed frames in, keyframes and mesh snapshots out (needs Warp).

Every depth frame is scored by a dry run against the frames kept so far, the
keyframe selector decides (``specs/SESSION_COMPACTION.md``), and only kept
frames are fused. The live model therefore shows exactly what a compact
recording will contain.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import cv2
import numpy as np

from mq3drecon.config.live_config import LiveConfig
from mq3drecon.models.side import Side
from mq3drecon.processing.live.exclusion import apply_exclusion, torso_capsule
from mq3drecon.processing.live.frame_matching import ColorHistory
from mq3drecon.processing.live.gpu_tsdf import GpuTsdfVolume, MeshArrays, VolumeSpec
from mq3drecon.processing.live.keyframes import KeyframeDecision, KeyframePolicy, KeyframeSelector, image_sharpness
from mq3drecon.processing.live.native_depth import MetricDepth, color_to_image, depth_to_metric
from mq3drecon.processing.live.pose_filter import PoseKalmanFilter
from mq3drecon.streaming.bounded_queue import DropOldestQueue
from mq3drecon.streaming.messages import ColorFrame, DepthFrame

logger = logging.getLogger(__name__)

KeyframeCallback = Callable[[KeyframeDecision, DepthFrame], None]


@dataclass(frozen=True)
class LiveStats:
    depth_frames_received: int = 0
    frames_integrated: int = 0
    color_keyframes: int = 0
    frames_without_color: int = 0
    frames_invalid: int = 0
    frames_dropped: int = 0
    excluded_pixels: int = 0
    integrate_ms: float = 0.0
    triangles: int = 0
    volume_dims: tuple[int, int, int] | None = None
    tracking_jumps: int = 0


@dataclass(frozen=True)
class LiveSnapshot:
    sequence: int
    mesh: MeshArrays
    camera_to_world: np.ndarray | None
    trajectory: np.ndarray
    stats: LiveStats = field(default_factory=LiveStats)
    previous_segments: tuple[MeshArrays, ...] = ()
    """Meshes of earlier segments, closed when tracking jumped (each in its own coordinate frame)."""
    segment: int = 0
    keyframes: tuple[KeyframeDecision, ...] = ()


def _sharpness(frame: ColorFrame) -> float:
    if frame.encoding == "jpeg":
        gray = cv2.imdecode(np.frombuffer(frame.data, np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    else:
        rgba = np.frombuffer(frame.data, np.uint8).reshape(frame.height, frame.width, 4)
        gray = cv2.cvtColor(cv2.resize(rgba[:, :, :3], None, fx=0.25, fy=0.25), cv2.COLOR_RGB2GRAY)
    return image_sharpness(gray) if gray is not None else 0.0


class LiveReconstructor:
    def __init__(self, config: LiveConfig, *, on_keyframe: KeyframeCallback | None = None):
        self.config = config
        self.on_keyframe = on_keyframe
        self._sides = tuple(Side[s.upper()] for s in config.integration.sides)
        self._policy = KeyframePolicy(**asdict(config.keyframes))
        self._history = ColorHistory()
        self._sharpness: dict[tuple[Side, int], float] = {}
        self._queue: DropOldestQueue[DepthFrame] = DropOldestQueue(config.integration.queue_size)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._idle = threading.Event()
        self._intrinsics: dict[Side, dict[str, Any]] = {}
        self._active = False
        self._reset_session_state()
        self._snapshot: LiveSnapshot | None = None
        self._sequence = 0
        self._last_mesh_time = 0.0
        self._thread = threading.Thread(target=self._run, name="mq3drecon-live-reconstruct", daemon=True)
        self._thread.start()

    def _reset_session_state(self) -> None:
        self._volume: GpuTsdfVolume | None = None
        self._stats = LiveStats()
        self._trajectory: list[np.ndarray] = []
        self._last_pose: np.ndarray | None = None
        self._segments: list[MeshArrays] = []
        self._pose_filter = PoseKalmanFilter(jump_min_m=self.config.integration.jump_min_m)
        self._selector = KeyframeSelector(self._policy)
        self._dirty = False

    # Session control -----------------------------------------------------
    def begin_session(self, intrinsics: dict[Side, dict[str, Any]]) -> None:
        with self._lock:
            self._reset_session_state()
            self._intrinsics = dict(intrinsics)
            self._snapshot = None
            self._history.clear()
            self._sharpness.clear()
            self._queue.close()
            self._queue = DropOldestQueue(self.config.integration.queue_size)
            self._active = True

    def end_session(self, timeout_s: float = 30.0) -> LiveSnapshot | None:
        """Integrate what is still queued, then return the final snapshot."""
        deadline = time.monotonic() + timeout_s
        while len(self._queue) and time.monotonic() < deadline:
            time.sleep(0.01)
        self._idle.wait(timeout=max(0.0, deadline - time.monotonic()))
        with self._lock:
            self._active = False
            if self._volume is not None:
                self._publish_snapshot_locked()
            return self._snapshot

    def add_color(self, frame: ColorFrame) -> None:
        if self._active and frame.side in self._sides:
            self._history.add(frame)

    def add_depth(self, frame: DepthFrame) -> None:
        if not self._active or frame.side not in self._sides:
            return
        self._stats = replace(self._stats, depth_frames_received=self._stats.depth_frames_received + 1)
        self._idle.clear()
        self._queue.put(frame)

    def latest_snapshot(self) -> LiveSnapshot | None:
        return self._snapshot

    @property
    def stats(self) -> LiveStats:
        return replace(self._stats, frames_dropped=self._queue.dropped)

    @property
    def keyframes(self) -> list[KeyframeDecision]:
        return list(self._selector.decisions)

    @property
    def segment(self) -> int:
        return len(self._segments)

    def close(self) -> None:
        self._stop.set()
        self._queue.close()
        self._thread.join(timeout=5.0)

    # Worker --------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            frame = self._queue.get(timeout=0.1)
            if frame is None:
                self._idle.set()
                self._maybe_publish()
                continue
            decision = None
            try:
                with self._lock:
                    if self._active:
                        decision = self._integrate_locked(frame)
            except Exception:  # keep the live session alive; the recording is unaffected
                logger.exception("Live integration failed for depth frame %s", frame.timestamp_ms)
            if decision is not None and self.on_keyframe is not None:
                try:
                    self.on_keyframe(decision, frame)
                except Exception:
                    logger.exception("Keyframe callback failed")
            if not len(self._queue):
                self._idle.set()
            self._maybe_publish()

    def _sharpest_color(self, side: Side, timestamp_us: int) -> ColorFrame | None:
        window = int(self._policy.color_window_s * 1e6)
        candidates = self._history.within(side, timestamp_us, window)
        if not candidates:
            return None
        for frame in candidates:
            key = (side, frame.timestamp_us)
            if key not in self._sharpness:
                self._sharpness[key] = _sharpness(frame)
        return max(candidates, key=lambda f: (self._sharpness[(side, f.timestamp_us)], -abs(f.timestamp_us - timestamp_us)))

    def _integrate_locked(self, frame: DepthFrame) -> KeyframeDecision | None:
        cfg = self.config.integration
        start = time.perf_counter()
        depth = depth_to_metric(frame, depth_min_m=cfg.depth_min_m, depth_max_m=cfg.depth_max_m)
        if depth is None:
            self._stats = replace(self._stats, frames_invalid=self._stats.frames_invalid + 1)
            return None
        view = depth.view
        motion = self._pose_filter.update(view.timestamp_us / 1e6, view.position, frame.create_pose.rotation)
        if motion.is_jump and self._volume is not None:
            self._start_new_segment()
        excluded = 0
        if cfg.exclude_torso:
            depth, excluded = apply_exclusion(depth, [torso_capsule(view.position, radius_m=cfg.torso_radius_m)])
        volume = self._ensure_volume(depth)
        intrinsics = self._intrinsics.get(frame.side)
        nearest = self._history.nearest(frame.side, view.timestamp_us, int(cfg.color_tolerance_ms * 1000))
        color_view = color_to_image(nearest, intrinsics).view if nearest is not None and intrinsics is not None else None
        if color_view is not None:
            size = (nearest.height, nearest.width)
            score = volume.score(depth.depth_m, view.intrinsic, view.world_to_camera, size, color_view.intrinsic, color_view.world_to_camera)
        else:
            score = volume.score(depth.depth_m, view.intrinsic, view.world_to_camera)
        focal = float(color_view.intrinsic[0, 0]) if color_view is not None else float(view.intrinsic[0, 0])
        decision = self._selector.decide(
            view.timestamp_us, score.novelty, score.info_gain, motion, focal,
            color_novelty=score.color_novelty if color_view is not None else None,
        )
        if decision.depth_kept:
            color = self._sharpest_color(frame.side, view.timestamp_us) if decision.color_wanted and intrinsics else None
            if color is not None:
                image = color_to_image(color, intrinsics)
                volume.integrate(depth.depth_m, view.intrinsic, view.world_to_camera,
                                 image.rgb, image.view.intrinsic, image.view.world_to_camera)
                self._stats = replace(self._stats, color_keyframes=self._stats.color_keyframes + 1)
            else:
                volume.integrate(depth.depth_m, view.intrinsic, view.world_to_camera)
                if decision.color_wanted:
                    self._stats = replace(self._stats, frames_without_color=self._stats.frames_without_color + 1)
            self._dirty = True
            self._stats = replace(self._stats, frames_integrated=self._stats.frames_integrated + 1)
        self._trajectory.append(view.position.copy())
        self._last_pose = view.camera_to_world
        self._stats = replace(
            self._stats,
            excluded_pixels=self._stats.excluded_pixels + excluded,
            integrate_ms=(time.perf_counter() - start) * 1000.0,
        )
        return decision

    def _start_new_segment(self) -> None:
        if self._volume is not None and self._volume.frames_integrated:
            mesh = self._volume.extract_mesh(max_sigma_m=self._max_sigma_m())
            if len(mesh.triangles):
                self._segments.append(mesh)
        logger.warning("Tracking jump detected: starting live segment %d", len(self._segments))
        self._volume = None
        self._trajectory = []
        self._stats = replace(self._stats, tracking_jumps=self._stats.tracking_jumps + 1)

    def _ensure_volume(self, depth: MetricDepth) -> GpuTsdfVolume:
        if self._volume is None:
            vol = self.config.volume
            center = depth.view.position + np.array([0.0, vol.vertical_offset_m, 0.0])
            spec = VolumeSpec.around(center, vol.extent_m, vol.voxel_size_m, vol.voxel_size_m * vol.trunc_voxels)
            self._volume = GpuTsdfVolume(spec, device=vol.device)
            self._stats = replace(self._stats, volume_dims=spec.dims)
            logger.info("Allocated live volume %s at %.0f cm on %s", spec.dims, vol.voxel_size_m * 100, self._volume.device)
        return self._volume

    def _max_sigma_m(self) -> float:
        return self.config.preview.max_sigma_voxels * self.config.volume.voxel_size_m

    def _maybe_publish(self) -> None:
        if not self._dirty or time.monotonic() - self._last_mesh_time < self.config.preview.mesh_interval_s:
            return
        with self._lock:
            if self._volume is not None and self._active:
                self._publish_snapshot_locked()

    def _publish_snapshot_locked(self) -> None:
        assert self._volume is not None
        mesh = self._volume.extract_mesh(max_sigma_m=self._max_sigma_m())
        self._sequence += 1
        self._dirty = False
        self._last_mesh_time = time.monotonic()
        self._stats = replace(self._stats, triangles=len(mesh.triangles))
        self._snapshot = LiveSnapshot(
            sequence=self._sequence,
            mesh=mesh,
            camera_to_world=None if self._last_pose is None else self._last_pose.copy(),
            trajectory=np.asarray(self._trajectory, dtype=np.float32).reshape(-1, 3),
            stats=self.stats,
            previous_segments=tuple(self._segments),
            segment=len(self._segments),
            keyframes=tuple(self._selector.decisions),
        )
