"""Background live reconstruction worker: streamed frames in, mesh snapshots out (needs Warp)."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from mq3drecon.config.live_config import LiveConfig
from mq3drecon.models.side import Side
from mq3drecon.processing.live.exclusion import apply_exclusion, torso_capsule
from mq3drecon.processing.live.frame_matching import ColorHistory
from mq3drecon.processing.live.gpu_tsdf import GpuTsdfVolume, MeshArrays, VolumeSpec
from mq3drecon.processing.live.native_depth import MetricDepth, color_to_image, depth_to_metric
from mq3drecon.streaming.bounded_queue import DropOldestQueue
from mq3drecon.streaming.messages import ColorFrame, DepthFrame

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveStats:
    depth_frames_received: int = 0
    frames_integrated: int = 0
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


class LiveReconstructor:
    def __init__(self, config: LiveConfig):
        self.config = config
        self._sides = tuple(Side[s.upper()] for s in config.integration.sides)
        self._history = ColorHistory()
        self._queue: DropOldestQueue[DepthFrame] = DropOldestQueue(config.integration.queue_size)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._idle = threading.Event()
        self._volume: GpuTsdfVolume | None = None
        self._intrinsics: dict[Side, dict[str, Any]] = {}
        self._active = False
        self._stats = LiveStats()
        self._trajectory: list[np.ndarray] = []
        self._last_pose: np.ndarray | None = None
        self._last_time_us: int | None = None
        self._segments: list[MeshArrays] = []
        self._snapshot: LiveSnapshot | None = None
        self._sequence = 0
        self._dirty = False
        self._last_mesh_time = 0.0
        self._thread = threading.Thread(target=self._run, name="mq3drecon-live-reconstruct", daemon=True)
        self._thread.start()

    # Session control -----------------------------------------------------
    def begin_session(self, intrinsics: dict[Side, dict[str, Any]]) -> None:
        with self._lock:
            self._volume = None
            self._intrinsics = dict(intrinsics)
            self._stats = LiveStats()
            self._trajectory = []
            self._last_pose = None
            self._last_time_us = None
            self._segments = []
            self._snapshot = None
            self._dirty = False
            self._history.clear()
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
    def volume(self) -> GpuTsdfVolume | None:
        return self._volume

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
            try:
                with self._lock:
                    if self._active:
                        self._integrate_locked(frame)
            except Exception:  # keep the live session alive; the recording is unaffected
                logger.exception("Live integration failed for depth frame %s", frame.timestamp_ms)
            if not len(self._queue):
                self._idle.set()
            self._maybe_publish()

    def _integrate_locked(self, frame: DepthFrame) -> None:
        cfg = self.config.integration
        start = time.perf_counter()
        depth = depth_to_metric(frame, depth_min_m=cfg.depth_min_m, depth_max_m=cfg.depth_max_m)
        if depth is None:
            self._stats = replace(self._stats, frames_invalid=self._stats.frames_invalid + 1)
            return
        if self._is_tracking_jump(depth):
            self._start_new_segment()
        excluded = 0
        if cfg.exclude_torso:
            depth, excluded = apply_exclusion(depth, [torso_capsule(depth.view.position, radius_m=cfg.torso_radius_m)])
        color = self._history.nearest(frame.side, depth.view.timestamp_us, int(cfg.color_tolerance_ms * 1000))
        intrinsics = self._intrinsics.get(frame.side)
        volume = self._ensure_volume(depth)
        if color is not None and intrinsics is not None:
            image = color_to_image(color, intrinsics)
            volume.integrate(depth.depth_m, depth.view.intrinsic, depth.view.world_to_camera,
                             image.rgb, image.view.intrinsic, image.view.world_to_camera)
        else:
            volume.integrate(depth.depth_m, depth.view.intrinsic, depth.view.world_to_camera)
            self._stats = replace(self._stats, frames_without_color=self._stats.frames_without_color + 1)
        self._trajectory.append(depth.view.position.copy())
        self._last_pose = depth.view.camera_to_world
        self._last_time_us = depth.view.timestamp_us
        self._dirty = True
        self._stats = replace(
            self._stats,
            frames_integrated=self._stats.frames_integrated + 1,
            excluded_pixels=self._stats.excluded_pixels + excluded,
            integrate_ms=(time.perf_counter() - start) * 1000.0,
        )

    def _is_tracking_jump(self, depth: MetricDepth) -> bool:
        """A pose change no person could make between frames: the tracking origin was reset."""
        if self._last_pose is None or self._last_time_us is None:
            return False
        distance = float(np.linalg.norm(depth.view.position - self._last_pose[:3, 3]))
        dt = max((depth.view.timestamp_us - self._last_time_us) / 1e6, 1e-3)
        cfg = self.config.integration
        return distance > cfg.jump_distance_m and distance / dt > cfg.jump_speed_mps

    def _start_new_segment(self) -> None:
        if self._volume is not None and self._volume.frames_integrated:
            mesh = self._volume.extract_mesh(min_weight=self.config.preview.min_weight)
            if len(mesh.triangles):
                self._segments.append(mesh)
        logger.warning("Tracking jump detected: starting live segment %d", len(self._segments) + 1)
        self._volume = None
        self._trajectory = []
        self._stats = replace(self._stats, tracking_jumps=self._stats.tracking_jumps + 1)

    def _ensure_volume(self, depth: MetricDepth) -> GpuTsdfVolume:
        if self._volume is None:
            vol = self.config.volume
            center = depth.view.position + np.array([0.0, vol.vertical_offset_m, 0.0])
            spec = VolumeSpec.around(center, vol.extent_m, vol.voxel_size_m, vol.voxel_size_m * vol.trunc_voxels, vol.max_weight)
            self._volume = GpuTsdfVolume(spec, device=vol.device)
            self._stats = replace(self._stats, volume_dims=spec.dims)
            logger.info("Allocated live volume %s at %.0f cm on %s", spec.dims, vol.voxel_size_m * 100, self._volume.device)
        return self._volume

    def _maybe_publish(self) -> None:
        if not self._dirty or time.monotonic() - self._last_mesh_time < self.config.preview.mesh_interval_s:
            return
        with self._lock:
            if self._volume is not None and self._active:
                self._publish_snapshot_locked()

    def _publish_snapshot_locked(self) -> None:
        assert self._volume is not None
        mesh = self._volume.extract_mesh(min_weight=self.config.preview.min_weight)
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
        )
