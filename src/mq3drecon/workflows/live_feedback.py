"""Laptop-to-headset feedback during live capture: keyframe notices, surface mesh, status.

The surface mesh and status use "latest wins" sending, so a slow link never
builds a backlog; keyframe notices are queued (bounded) so markers are not lost.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from mq3drecon.config.live_config import LiveConfig
from mq3drecon.streaming.feedback import KeyframeNotice, LiveStatusUpdate, pack_surface_mesh
from mq3drecon.streaming.messages import DepthFrame

logger = logging.getLogger(__name__)


def _surface_area_m2(vertices: np.ndarray, triangles: np.ndarray) -> float:
    if not len(triangles):
        return 0.0
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


class FeedbackPublisher:
    """Runs a small thread that pushes the coarse surface mesh and status to the headset."""

    def __init__(self, config: LiveConfig, receiver: Any, reconstructor: Any, controller: Any):
        self.config = config
        self.receiver = receiver
        self.reconstructor = reconstructor
        self.controller = controller
        self._stop = threading.Event()
        self._last_sequence = -1
        self._surface_area = 0.0
        self._thread = threading.Thread(target=self._run, name="mq3drecon-live-feedback", daemon=True)

    def start(self) -> None:
        if self.config.feedback.enabled:
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def on_keyframe(self, decision, frame: DepthFrame) -> None:
        if not self.config.feedback.enabled or self.receiver is None:
            return
        notice = KeyframeNotice(
            timestamp_us=decision.timestamp_us, depth_kept=decision.depth_kept, color_wanted=decision.color_wanted,
            reason=decision.reason, novelty=decision.novelty, info_gain=decision.info_gain, pose=frame.create_pose,
        )
        self.receiver.send(notice)

    def _run(self) -> None:
        cfg = self.config.feedback
        next_surface = next_status = 0.0
        tick = 0.0
        while not self._stop.wait(0.2):
            tick += 0.2
            if tick >= next_surface:
                next_surface = tick + cfg.surface_interval_s
                self._send_surface()
            if tick >= next_status:
                next_status = tick + cfg.status_interval_s
                self._send_status()

    def _send_surface(self) -> None:
        snapshot = self.reconstructor.latest_snapshot()
        if snapshot is None or snapshot.sequence == self._last_sequence or not len(snapshot.mesh.triangles):
            return
        self._last_sequence = snapshot.sequence
        try:
            from mq3drecon.processing.live.headset_mesh import coarse_mesh

            cfg = self.config.feedback
            vertices, triangles, rgb, confidence = coarse_mesh(
                snapshot.mesh, voxel_size_m=self.config.volume.voxel_size_m,
                cluster_m=cfg.surface_cluster_m, max_triangles=cfg.surface_max_triangles,
            )
            self._surface_area = _surface_area_m2(snapshot.mesh.vertices.astype(np.float64), snapshot.mesh.triangles)
            if len(triangles):
                message = pack_surface_mesh(vertices, triangles, rgb, confidence, sequence=snapshot.sequence, segment=snapshot.segment)
                self.receiver.send(message, replace_key="surface")
        except Exception:
            logger.exception("Surface feedback failed")

    def _send_status(self) -> None:
        stats = self.reconstructor.stats
        status = self.controller.status()
        snapshot = self.reconstructor.latest_snapshot()
        sigma = snapshot.mesh.sigma_m if snapshot is not None and snapshot.mesh.sigma_m is not None else np.zeros(0)
        values = {
            "session": status.session_name or "",
            "recording": status.session_name is not None,
            "depth_frames": stats.depth_frames_received,
            "keyframes": stats.frames_integrated,
            "color_keyframes": stats.color_keyframes,
            "triangles": stats.triangles,
            "surface_m2": round(self._surface_area, 2),
            "median_sigma_mm": round(float(np.median(sigma)) * 1000.0, 1) if len(sigma) else 0.0,
            "dropped_live": stats.frames_dropped,
            "dropped_headset": sum(status.headset_dropped.values()),
            "tracking_jumps": stats.tracking_jumps,
            "segment": self.reconstructor.segment,
        }
        self.receiver.send(LiveStatusUpdate(values=values), replace_key="status")
