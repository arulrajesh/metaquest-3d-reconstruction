"""Keyframe selection from Kalman information gain and predicted blur (lightweight).

Depth and color are judged separately:

* **Depth** is kept when it carries enough *useful* information (absolute
  variance reduction above the target precision, scored by a dry run against
  the frames kept so far). Depth is cheap (16-bit PNG) and geometry needs
  repeated looks, so the bar is low.
* **Color** (the expensive JPEG texture source) is kept only at a depth
  keyframe that is novel, sharp, and not too close in time to the previous color
  keyframe. A coverage guarantee keeps one after ``max_gap_s`` while moving.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np

from mq3drecon.processing.live.pose_filter import PoseFilterResult


@dataclass(frozen=True)
class KeyframePolicy:
    depth_min_info_gain: float = 1000.0
    """Useful variance reduction (voxel-equivalents) a depth frame must add. Measured on real data:
    keeping frames above 1000 kept 76% of an active 14 s scan with 99% coverage of the full model."""
    color_min_novelty: float = 0.15
    """Share of the visible surface still lacking good color that justifies a new color keyframe."""
    color_min_gap_s: float = 0.5
    max_gap_s: float = 2.0
    max_blur_px: float = 12.0
    """Predicted blur beyond which a frame is rejected without looking at it (only extreme head turns).
    Ordinary blur is handled by picking the locally sharpest color frame (``select_color_frames``)."""
    exposure_s: float = 0.004
    """Assumed passthrough exposure for blur prediction (not reported by the Quest API). On real data,
    image sharpness could not calibrate it: sharpness varies more with scene content than with head speed."""
    moving_speed_mps: float = 0.05
    moving_angular_dps: float = 5.0
    color_window_s: float = 0.25
    splat_color_gap_s: float | None = 0.25
    """Compaction also keeps the sharpest color frame around every ``splat_color_gap_s`` step, for Gaussian
    splats. The color keyframes above are tuned for TSDF coloring (~1/s), but splats need many views to
    generalize: on the real recapture (2026-11-06), halving 286 views cost 3 dB of held-out PSNR. None keeps
    only the TSDF color keyframes."""


@dataclass(frozen=True)
class KeyframeDecision:
    timestamp_us: int
    depth_kept: bool
    color_wanted: bool
    reason: str
    novelty: float
    info_gain: float
    blur_px: float

    @property
    def kept(self) -> bool:
        return self.depth_kept


class KeyframeSelector:
    def __init__(self, policy: KeyframePolicy | None = None):
        self.policy = policy or KeyframePolicy()
        self._last_depth_us: int | None = None
        self._last_color_us: int | None = None
        self.decisions: list[KeyframeDecision] = []

    def decide(
        self,
        timestamp_us: int,
        novelty: float,
        info_gain: float,
        motion: PoseFilterResult,
        focal_px: float,
        color_novelty: float | None = None,
    ) -> KeyframeDecision:
        """``color_novelty`` (from a color-aware dry run) drives color keyframes; without it, geometric novelty is used."""
        p = self.policy
        t = int(timestamp_us)
        blur = motion.predicted_blur_px(focal_px, p.exposure_s)
        moving = motion.speed_mps > p.moving_speed_mps or motion.angular_speed_dps > p.moving_angular_dps
        depth_gap = None if self._last_depth_us is None else (t - self._last_depth_us) / 1e6
        if depth_gap is None:
            depth_kept, reason = True, "first"
        elif motion.is_jump:
            depth_kept, reason = True, "segment_start"
        elif info_gain >= p.depth_min_info_gain:
            depth_kept, reason = True, "informative"
        elif depth_gap >= p.max_gap_s and moving:
            depth_kept, reason = True, "coverage_gap"
        else:
            depth_kept, reason = False, "redundant"

        color_gap = None if self._last_color_us is None else (t - self._last_color_us) / 1e6
        color_need = novelty if color_novelty is None else color_novelty
        sharp = blur <= p.max_blur_px
        color_wanted = depth_kept and sharp and (
            color_gap is None
            or reason == "segment_start"
            or (color_gap >= p.color_min_gap_s and (color_need >= p.color_min_novelty or (color_gap >= p.max_gap_s and moving)))
        )
        if depth_kept and not sharp:
            reason += "+blurry"
        decision = KeyframeDecision(t, depth_kept, bool(color_wanted), reason, float(novelty), float(info_gain), float(blur))
        if depth_kept:
            self._last_depth_us = t
        if color_wanted:
            self._last_color_us = t
        self.decisions.append(decision)
        return decision


def image_sharpness(gray_small: np.ndarray) -> float:
    """Variance of the Laplacian; only comparable between frames that see similar content."""
    import cv2

    return float(cv2.Laplacian(gray_small, cv2.CV_64F).var())


def select_color_frames(
    keyframe_times_us: list[int],
    color_times_us: list[int],
    color_blur_px: list[float],
    policy: KeyframePolicy | None = None,
    sharpness: list[float] | None = None,
) -> list[int]:
    """For each color keyframe, the best color frame within ``color_window_s``.

    With ``sharpness``, the locally sharpest frame wins: neighbours in a short window see nearly the same
    content, so measured sharpness compares motion blur fairly. Frames with extreme predicted blur are skipped.
    """
    p = policy or KeyframePolicy()
    order = np.argsort(color_times_us)
    times = [int(color_times_us[i]) for i in order]
    blur = [float(color_blur_px[i]) for i in order]
    sharp = [float(sharpness[i]) for i in order] if sharpness is not None else None
    window = int(p.color_window_s * 1e6)
    chosen: set[int] = set()
    for t in keyframe_times_us:
        lo, hi = bisect.bisect_left(times, t - window), bisect.bisect_right(times, t + window)
        candidates = [i for i in range(lo, hi) if blur[i] <= p.max_blur_px]
        if not candidates:
            continue
        if sharp is not None:
            best = max(candidates, key=lambda i: (sharp[i], -abs(times[i] - t)))
        else:
            best = min(candidates, key=lambda i: (blur[i], abs(times[i] - t)))
        chosen.add(times[best])
    return sorted(chosen)
