"""Matching streamed depth frames to the nearest color frame (lightweight)."""

from __future__ import annotations

import bisect
import threading
from collections import deque

from mq3drecon.models.side import Side
from mq3drecon.streaming.messages import ColorFrame


class ColorHistory:
    """Thread-safe recent color frames per side, for nearest-in-time lookup."""

    def __init__(self, max_frames_per_side: int = 32):
        self._frames: dict[Side, deque[ColorFrame]] = {side: deque(maxlen=max_frames_per_side) for side in Side}
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            for frames in self._frames.values():
                frames.clear()

    def add(self, frame: ColorFrame) -> None:
        with self._lock:
            frames = self._frames[frame.side]
            if frames and frame.timestamp_us < frames[-1].timestamp_us:
                ordered = sorted([*frames, frame], key=lambda f: f.timestamp_us)[-frames.maxlen:]  # type: ignore[index]
                frames.clear()
                frames.extend(ordered)
            else:
                frames.append(frame)

    def nearest(self, side: Side, timestamp_us: int, tolerance_us: int) -> ColorFrame | None:
        with self._lock:
            frames = list(self._frames[side])
        if not frames:
            return None
        times = [f.timestamp_us for f in frames]
        index = bisect.bisect_left(times, timestamp_us)
        candidates = [frames[i] for i in (index - 1, index) if 0 <= i < len(frames)]
        best = min(candidates, key=lambda f: abs(f.timestamp_us - timestamp_us))
        return best if abs(best.timestamp_us - timestamp_us) <= tolerance_us else None
