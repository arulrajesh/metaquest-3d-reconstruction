"""Writes a streamed session to disk in the MRUK project layout (the laptop master copy)."""

from __future__ import annotations

import bisect
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any

import numpy as np

from mq3drecon.config.project_path_config import ProjectPathConfig
from mq3drecon.dataio.depth_frame_codec import depth_png_path, write_ndc_depth_png
from mq3drecon.dataio.qrc_csv_headers import COLOR_METADATA_HEADER, DEPTH_DESCRIPTOR_HEADER, STEREO_PAIR_HEADER
from mq3drecon.models.side import Side
from mq3drecon.streaming.messages import ColorFrame, DepthFrame, SessionStart
from mq3drecon.streaming.payloads import decode_color_rgba, decode_depth_ndc, encode_color_jpeg

LIVE_SESSION_SUMMARY_JSON = "live_session.json"


@dataclass
class RecordedSessionSummary:
    project_dir: str
    session_name: str
    color_frames: dict[str, int] = field(default_factory=dict)
    depth_frames: dict[str, int] = field(default_factory=dict)
    duplicate_frames: int = 0
    stereo_pairs: int = 0
    end_reason: str = ""
    headset_dropped: dict[str, int] = field(default_factory=dict)


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _float_text(value: float) -> str:
    return "inf" if np.isinf(value) else repr(float(value))


@dataclass(frozen=True)
class RecordingOptions:
    depth_format: str = "png16"
    """``png16`` (16-bit PNG of NDC, compact) or ``raw`` (float32, as the headset writes)."""
    depth_sides: tuple[Side, ...] = (Side.LEFT,)
    """Depth eyes to record. Live and build paths use the left eye; the right eye is redundant."""


ROOM_MODEL_JSON = "room_model.json"
MARKERS_JSON = "markers.json"


class SessionRecorder:
    """Appends frames to an MRUK project directory. The directory must be new or empty."""

    def __init__(self, project_dir: Path, start: SessionStart, options: RecordingOptions | None = None):
        self.options = options or RecordingOptions()
        if self.options.depth_format not in ("png16", "raw"):
            raise ValueError(f"Unsupported depth format: {self.options.depth_format}")
        self.project_dir = Path(project_dir)
        if self.project_dir.exists() and any(self.project_dir.iterdir()):
            raise FileExistsError(f"Session directory already exists and is not empty: {self.project_dir}")
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self._paths = ProjectPathConfig(self.project_dir)
        self._color_writers: dict[Side, tuple[IO[str], Any]] = {}
        self._depth_writers: dict[Side, tuple[IO[str], Any]] = {}
        self._color_index: dict[Side, list[tuple[int, int, str]]] = {Side.LEFT: [], Side.RIGHT: []}
        self._seen: set[tuple[str, Side, int]] = set()
        self.summary = RecordedSessionSummary(project_dir=str(self.project_dir), session_name=start.session_name)
        self._closed = False

        self._write_json(self._paths.image.get_session_info_json_path(), {"sessionFormatVersion": 2, "captureBackend": "MRUK"})
        for side, intrinsics in start.intrinsics.items():
            self._write_json(self._paths.image.get_mruk_intrinsics_json_path(side), intrinsics)

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _writer(self, table: dict[Side, tuple[IO[str], Any]], side: Side, path: Path, header: tuple[str, ...]):
        if side not in table:
            handle = open(path, "w", newline="", encoding="utf-8")
            writer = csv.writer(handle)
            writer.writerow(header)
            table[side] = (handle, writer)
        return table[side]

    def _is_duplicate(self, kind: str, side: Side, timestamp: int) -> bool:
        key = (kind, side, timestamp)
        if key in self._seen:
            self.summary.duplicate_frames += 1
            return True
        self._seen.add(key)
        return False

    def write_color(self, frame: ColorFrame) -> None:
        if self._closed or self._is_duplicate("color", frame.side, frame.timestamp_us):
            return
        frame_dir = self._paths.image.get_mruk_rgba_dir(frame.side)
        frame_dir.mkdir(parents=True, exist_ok=True)
        if frame.encoding == "jpeg":
            file_name, fmt = f"{frame.timestamp_us}.jpg", "JPEG"
            data = frame.data if frame.row_order == "top_down" else encode_color_jpeg(decode_color_rgba(frame), 95)
        else:
            file_name, fmt = f"{frame.timestamp_us}.rgba", "RGBA32"
            data = np.ascontiguousarray(np.flipud(decode_color_rgba(frame))).tobytes()  # .rgba is bottom-up
        (frame_dir / file_name).write_bytes(data)

        handle, writer = self._writer(
            self._color_writers, frame.side, self._paths.image.get_mruk_frame_metadata_csv_path(frame.side), COLOR_METADATA_HEADER
        )
        pixels = frame.width * frame.height
        writer.writerow([
            frame.frame_index, file_name, frame.unix_time_ms, frame.timestamp_us, frame.width, frame.height, fmt,
            frame.side.name.capitalize(), *frame.pose.position, *frame.pose.rotation, _bool(frame.pose_ok),
            "true", "true", frame.width, frame.height, pixels, pixels, pixels * 4, "true", "true", "",
        ])
        handle.flush()
        self._color_index[frame.side].append((frame.timestamp_us, frame.frame_index, file_name))
        self.summary.color_frames[frame.side.name.lower()] = self.summary.color_frames.get(frame.side.name.lower(), 0) + 1

    def write_depth(self, frame: DepthFrame) -> None:
        if self._closed or frame.side not in self.options.depth_sides:
            return
        if self._is_duplicate("depth", frame.side, frame.timestamp_ms):
            return
        path = self._paths.depth.get_depth_map_path(side=frame.side, timestamp=frame.timestamp_ms)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.options.depth_format == "png16":
            write_ndc_depth_png(depth_png_path(path), decode_depth_ndc(frame))
        else:
            decode_depth_ndc(frame).astype("<f4").tofile(path)

        handle, writer = self._writer(
            self._depth_writers, frame.side, self._paths.depth.get_depth_descriptor_path(frame.side), DEPTH_DESCRIPTOR_HEADER
        )
        writer.writerow([
            frame.timestamp_ms, repr(frame.ovr_timestamp), *frame.create_pose.position, *frame.create_pose.rotation,
            frame.fov_left, frame.fov_right, frame.fov_top, frame.fov_down,
            _float_text(frame.near_z), _float_text(frame.far_z), frame.width, frame.height,
        ])
        handle.flush()
        self.summary.depth_frames[frame.side.name.lower()] = self.summary.depth_frames.get(frame.side.name.lower(), 0) + 1

    def write_room_model(self, room: dict[str, Any]) -> None:
        """Latest Space Setup room from the headset (replaced when the operator edits the room)."""
        if not self._closed:
            self._write_json(self.project_dir / ROOM_MODEL_JSON, room)

    def add_marker(self, kind: str, data: dict[str, Any]) -> None:
        """Operator markers, kept in ``markers.json`` (e.g. "detail" windows that compaction keeps in full)."""
        if self._closed:
            return
        path = self.project_dir / MARKERS_JSON
        markers = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        markers.append({"kind": kind, **data})
        path.write_text(json.dumps(markers, indent=2), encoding="utf-8")

    def finalize(self, reason: str = "user_stop", headset_dropped: dict[str, int] | None = None) -> RecordedSessionSummary:
        if self._closed:
            return self.summary
        self._closed = True
        for handle, _writer in [*self._color_writers.values(), *self._depth_writers.values()]:
            handle.close()
        self.summary.stereo_pairs = self._write_stereo_pairs()
        self.summary.end_reason = reason
        self.summary.headset_dropped = dict(headset_dropped or {})
        self._write_json(self.project_dir / LIVE_SESSION_SUMMARY_JSON, asdict(self.summary))
        return self.summary

    def _write_stereo_pairs(self) -> int:
        left = sorted(self._color_index[Side.LEFT])
        right = sorted(self._color_index[Side.RIGHT])
        if not left or not right:
            return 0
        pairs = nearest_timestamp_pairs([t for t, _, _ in left], [t for t, _, _ in right])
        with open(self._paths.image.get_mruk_stereo_pairs_csv_path(), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(STEREO_PAIR_HEADER)
            for pair_index, (li, ri) in enumerate(pairs, start=1):
                (lt, lf, ln), (rt, rf, rn) = left[li], right[ri]
                writer.writerow([pair_index, ln, rn, lt, rt, rt - lt, lf, rf])
        return len(pairs)


def nearest_timestamp_pairs(left: list[int], right: list[int]) -> list[tuple[int, int]]:
    """Greedy one-to-one pairing by smallest time difference (same rule as the headset app), sorted by left time.

    Candidates are each frame's nearest neighbours on the other side, so the
    cost is O(n log n) instead of the headset's O(n^3) exhaustive search.
    """
    candidates: set[tuple[int, int, int]] = set()
    for li, t in enumerate(left):
        j = bisect.bisect_left(right, t)
        for ri in (j - 1, j):
            if 0 <= ri < len(right):
                candidates.add((abs(right[ri] - t), li, ri))
    for ri, t in enumerate(right):
        j = bisect.bisect_left(left, t)
        for li in (j - 1, j):
            if 0 <= li < len(left):
                candidates.add((abs(left[li] - t), li, ri))
    used_left: set[int] = set()
    used_right: set[int] = set()
    pairs = []
    for _delta, li, ri in sorted(candidates):
        if li not in used_left and ri not in used_right:
            used_left.add(li)
            used_right.add(ri)
            pairs.append((li, ri))
    return sorted(pairs)
