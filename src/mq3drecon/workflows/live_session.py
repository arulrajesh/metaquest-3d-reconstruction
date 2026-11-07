"""Session control for live capture: receiver messages to recorder + reconstructor (lightweight).

The reconstructor is injected, so this module needs neither Open3D nor Warp.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mq3drecon.models.side import Side
from mq3drecon.streaming.messages import (
    Bye, ColorFrame, DepthFrame, Heartbeat, Hello, Marker, Message, RoomModelUpdate, SessionEnd, SessionStart,
)
from mq3drecon.streaming.session_recorder import RecordedSessionSummary, RecordingOptions, SessionRecorder

logger = logging.getLogger(__name__)


class Reconstructor(Protocol):
    def begin_session(self, intrinsics: dict[Side, dict[str, Any]]) -> None: ...

    def add_color(self, frame: ColorFrame) -> None: ...

    def add_depth(self, frame: DepthFrame) -> None: ...

    def end_session(self, timeout_s: float = 30.0) -> Any: ...


@dataclass(frozen=True)
class CompletedSession:
    summary: RecordedSessionSummary
    final_snapshot: Any
    project_dir: Path


@dataclass(frozen=True)
class LiveStatus:
    connected_peer: str | None
    device: str | None
    session_name: str | None
    recording_dir: str | None
    color_frames: int
    depth_frames: int
    headset_dropped: dict[str, int]
    completed_sessions: int


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned or "session"


def unique_session_dir(sessions_dir: Path, name: str) -> Path:
    base = Path(sessions_dir) / _safe_name(name)
    candidate, counter = base, 2
    while candidate.exists() and any(candidate.iterdir()):
        candidate = base.with_name(f"{base.name}_{counter}")
        counter += 1
    return candidate


class LiveSessionController:
    """Implements ``LiveStreamHandler``. Disconnects keep a session open for ``resume_grace_s``."""

    def __init__(
        self,
        sessions_dir: Path,
        reconstructor: Reconstructor | None = None,
        on_session_complete: Callable[[CompletedSession], None] | None = None,
        *,
        resume_grace_s: float = 30.0,
        recording_options: RecordingOptions | None = None,
    ):
        self._recording_options = recording_options
        self.sessions_dir = Path(sessions_dir)
        self._reconstructor = reconstructor
        self._on_complete = on_session_complete
        self._grace_s = resume_grace_s
        self._lock = threading.RLock()
        self._recorder: SessionRecorder | None = None
        self._session_name: str | None = None
        self._grace_timer: threading.Timer | None = None
        self._peer: str | None = None
        self._device: str | None = None
        self._dropped: dict[str, int] = {}
        self._room: dict | None = None  # latest Space Setup room; the headset may send it before recording starts
        self.completed: list[CompletedSession] = []

    # LiveStreamHandler ---------------------------------------------------
    def on_connect(self, peer: str) -> None:
        with self._lock:
            self._peer = peer
        logger.info("Headset connected from %s", peer)

    def on_message(self, message: Message) -> None:
        with self._lock:
            if isinstance(message, Hello):
                self._device = message.device
            elif isinstance(message, SessionStart):
                self._start(message)
            elif isinstance(message, ColorFrame) and self._recorder is not None:
                self._recorder.write_color(message)
                if self._reconstructor is not None:
                    self._reconstructor.add_color(message)
            elif isinstance(message, DepthFrame) and self._recorder is not None:
                self._recorder.write_depth(message)
                if self._reconstructor is not None:
                    self._reconstructor.add_depth(message)
            elif isinstance(message, Heartbeat):
                self._dropped = dict(message.dropped)
            elif isinstance(message, SessionEnd):
                self._finish(message.reason, message.dropped)
            elif isinstance(message, RoomModelUpdate):
                self._room = dict(message.room)
                if self._recorder is not None:
                    self._recorder.write_room_model(self._room)
                logger.info("Room model: %d anchors", len(self._room.get("anchors", [])))
            elif isinstance(message, Marker) and self._recorder is not None:
                self._recorder.add_marker(message.kind, message.data)
            elif isinstance(message, Bye):
                logger.info("Headset said goodbye: %s", message.reason)

    def on_disconnect(self, peer: str, error: BaseException | None) -> None:
        with self._lock:
            if self._peer == peer:
                self._peer = None
            if self._recorder is not None and self._grace_timer is None:
                logger.warning("Headset disconnected mid-session (%s); waiting %.0f s for it to resume", error, self._grace_s)
                self._grace_timer = threading.Timer(self._grace_s, self._finish, args=("disconnected", None))
                self._grace_timer.daemon = True
                self._grace_timer.start()

    # Session lifecycle ---------------------------------------------------
    def _start(self, message: SessionStart) -> None:
        if message.resume and self._recorder is not None and self._session_name == message.session_name:
            self._cancel_grace()
            logger.info("Resumed session %s", message.session_name)
            return
        if self._recorder is not None:
            self._finish("superseded", None)
        project_dir = unique_session_dir(self.sessions_dir, message.session_name)
        self._recorder = SessionRecorder(project_dir, message, self._recording_options)
        if self._room is not None:
            self._recorder.write_room_model(self._room)
        self._session_name = message.session_name
        self._dropped = {}
        if self._reconstructor is not None:
            self._reconstructor.begin_session(message.intrinsics)
        logger.info("Recording session %s to %s", message.session_name, project_dir)

    def _finish(self, reason: str, dropped: dict[str, int] | None) -> None:
        with self._lock:
            self._cancel_grace()
            recorder, self._recorder, self._session_name = self._recorder, None, None
            if recorder is None:
                return
            summary = recorder.finalize(reason, dropped if dropped is not None else self._dropped)
            snapshot = self._reconstructor.end_session() if self._reconstructor is not None else None
            completed = CompletedSession(summary=summary, final_snapshot=snapshot, project_dir=recorder.project_dir)
            self.completed.append(completed)
        logger.info("Session finished (%s): %s", reason, summary.project_dir)
        if self._on_complete is not None:
            self._on_complete(completed)

    def _cancel_grace(self) -> None:
        if self._grace_timer is not None:
            self._grace_timer.cancel()
            self._grace_timer = None

    def finish_open_session(self, reason: str = "shutdown") -> None:
        self._finish(reason, None)

    def status(self) -> LiveStatus:
        with self._lock:
            summary = self._recorder.summary if self._recorder is not None else None
            return LiveStatus(
                connected_peer=self._peer,
                device=self._device,
                session_name=self._session_name,
                recording_dir=summary.project_dir if summary else None,
                color_frames=sum(summary.color_frames.values()) if summary else 0,
                depth_frames=sum(summary.depth_frames.values()) if summary else 0,
                headset_dropped=dict(self._dropped),
                completed_sessions=len(self.completed),
            )
