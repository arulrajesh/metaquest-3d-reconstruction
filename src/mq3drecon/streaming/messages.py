"""Typed live streaming messages and their header encoding.

Header field names and units match the on-disk recording formats so a received
session can be written without translation (see ``docs/DATA_FORMAT.md``):
color timestamps are microseconds (``timestamp_us_realtime``), depth timestamps
are milliseconds (``timestamp_ms``), and poses are Unity camera-to-world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Union

from mq3drecon.models.side import Side
from mq3drecon.streaming.framing import RawMessage, StreamProtocolError, encode_frame


class MessageType(IntEnum):
    HELLO = 1
    SESSION_START = 2
    COLOR = 3
    DEPTH = 4
    HEARTBEAT = 5
    SESSION_END = 6
    BYE = 7
    # Laptop -> headset feedback (mq3drecon.streaming.feedback)
    KEYFRAME = 20
    SURFACE_MESH = 21
    STATUS = 22


COLOR_ENCODINGS = ("jpeg", "rgba")
ROW_ORDERS = ("top_down", "bottom_up")
DEPTH_ENCODINGS = {"f32": 4, "f16": 2, "u16_ndc": 2}
"""Bytes per pixel. ``u16_ndc`` is ``round(ndc_depth * 65535)``, the default on Wi-Fi."""


@dataclass(frozen=True)
class Pose:
    """Unity camera-to-world pose. Rotation is a quaternion ordered (x, y, z, w)."""

    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]

    def to_header(self) -> dict[str, list[float]]:
        return {"position": [float(v) for v in self.position], "rotation": [float(v) for v in self.rotation]}

    @classmethod
    def from_header(cls, data: Any) -> "Pose":
        try:
            position = tuple(float(v) for v in data["position"])
            rotation = tuple(float(v) for v in data["rotation"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StreamProtocolError(f"Invalid pose: {data!r}") from exc
        if len(position) != 3 or len(rotation) != 4:
            raise StreamProtocolError(f"Pose needs 3 position and 4 rotation values: {data!r}")
        return cls(position=position, rotation=rotation)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Hello:
    device: str
    app_version: str


@dataclass(frozen=True)
class SessionStart:
    """Starts (or resumes, after a reconnect) a recording session."""

    session_name: str
    intrinsics: dict[Side, dict[str, Any]]
    """Per-side MRUK intrinsics objects, same schema as ``*_camera_mruk_intrinsics.json``."""
    resume: bool = False


@dataclass(frozen=True)
class ColorFrame:
    side: Side
    frame_index: int
    timestamp_us: int
    unix_time_ms: int
    width: int
    height: int
    encoding: str
    row_order: str
    pose: Pose
    pose_ok: bool
    data: bytes = field(repr=False)


@dataclass(frozen=True)
class DepthFrame:
    """One eye of a Quest Depth API frame. ``data`` holds the raw NDC depth buffer."""

    side: Side
    timestamp_ms: int
    ovr_timestamp: float
    width: int
    height: int
    encoding: str
    create_pose: Pose
    fov_left: float
    fov_right: float
    fov_top: float
    fov_down: float
    near_z: float
    far_z: float
    data: bytes = field(repr=False)


@dataclass(frozen=True)
class Heartbeat:
    time_ms: int
    queued: dict[str, int] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionEnd:
    reason: str = "user_stop"
    dropped: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Bye:
    reason: str = "closing"


Message = Union[Hello, SessionStart, ColorFrame, DepthFrame, Heartbeat, SessionEnd, Bye]


def _side_name(side: Side) -> str:
    return side.name.lower()


def _parse_side(value: Any) -> Side:
    try:
        return Side[str(value).upper()]
    except KeyError as exc:
        raise StreamProtocolError(f"Invalid side: {value!r}") from exc


def _counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(k): int(v) for k, v in value.items()}


def encode_message(message: Message) -> bytes:
    if isinstance(message, Hello):
        return encode_frame(MessageType.HELLO, {"device": message.device, "app_version": message.app_version})
    if isinstance(message, SessionStart):
        header = {
            "session_name": message.session_name,
            "resume": message.resume,
            "intrinsics": {_side_name(side): obj for side, obj in message.intrinsics.items()},
        }
        return encode_frame(MessageType.SESSION_START, header)
    if isinstance(message, ColorFrame):
        header = {
            "side": _side_name(message.side),
            "frame_index": message.frame_index,
            "timestamp_us_realtime": message.timestamp_us,
            "unix_time_ms": message.unix_time_ms,
            "width": message.width,
            "height": message.height,
            "encoding": message.encoding,
            "row_order": message.row_order,
            "pose": message.pose.to_header(),
            "pose_ok": message.pose_ok,
        }
        return encode_frame(MessageType.COLOR, header, message.data)
    if isinstance(message, DepthFrame):
        header = {
            "side": _side_name(message.side),
            "timestamp_ms": message.timestamp_ms,
            "ovr_timestamp": message.ovr_timestamp,
            "width": message.width,
            "height": message.height,
            "encoding": message.encoding,
            "create_pose": message.create_pose.to_header(),
            "fov": {
                "left": message.fov_left,
                "right": message.fov_right,
                "top": message.fov_top,
                "down": message.fov_down,
            },
            "near_z": message.near_z,
            "far_z": message.far_z,
        }
        return encode_frame(MessageType.DEPTH, header, message.data)
    if isinstance(message, Heartbeat):
        header = {"time_ms": message.time_ms, "queued": message.queued, "dropped": message.dropped}
        return encode_frame(MessageType.HEARTBEAT, header)
    if isinstance(message, SessionEnd):
        return encode_frame(MessageType.SESSION_END, {"reason": message.reason, "dropped": message.dropped})
    if isinstance(message, Bye):
        return encode_frame(MessageType.BYE, {"reason": message.reason})
    from mq3drecon.streaming.feedback import KeyframeNotice, LiveStatusUpdate, SurfaceMesh

    if isinstance(message, KeyframeNotice):
        return encode_frame(MessageType.KEYFRAME, message.to_header())
    if isinstance(message, SurfaceMesh):
        return encode_frame(MessageType.SURFACE_MESH, message.to_header(), message.data)
    if isinstance(message, LiveStatusUpdate):
        return encode_frame(MessageType.STATUS, dict(message.values))
    raise TypeError(f"Unsupported message type: {type(message).__name__}")


def decode_message(raw: RawMessage) -> Message:
    try:
        message_type = MessageType(raw.type_id)
    except ValueError as exc:
        raise StreamProtocolError(f"Unknown message type id {raw.type_id}") from exc

    h = raw.header
    try:
        if message_type == MessageType.HELLO:
            return Hello(device=str(h.get("device", "")), app_version=str(h.get("app_version", "")))
        if message_type == MessageType.SESSION_START:
            intrinsics = {_parse_side(k): dict(v) for k, v in dict(h.get("intrinsics", {})).items()}
            return SessionStart(session_name=str(h["session_name"]), intrinsics=intrinsics, resume=bool(h.get("resume", False)))
        if message_type == MessageType.COLOR:
            return _decode_color(h, raw.payload)
        if message_type == MessageType.DEPTH:
            return _decode_depth(h, raw.payload)
        if message_type == MessageType.HEARTBEAT:
            return Heartbeat(time_ms=int(h.get("time_ms", 0)), queued=_counts(h.get("queued")), dropped=_counts(h.get("dropped")))
        if message_type == MessageType.SESSION_END:
            return SessionEnd(reason=str(h.get("reason", "user_stop")), dropped=_counts(h.get("dropped")))
        if message_type == MessageType.BYE:
            return Bye(reason=str(h.get("reason", "closing")))
        from mq3drecon.streaming.feedback import KeyframeNotice, LiveStatusUpdate, SurfaceMesh

        if message_type == MessageType.KEYFRAME:
            return KeyframeNotice.from_header(h)  # type: ignore[return-value]
        if message_type == MessageType.SURFACE_MESH:
            return SurfaceMesh.from_header(h, raw.payload)  # type: ignore[return-value]
        return LiveStatusUpdate(values=dict(h))  # type: ignore[return-value]
    except (KeyError, TypeError, ValueError) as exc:
        raise StreamProtocolError(f"Invalid {message_type.name} header: {exc}") from exc


def _decode_color(h: dict[str, Any], payload: bytes) -> ColorFrame:
    encoding = str(h["encoding"])
    row_order = str(h.get("row_order", "bottom_up" if encoding == "rgba" else "top_down"))
    width, height = int(h["width"]), int(h["height"])
    if encoding not in COLOR_ENCODINGS:
        raise StreamProtocolError(f"Unsupported color encoding {encoding!r}")
    if row_order not in ROW_ORDERS:
        raise StreamProtocolError(f"Unsupported row order {row_order!r}")
    if encoding == "rgba" and len(payload) != width * height * 4:
        raise StreamProtocolError(f"RGBA payload is {len(payload)} bytes; expected {width * height * 4}")
    if not payload:
        raise StreamProtocolError("COLOR message has an empty payload")
    return ColorFrame(
        side=_parse_side(h["side"]),
        frame_index=int(h.get("frame_index", 0)),
        timestamp_us=int(h["timestamp_us_realtime"]),
        unix_time_ms=int(h.get("unix_time_ms", int(h["timestamp_us_realtime"]) // 1000)),
        width=width,
        height=height,
        encoding=encoding,
        row_order=row_order,
        pose=Pose.from_header(h["pose"]),
        pose_ok=bool(h.get("pose_ok", True)),
        data=payload,
    )


def _decode_depth(h: dict[str, Any], payload: bytes) -> DepthFrame:
    encoding = str(h["encoding"])
    width, height = int(h["width"]), int(h["height"])
    if encoding not in DEPTH_ENCODINGS:
        raise StreamProtocolError(f"Unsupported depth encoding {encoding!r}")
    expected = width * height * DEPTH_ENCODINGS[encoding]
    if len(payload) != expected:
        raise StreamProtocolError(f"Depth payload is {len(payload)} bytes; expected {expected}")
    fov = h["fov"]
    return DepthFrame(
        side=_parse_side(h["side"]),
        timestamp_ms=int(h["timestamp_ms"]),
        ovr_timestamp=float(h.get("ovr_timestamp", 0.0)),
        width=width,
        height=height,
        encoding=encoding,
        create_pose=Pose.from_header(h["create_pose"]),
        fov_left=float(fov["left"]),
        fov_right=float(fov["right"]),
        fov_top=float(fov["top"]),
        fov_down=float(fov["down"]),
        near_z=float(h["near_z"]),
        far_z=_float_or_inf(h["far_z"]),
        data=payload,
    )


def _float_or_inf(value: Any) -> float:
    """Depth far planes are often infinite; accept ``null``, ``"inf"`` or ``Infinity``."""
    if value is None:
        return float("inf")
    return float(value)
