"""Wire framing for the MQ3DRecon live streaming protocol.

Every message on the wire is::

    magic   4 bytes  b"MQLS"
    version u16      protocol version
    type    u16      message type id
    hlen    u32      JSON header length in bytes
    plen    u32      binary payload length in bytes
    header  hlen     UTF-8 JSON object
    payload plen     raw bytes

All integers are little-endian. See ``specs/LIVE_STREAMING.md``.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any, BinaryIO, Protocol

from mq3drecon.errors import MQ3DReconError

MAGIC = b"MQLS"
PROTOCOL_VERSION = 1
PREFIX = struct.Struct("<4sHHII")
MAX_HEADER_BYTES = 64 * 1024
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


class StreamProtocolError(MQ3DReconError):
    """Raised when bytes on the wire do not form a valid protocol message."""


class _Readable(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...


@dataclass(frozen=True)
class RawMessage:
    """A framed message before type-specific decoding."""

    version: int
    type_id: int
    header: dict[str, Any]
    payload: bytes


def encode_frame(type_id: int, header: dict[str, Any], payload: bytes = b"") -> bytes:
    header_bytes = json.dumps(header, separators=(",", ":"), allow_nan=True).encode("utf-8")
    if len(header_bytes) > MAX_HEADER_BYTES:
        raise StreamProtocolError(f"Header too large: {len(header_bytes)} bytes")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise StreamProtocolError(f"Payload too large: {len(payload)} bytes")
    prefix = PREFIX.pack(MAGIC, PROTOCOL_VERSION, int(type_id), len(header_bytes), len(payload))
    return b"".join((prefix, header_bytes, payload))


def _read_exact(stream: _Readable | BinaryIO, size: int, *, allow_eof: bool) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            if allow_eof and remaining == size:
                return None
            raise StreamProtocolError(f"Connection closed mid-message ({size - remaining}/{size} bytes read)")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(stream: _Readable | BinaryIO) -> RawMessage | None:
    """Read one framed message. Returns ``None`` on a clean end of stream between messages."""
    prefix = _read_exact(stream, PREFIX.size, allow_eof=True)
    if prefix is None:
        return None

    magic, version, type_id, header_len, payload_len = PREFIX.unpack(prefix)
    if magic != MAGIC:
        raise StreamProtocolError(f"Bad magic {magic!r}; expected {MAGIC!r}")
    if version != PROTOCOL_VERSION:
        raise StreamProtocolError(
            f"Unsupported protocol version {version}; this receiver speaks version {PROTOCOL_VERSION}"
        )
    if header_len > MAX_HEADER_BYTES:
        raise StreamProtocolError(f"Header length {header_len} exceeds limit {MAX_HEADER_BYTES}")
    if payload_len > MAX_PAYLOAD_BYTES:
        raise StreamProtocolError(f"Payload length {payload_len} exceeds limit {MAX_PAYLOAD_BYTES}")

    header_bytes = _read_exact(stream, header_len, allow_eof=False) or b""
    payload = _read_exact(stream, payload_len, allow_eof=False) if payload_len else b""

    try:
        header = json.loads(header_bytes.decode("utf-8")) if header_bytes else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StreamProtocolError(f"Header is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise StreamProtocolError("Header must be a JSON object")

    return RawMessage(version=version, type_id=type_id, header=header, payload=payload or b"")
