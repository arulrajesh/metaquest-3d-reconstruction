"""Laptop-to-headset feedback messages: keyframe notices, the surface mesh, and live status.

The surface mesh is sent in the headset's **Unity world frame** (Y-up, left-handed):
positions are converted from MQ3DRecon's Open3D world frame by negating Z, and
triangle winding is reversed to match the handedness change.

``SURFACE_MESH`` payload, all little-endian:

* positions: ``vertex_count`` × 3 × uint16, multiples of ``step_m`` above ``origin``;
* colors: ``vertex_count`` × 4 × uint8 (R, G, B, confidence 0–255);
* indices: ``triangle_count`` × 3 × (uint16 if ``index_bits`` is 16, else uint32).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mq3drecon.streaming.messages import Pose

OPEN3D_TO_UNITY = np.array([1.0, 1.0, -1.0], dtype=np.float64)


@dataclass(frozen=True)
class KeyframeNotice:
    timestamp_us: int
    depth_kept: bool
    color_wanted: bool
    reason: str
    novelty: float
    info_gain: float
    pose: Pose  # Unity camera-to-world pose of the depth view

    def to_header(self) -> dict[str, Any]:
        return {
            "timestamp_us": self.timestamp_us, "depth_kept": self.depth_kept, "color_wanted": self.color_wanted,
            "reason": self.reason, "novelty": self.novelty, "info_gain": self.info_gain, "pose": self.pose.to_header(),
        }

    @classmethod
    def from_header(cls, h: dict[str, Any]) -> "KeyframeNotice":
        return cls(
            timestamp_us=int(h["timestamp_us"]), depth_kept=bool(h["depth_kept"]), color_wanted=bool(h["color_wanted"]),
            reason=str(h.get("reason", "")), novelty=float(h.get("novelty", 0.0)), info_gain=float(h.get("info_gain", 0.0)),
            pose=Pose.from_header(h["pose"]),
        )


@dataclass(frozen=True)
class SurfaceMesh:
    sequence: int
    segment: int
    origin: tuple[float, float, float]
    step_m: float
    vertex_count: int
    triangle_count: int
    index_bits: int
    data: bytes = field(repr=False)

    def to_header(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence, "segment": self.segment, "origin": list(self.origin), "step_m": self.step_m,
            "vertex_count": self.vertex_count, "triangle_count": self.triangle_count, "index_bits": self.index_bits,
        }

    @classmethod
    def from_header(cls, h: dict[str, Any], payload: bytes) -> "SurfaceMesh":
        mesh = cls(
            sequence=int(h["sequence"]), segment=int(h.get("segment", 0)), origin=tuple(float(v) for v in h["origin"]),  # type: ignore[arg-type]
            step_m=float(h["step_m"]), vertex_count=int(h["vertex_count"]), triangle_count=int(h["triangle_count"]),
            index_bits=int(h.get("index_bits", 32)), data=payload,
        )
        if len(payload) != mesh.expected_payload_bytes:
            raise ValueError(f"SURFACE_MESH payload is {len(payload)} bytes; expected {mesh.expected_payload_bytes}")
        return mesh

    @property
    def expected_payload_bytes(self) -> int:
        return self.vertex_count * (6 + 4) + self.triangle_count * 3 * (2 if self.index_bits == 16 else 4)

    def unpack(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (Unity positions (N, 3) float32, RGBA uint8 (N, 4), triangles (M, 3) int64)."""
        n, m = self.vertex_count, self.triangle_count
        q = np.frombuffer(self.data, dtype="<u2", count=n * 3).reshape(n, 3)
        positions = (np.asarray(self.origin, np.float64) + q * self.step_m).astype(np.float32)
        rgba = np.frombuffer(self.data, dtype=np.uint8, count=n * 4, offset=n * 6).reshape(n, 4)
        dtype = "<u2" if self.index_bits == 16 else "<u4"
        triangles = np.frombuffer(self.data, dtype=dtype, count=m * 3, offset=n * 10).reshape(m, 3).astype(np.int64)
        return positions, rgba, triangles


@dataclass(frozen=True)
class LiveStatusUpdate:
    values: dict[str, float | int | str] = field(default_factory=dict)


def pack_surface_mesh(
    vertices_open3d: np.ndarray,
    triangles: np.ndarray,
    rgb: np.ndarray,
    confidence: np.ndarray,
    *,
    sequence: int,
    segment: int = 0,
    step_m: float = 0.005,
) -> SurfaceMesh:
    """Quantize a mesh (Open3D world) into a ``SURFACE_MESH`` message in the Unity world frame."""
    unity = np.asarray(vertices_open3d, np.float64) * OPEN3D_TO_UNITY
    tris = np.asarray(triangles, np.int64)[:, [0, 2, 1]]  # reflection flips winding
    n, m = len(unity), len(tris)
    origin = unity.min(axis=0) if n else np.zeros(3)
    q = np.round((unity - origin) / step_m)
    if n and q.max() > 65535:
        raise ValueError(f"Mesh extent exceeds the uint16 range at step {step_m} m")
    rgba = np.concatenate([
        np.clip(np.round(np.asarray(rgb, np.float64) * 255.0), 0, 255).astype(np.uint8).reshape(n, 3),
        np.clip(np.round(np.asarray(confidence, np.float64) * 255.0), 0, 255).astype(np.uint8).reshape(n, 1),
    ], axis=1)
    index_bits = 16 if n <= 65535 else 32
    indices = tris.astype("<u2" if index_bits == 16 else "<u4")
    data = b"".join([q.astype("<u2").tobytes(), rgba.tobytes(), indices.tobytes()])
    return SurfaceMesh(sequence, segment, tuple(float(v) for v in origin), step_m, n, m, index_bits, data)  # type: ignore[arg-type]
