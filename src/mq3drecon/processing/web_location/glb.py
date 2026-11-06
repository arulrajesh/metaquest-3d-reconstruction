"""Minimal binary glTF 2.0 (GLB) writer and reader for triangle meshes (numpy only).

Writes a single mesh with ``POSITION`` (float32), optional ``COLOR_0`` (normalized
uint8 RGBA, 4-byte aligned as the spec requires for vertex attributes) and uint32
indices, all in one BIN chunk. Unlit vertex-colored meshes need no normals.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_FLOAT, _UBYTE, _UINT = 5126, 5121, 5125
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963


def _pad(data: bytes, fill: bytes) -> bytes:
    return data + fill * ((4 - len(data) % 4) % 4)


def write_glb(path: Path, vertices: np.ndarray, triangles: np.ndarray, colors: np.ndarray | None = None) -> int:
    """Write a GLB. ``colors`` are RGB in [0, 1] per vertex. Returns the file size in bytes."""
    positions = np.ascontiguousarray(vertices, dtype="<f4")
    indices = np.ascontiguousarray(triangles, dtype="<u4").reshape(-1)
    chunks = [positions.tobytes()]
    views = [{"buffer": 0, "byteOffset": 0, "byteLength": positions.nbytes, "target": _ARRAY_BUFFER}]
    attributes = {"POSITION": 0}
    accessors = [{
        "bufferView": 0, "componentType": _FLOAT, "count": len(positions), "type": "VEC3",
        "min": positions.min(axis=0).tolist(), "max": positions.max(axis=0).tolist(),
    }]
    offset = positions.nbytes
    if colors is not None:
        rgba = np.concatenate([
            np.clip(np.round(np.asarray(colors) * 255.0), 0, 255).astype(np.uint8).reshape(-1, 3),
            np.full((len(positions), 1), 255, np.uint8),
        ], axis=1)
        chunks.append(rgba.tobytes())
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": rgba.nbytes, "target": _ARRAY_BUFFER})
        attributes["COLOR_0"] = len(accessors)
        accessors.append({"bufferView": len(views) - 1, "componentType": _UBYTE, "normalized": True, "count": len(rgba), "type": "VEC4"})
        offset += rgba.nbytes
    chunks.append(indices.tobytes())
    views.append({"buffer": 0, "byteOffset": offset, "byteLength": indices.nbytes, "target": _ELEMENT_ARRAY_BUFFER})
    accessors.append({"bufferView": len(views) - 1, "componentType": _UINT, "count": len(indices), "type": "SCALAR"})
    binary = _pad(b"".join(chunks), b"\x00")
    document = {
        "asset": {"version": "2.0", "generator": "mq3drecon"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": attributes, "indices": len(accessors) - 1, "mode": 4}]}],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(binary)}],
    }
    json_chunk = _pad(json.dumps(document, separators=(",", ":")).encode("utf-8"), b" ")
    total = 12 + 8 + len(json_chunk) + 8 + len(binary)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(json_chunk), b"JSON"))
        f.write(json_chunk)
        f.write(struct.pack("<I4s", len(binary), b"BIN\x00"))
        f.write(binary)
    return total


def read_glb(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Read back a GLB written by :func:`write_glb` (vertices, triangles, RGB colors or None)."""
    data = Path(path).read_bytes()
    magic, version, total = struct.unpack("<4sII", data[:12])
    if magic != b"glTF" or version != 2 or total != len(data):
        raise ValueError(f"Not a valid GLB: {path}")
    json_len = struct.unpack("<I", data[12:16])[0]
    document = json.loads(data[20:20 + json_len])
    bin_start = 20 + json_len + 8
    accessors, views = document["accessors"], document["bufferViews"]
    primitive = document["meshes"][0]["primitives"][0]

    def array(index: int, dtype: str, width: int) -> np.ndarray:
        accessor = accessors[index]
        view = views[accessor["bufferView"]]
        start = bin_start + view["byteOffset"]
        return np.frombuffer(data, dtype=dtype, count=accessor["count"] * width, offset=start).reshape(accessor["count"], width)

    vertices = array(primitive["attributes"]["POSITION"], "<f4", 3)
    triangles = array(primitive["indices"], "<u4", 1).reshape(-1, 3)
    colors = None
    if "COLOR_0" in primitive["attributes"]:
        colors = array(primitive["attributes"]["COLOR_0"], np.uint8, 4)[:, :3].astype(np.float32) / 255.0
    return vertices, triangles, colors
