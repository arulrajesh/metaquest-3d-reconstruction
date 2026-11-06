"""Minimal binary glTF 2.0 (GLB) writer and reader for triangle meshes (numpy only).

A mesh is written as one primitive in a single BIN chunk with:

* ``POSITION``: float32;
* optional ``COLOR_0``: vertex colors converted from sRGB to **linear** (glTF requires
  linear vertex colors), normalized uint16 RGBA (4-byte aligned, no 8-bit banding);
* optional ``TEXCOORD_0`` plus an embedded JPEG base color texture, with an unlit material
  (``KHR_materials_unlit``: captured lighting is already baked into the colors);
* uint32 indices.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_FLOAT, _USHORT, _UINT = 5126, 5123, 5125
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963
_LINEAR, _LINEAR_MIPMAP_LINEAR, _CLAMP = 9729, 9987, 33071


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, np.float64), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def _pad(data: bytes, fill: bytes) -> bytes:
    return data + fill * ((4 - len(data) % 4) % 4)


def write_glb(
    path: Path,
    vertices: np.ndarray,
    triangles: np.ndarray,
    colors: np.ndarray | None = None,
    *,
    uvs: np.ndarray | None = None,
    texture_jpeg: bytes | None = None,
) -> int:
    """Write a GLB. ``colors``: per-vertex sRGB in [0, 1]. ``uvs`` + ``texture_jpeg``: textured, unlit. Returns bytes."""
    positions = np.ascontiguousarray(vertices, dtype="<f4")
    sections: list[bytes] = []
    views: list[dict] = []
    accessors: list[dict] = []
    attributes: dict[str, int] = {}

    def add_view(data: bytes, target: int | None) -> int:
        offset = sum(len(s) for s in sections)
        sections.append(_pad(data, b"\x00"))
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        views.append(view)
        return len(views) - 1

    def add_accessor(view: int, component: int, count: int, kind: str, **extra) -> int:
        accessors.append({"bufferView": view, "componentType": component, "count": count, "type": kind, **extra})
        return len(accessors) - 1

    attributes["POSITION"] = add_accessor(
        add_view(positions.tobytes(), _ARRAY_BUFFER), _FLOAT, len(positions), "VEC3",
        min=positions.min(axis=0).tolist(), max=positions.max(axis=0).tolist(),
    )
    if colors is not None:
        linear = srgb_to_linear(np.asarray(colors).reshape(-1, 3))
        rgba = np.concatenate([linear, np.ones((len(linear), 1))], axis=1)
        data = np.round(rgba * 65535.0).astype("<u2").tobytes()
        attributes["COLOR_0"] = add_accessor(add_view(data, _ARRAY_BUFFER), _USHORT, len(rgba), "VEC4", normalized=True)
    if uvs is not None:
        tex = np.ascontiguousarray(uvs, dtype="<f4").reshape(-1, 2)
        attributes["TEXCOORD_0"] = add_accessor(add_view(tex.tobytes(), _ARRAY_BUFFER), _FLOAT, len(tex), "VEC2")
    indices = np.ascontiguousarray(triangles, dtype="<u4").reshape(-1)
    index_accessor = add_accessor(add_view(indices.tobytes(), _ELEMENT_ARRAY_BUFFER), _UINT, len(indices), "SCALAR")

    primitive: dict = {"attributes": attributes, "indices": index_accessor, "mode": 4}
    document: dict = {
        "asset": {"version": "2.0", "generator": "mq3drecon"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [primitive]}],
    }
    if uvs is not None and texture_jpeg is not None:
        image_view = add_view(texture_jpeg, None)
        document["images"] = [{"bufferView": image_view, "mimeType": "image/jpeg"}]
        document["samplers"] = [{"magFilter": _LINEAR, "minFilter": _LINEAR_MIPMAP_LINEAR, "wrapS": _CLAMP, "wrapT": _CLAMP}]
        document["textures"] = [{"source": 0, "sampler": 0}]
        document["materials"] = [{
            "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 1.0},
            "extensions": {"KHR_materials_unlit": {}},
            "doubleSided": True,
        }]
        document["extensionsUsed"] = ["KHR_materials_unlit"]
        primitive["material"] = 0
    binary = b"".join(sections)
    document["accessors"] = accessors
    document["bufferViews"] = views
    document["buffers"] = [{"byteLength": len(binary)}]
    json_chunk = _pad(json.dumps(document, separators=(",", ":")).encode("utf-8"), b" ")
    total = 12 + 8 + len(json_chunk) + 8 + len(binary)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(json_chunk), b"JSON"))
        f.write(json_chunk)
        f.write(struct.pack("<I4s", len(binary), b"BIN\x00"))
        f.write(binary)
    return total


def read_glb(path: Path) -> dict:
    """Read a GLB written by :func:`write_glb`.

    Returns ``vertices``, ``triangles``, ``colors`` (sRGB) or None, ``uvs`` or None, and ``texture_jpeg`` or None.
    """
    data = Path(path).read_bytes()
    magic, version, total = struct.unpack("<4sII", data[:12])
    if magic != b"glTF" or version != 2 or total != len(data):
        raise ValueError(f"Not a valid GLB: {path}")
    json_len = struct.unpack("<I", data[12:16])[0]
    document = json.loads(data[20:20 + json_len])
    bin_start = 20 + json_len + 8
    accessors, views = document["accessors"], document["bufferViews"]
    primitive = document["meshes"][0]["primitives"][0]
    attributes = primitive["attributes"]

    def array(index: int, dtype: str, width: int) -> np.ndarray:
        accessor = accessors[index]
        start = bin_start + views[accessor["bufferView"]]["byteOffset"]
        return np.frombuffer(data, dtype=dtype, count=accessor["count"] * width, offset=start).reshape(accessor["count"], width)

    result: dict = {
        "vertices": array(attributes["POSITION"], "<f4", 3),
        "triangles": array(primitive["indices"], "<u4", 1).reshape(-1, 3),
        "colors": None,
        "uvs": None,
        "texture_jpeg": None,
    }
    if "COLOR_0" in attributes:
        rgba = array(attributes["COLOR_0"], "<u2", 4).astype(np.float64) / 65535.0
        result["colors"] = linear_to_srgb(rgba[:, :3]).astype(np.float32)
    if "TEXCOORD_0" in attributes:
        result["uvs"] = array(attributes["TEXCOORD_0"], "<f4", 2)
    if document.get("images"):
        view = views[document["images"][0]["bufferView"]]
        start = bin_start + view["byteOffset"]
        result["texture_jpeg"] = data[start:start + view["byteLength"]]
    return result
