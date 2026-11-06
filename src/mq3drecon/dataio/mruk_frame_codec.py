"""Decoding of MRUK color frame files (``.rgba`` raw buffers and ``.jpg`` stream recordings).

Orientation rules (``specs/RECORDING_FORMATS.md``):

* ``.rgba`` files are raw RGBA32 buffers in bottom-up row order and are flipped
  vertically on load.
* ``.jpg`` files are standard top-down JPEG images and are not flipped.

Both decode to top-down ``(height, width, 4)`` uint8 RGBA arrays.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

RAW_SUFFIX = ".rgba"
JPEG_SUFFIXES = (".jpg", ".jpeg")
MRUK_FRAME_SUFFIXES = (RAW_SUFFIX, *JPEG_SUFFIXES)


def is_mruk_frame_file(path: Path) -> bool:
    return path.suffix.lower() in MRUK_FRAME_SUFFIXES


def validate_mruk_frame_file(path: Path, width: int, height: int) -> None:
    """Cheap structural validation without decoding JPEG content."""
    if not path.exists():
        raise FileNotFoundError(f"MRUK color frame listed in metadata does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == RAW_SUFFIX:
        expected = int(width) * int(height) * 4
        actual = path.stat().st_size
        if actual != expected:
            raise ValueError(f"Invalid MRUK RGBA file size for {path}: expected {expected} bytes, got {actual}")
        return
    if suffix in JPEG_SUFFIXES:
        if path.stat().st_size == 0:
            raise ValueError(f"Empty MRUK JPEG frame: {path}")
        return
    raise ValueError(f"Unsupported MRUK color frame format: {path}")


def decode_jpeg_rgba(data: bytes | np.ndarray, width: int | None = None, height: int | None = None) -> np.ndarray:
    buffer = np.frombuffer(data, dtype=np.uint8) if isinstance(data, (bytes, bytearray, memoryview)) else data
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("JPEG data could not be decoded")
    if width is not None and height is not None and bgr.shape[:2] != (int(height), int(width)):
        raise ValueError(f"JPEG frame is {bgr.shape[1]}x{bgr.shape[0]}; metadata says {width}x{height}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)


def read_mruk_frame_rgba(path: Path, width: int, height: int) -> np.ndarray:
    """Read a ``.rgba`` or ``.jpg`` MRUK frame as a top-down RGBA image."""
    validate_mruk_frame_file(path, width, height)
    if path.suffix.lower() == RAW_SUFFIX:
        rgba = np.fromfile(path, dtype=np.uint8).reshape(int(height), int(width), 4)
        return np.ascontiguousarray(np.flipud(rgba))
    return decode_jpeg_rgba(np.fromfile(path, dtype=np.uint8), width, height)
