"""Quest NDC depth frame files: ``<timestamp_ms>.raw`` (float32) or ``<timestamp_ms>.png`` (16-bit).

``.png`` stores ``round(clip(ndc, 0, 1) * 65535)`` as a single-channel 16-bit
PNG (lossless at that precision: about 0.6 mm at 2 m with a 0.1 m near plane).
Both decode to the raw float32 NDC buffer, rows as stored. See
``specs/RECORDING_FORMATS.md``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

U16_SCALE = 65535.0


def depth_png_path(raw_path: Path) -> Path:
    return Path(raw_path).with_suffix(".png")


def read_ndc_depth(raw_path: Path, width: int, height: int) -> np.ndarray | None:
    """Read ``.raw`` if present, else the ``.png`` sibling. Returns ``None`` when neither exists."""
    raw_path = Path(raw_path)
    if raw_path.exists():
        return np.fromfile(raw_path, dtype="<f4").reshape((int(height), int(width)))
    png = depth_png_path(raw_path)
    if not png.exists():
        return None
    values = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    if values is None or values.dtype != np.uint16 or values.shape != (int(height), int(width)):
        raise ValueError(f"Invalid 16-bit depth PNG {png}: expected {width}x{height} uint16")
    return (values.astype(np.float32) / np.float32(U16_SCALE)).astype(np.float32)


def write_ndc_depth_png(path: Path, ndc: np.ndarray) -> None:
    values = np.round(np.clip(np.nan_to_num(np.asarray(ndc, np.float32), nan=0.0), 0.0, 1.0) * U16_SCALE).astype(np.uint16)
    if not cv2.imwrite(str(path), values, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f"Failed to write depth PNG: {path}")
