"""Encoding and decoding of COLOR and DEPTH message payloads."""

from __future__ import annotations

import cv2
import numpy as np

from mq3drecon.dataio.mruk_frame_codec import decode_jpeg_rgba
from mq3drecon.streaming.messages import ColorFrame, DepthFrame

_U16_SCALE = 65535.0


def decode_color_rgba(frame: ColorFrame) -> np.ndarray:
    """Decode a COLOR payload to a top-down ``(H, W, 4)`` uint8 RGBA image."""
    if frame.encoding == "jpeg":
        rgba = decode_jpeg_rgba(frame.data, frame.width, frame.height)
    else:
        rgba = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 4)
    if frame.row_order == "bottom_up":
        rgba = np.flipud(rgba)
    return np.ascontiguousarray(rgba)


def encode_color_jpeg(rgba_top_down: np.ndarray, quality: int = 90) -> bytes:
    bgr = cv2.cvtColor(np.ascontiguousarray(rgba_top_down[:, :, :4]), cv2.COLOR_RGBA2BGR)
    ok, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buffer.tobytes()


def decode_depth_ndc(frame: DepthFrame) -> np.ndarray:
    """Decode a DEPTH payload to the raw ``(H, W)`` float32 NDC depth buffer (as stored in ``.raw`` files)."""
    shape = (frame.height, frame.width)
    if frame.encoding == "f32":
        return np.frombuffer(frame.data, dtype="<f4").reshape(shape).astype(np.float32, copy=True)
    if frame.encoding == "f16":
        return np.frombuffer(frame.data, dtype="<f2").reshape(shape).astype(np.float32)
    values = np.frombuffer(frame.data, dtype="<u2").reshape(shape)
    return (values.astype(np.float32) / np.float32(_U16_SCALE)).astype(np.float32)


def encode_depth_ndc(ndc: np.ndarray, encoding: str = "u16_ndc") -> bytes:
    ndc = np.asarray(ndc, dtype=np.float32)
    if encoding == "f32":
        return ndc.astype("<f4").tobytes()
    if encoding == "f16":
        return ndc.astype("<f2").tobytes()
    if encoding == "u16_ndc":
        clipped = np.clip(np.nan_to_num(ndc, nan=0.0), 0.0, 1.0)
        return np.round(clipped * _U16_SCALE).astype("<u2").tobytes()
    raise ValueError(f"Unsupported depth encoding: {encoding}")
