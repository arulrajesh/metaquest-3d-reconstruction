"""MRUK passthrough-camera intrinsics in the pixels of the delivered image.

MRUK reports ``focalLength`` and ``principalPoint`` in **sensor** pixels
(``sensorResolution``, 1728x1728 on Quest 3). The image an app receives
(``resolution`` / ``currentResolution``, e.g. 1280x1280) is the sensor image
center-cropped to its own aspect ratio and then scaled uniformly; this is the
mapping in Meta's ``PassthroughCameraUtils``. Using the sensor values with the
delivered image directly makes the focal length 35% too long and puts the
principal point about 225 px off-center (found on a real capture, 2026-11-06:
splat training PSNR went from 22.5 to 28.6 dB once fixed).

Files without ``sensorResolution`` are treated as already in image pixels.
Pure Python (no numpy).
"""

from __future__ import annotations

from typing import Any, Mapping


def image_pinhole(intrinsics: Mapping[str, Any], width: int | None = None, height: int | None = None) -> tuple[float, float, float, float]:
    """``(fx, fy, cx, cy)`` for a ``width`` x ``height`` image (default: the file's ``resolution``)."""
    resolution = intrinsics.get("resolution") or intrinsics.get("currentResolution") or {}
    width = float(width if width is not None else resolution.get("width"))
    height = float(height if height is not None else resolution.get("height"))
    reference = intrinsics.get("sensorResolution") or resolution or {}
    sensor_w = float(reference.get("width") or width)
    sensor_h = float(reference.get("height") or height)
    scale_x, scale_y = width / sensor_w, height / sensor_h
    largest = max(scale_x, scale_y)
    crop_w, crop_h = sensor_w * scale_x / largest, sensor_h * scale_y / largest
    crop_x, crop_y = (sensor_w - crop_w) * 0.5, (sensor_h - crop_h) * 0.5
    sx, sy = width / crop_w, height / crop_h
    focal, principal = intrinsics["focalLength"], intrinsics["principalPoint"]
    return (
        float(focal["x"]) * sx,
        float(focal["y"]) * sy,
        (float(principal["x"]) - crop_x) * sx,
        (float(principal["y"]) - crop_y) * sy,
    )
