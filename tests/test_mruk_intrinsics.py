"""MRUK intrinsics are in sensor pixels; images are a center-cropped, scaled sensor image."""

import pytest

from mq3drecon.dataio.mruk_intrinsics import image_pinhole
from mq3drecon.processing.live.native_depth import color_intrinsic

# Values from a real Quest 3 capture (20261106_175535_2, left camera).
REAL = {
    "focalLength": {"x": 1175.736083984375, "y": 1175.736083984375},
    "principalPoint": {"x": 867.7169799804688, "y": 862.8403930664062},
    "resolution": {"width": 1280, "height": 1280},
    "sensorResolution": {"width": 1728, "height": 1728},
    "currentResolution": {"width": 1280, "height": 1280},
}


def test_square_image_is_a_uniform_scale_of_the_sensor():
    fx, fy, cx, cy = image_pinhole(REAL)
    s = 1280 / 1728
    assert fx == pytest.approx(1175.736 * s, abs=1e-3) and fy == pytest.approx(fx)
    assert cx == pytest.approx(867.717 * s, abs=1e-3)  # 642.75: near the 1280 image's center, not at 867
    assert cy == pytest.approx(862.840 * s, abs=1e-3)


def test_other_aspect_ratio_is_a_center_crop_then_scale():
    # 1280x960 from a 1728x1728 sensor: crop 1728x1296 (216 px off top and bottom), scale 1280/1728.
    fx, fy, cx, cy = image_pinhole(REAL, 1280, 960)
    s = 1280 / 1728
    assert fx == pytest.approx(1175.736 * s, abs=1e-3) and fy == pytest.approx(fx)
    assert cx == pytest.approx(867.717 * s, abs=1e-3)
    assert cy == pytest.approx((862.840 - 216.0) * s, abs=1e-3)


def test_without_sensor_resolution_values_are_already_image_pixels():
    legacy = {k: v for k, v in REAL.items() if k not in ("sensorResolution", "currentResolution")}
    assert image_pinhole(legacy) == pytest.approx((1175.736, 1175.736, 867.717, 862.840), abs=1e-3)
    # A half-size streamed frame still scales from the file's resolution.
    assert image_pinhole(legacy, 640, 640) == pytest.approx((587.868, 587.868, 433.858, 431.420), abs=1e-3)


def test_color_intrinsic_matrix_uses_the_mapping():
    k = color_intrinsic(REAL, 1280, 1280)
    fx, fy, cx, cy = image_pinhole(REAL)
    assert (k[0, 0], k[1, 1], k[0, 2], k[1, 2]) == pytest.approx((fx, fy, cx, cy))
