"""Synthetic capture generation for tests and demos (no headset required)."""

from mq3drecon.synthetic.room_capture import (
    SyntheticCaptureSpec,
    head_pose,
    render,
    scene_color,
    surface_distance,
    write_synthetic_capture,
)

__all__ = [
    "SyntheticCaptureSpec",
    "head_pose",
    "render",
    "scene_color",
    "surface_distance",
    "write_synthetic_capture",
]
