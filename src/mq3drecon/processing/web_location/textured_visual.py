"""Textured visual mesh for a web location: decimate, unwrap, bake from the session's color keyframes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mq3drecon.processing.live.keyframes import image_sharpness
from mq3drecon.processing.live.native_depth import color_to_image
from mq3drecon.processing.web_location.texture_bake import BakeSettings, BakeView, bake_texture
from mq3drecon.processing.web_location.uv_atlas import unwrap
from mq3drecon.streaming.messages import ColorFrame, SessionStart
from mq3drecon.streaming.replay import ReplayOptions, iter_project_messages


@dataclass(frozen=True)
class TexturedVisual:
    vertices: np.ndarray
    triangles: np.ndarray
    uvs: np.ndarray
    texture_jpeg: bytes
    texture_size: int
    views: int
    textured_fraction: float


def load_bake_views(project_dir: Path, *, max_views: int = 400) -> list[BakeView]:
    """All color frames of the session (both eyes), thinned evenly in time if there are more than ``max_views``."""
    intrinsics: dict = {}
    frames: list[ColorFrame] = []
    for _, message in iter_project_messages(project_dir, ReplayOptions(depth_sides=())):
        if isinstance(message, SessionStart):
            intrinsics = message.intrinsics
        elif isinstance(message, ColorFrame) and message.side in intrinsics:
            frames.append(message)
    if len(frames) > max_views:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, max_views).round().astype(int)]
    views = []
    for frame in frames:
        image = color_to_image(frame, intrinsics[frame.side])
        gray = cv2.cvtColor(cv2.resize(image.rgb, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        views.append(BakeView(image.rgb, image.view.intrinsic, image.view.world_to_camera, image_sharpness(gray)))
    return views


def build_textured_visual(
    vertices: np.ndarray,
    triangles: np.ndarray,
    vertex_colors: np.ndarray,
    views: list[BakeView],
    *,
    texture_size: int = 4096,
    jpeg_quality: int = 90,
    device: str = "cuda:0",
) -> TexturedVisual:
    uv = unwrap(vertices, triangles, size=texture_size)
    result = bake_texture(
        uv.vertices, uv.triangles, uv.uvs, np.asarray(vertex_colors)[uv.source_vertex], views,
        BakeSettings(size=texture_size), device=device,
    )
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(result.texture, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    if not ok:
        raise ValueError("Texture JPEG encoding failed")
    return TexturedVisual(
        vertices=uv.vertices, triangles=uv.triangles, uvs=uv.uvs, texture_jpeg=jpeg.tobytes(),
        texture_size=texture_size, views=len(views), textured_fraction=result.textured_fraction,
    )
