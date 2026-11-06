"""Configuration for live streaming reconstruction (``mq3drecon live``). Lightweight: no Open3D/Warp."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

MODES = ("indoor", "outdoor")


@dataclass(frozen=True)
class LiveServerConfig:
    host: str = "0.0.0.0"
    port: int = 9870
    idle_timeout_s: float = 10.0
    discovery: bool = True
    discovery_port: int = 9871
    adb: bool = True
    adb_path: str | None = None
    push_endpoint: bool = True
    app_package: str = "com.t34400.QuestRealityCapture"


@dataclass(frozen=True)
class LiveVolumeConfig:
    device: str = "cuda:0"
    voxel_size_m: float = 0.04
    extent_m: tuple[float, float, float] = (12.0, 5.0, 12.0)
    vertical_offset_m: float = -0.3
    trunc_voxels: float = 4.0


@dataclass(frozen=True)
class LiveIntegrationConfig:
    sides: tuple[str, ...] = ("left",)
    depth_min_m: float = 0.3
    depth_max_m: float = 5.0
    color_tolerance_ms: float = 150.0
    queue_size: int = 4
    exclude_torso: bool = True
    torso_radius_m: float = 0.35
    jump_distance_m: float = 0.3
    jump_speed_mps: float = 2.0


@dataclass(frozen=True)
class LivePreviewConfig:
    viewer: bool = True
    mesh_interval_s: float = 1.0
    max_sigma_voxels: float = 0.5
    """Keep surface points whose Kalman posterior std-dev is at most this fraction of a voxel."""


@dataclass(frozen=True)
class LiveBuildConfig:
    auto_build: bool = True
    collider_triangles: int = 100_000
    visual_triangles: int = 400_000


@dataclass(frozen=True)
class LiveConfig:
    mode: str = "indoor"
    server: LiveServerConfig = field(default_factory=LiveServerConfig)
    volume: LiveVolumeConfig = field(default_factory=LiveVolumeConfig)
    integration: LiveIntegrationConfig = field(default_factory=LiveIntegrationConfig)
    preview: LivePreviewConfig = field(default_factory=LivePreviewConfig)
    build: LiveBuildConfig = field(default_factory=LiveBuildConfig)


def preset(mode: str) -> LiveConfig:
    """Built-in defaults per capture mode."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    base = LiveConfig(mode=mode)
    if mode == "outdoor":
        # Larger, coarser volume. Native depth still stops at ~5 m; the far field comes from splats.
        base = replace(base, volume=replace(base.volume, voxel_size_m=0.06, extent_m=(48.0, 8.0, 48.0), vertical_offset_m=-0.8))
    return base


def _merge(instance: Any, overrides: dict[str, Any], path: str) -> Any:
    known = {f.name: f for f in fields(instance)}
    unknown = sorted(set(overrides) - set(known))
    if unknown:
        raise ValueError(f"Unknown key(s) in live config section '{path}': {unknown}")
    values: dict[str, Any] = {}
    for name, value in overrides.items():
        current = getattr(instance, name)
        if hasattr(current, "__dataclass_fields__"):
            if not isinstance(value, dict):
                raise ValueError(f"'{path}.{name}' must be a mapping")
            values[name] = _merge(current, value, f"{path}.{name}")
        elif isinstance(current, tuple):
            values[name] = tuple(value)
        else:
            values[name] = type(current)(value) if current is not None and value is not None else value
    return replace(instance, **values)


def live_config_from_dict(data: dict[str, Any] | None, mode: str | None = None) -> LiveConfig:
    """Build a config from a ``live:`` mapping. ``mode`` (argument, else ``data['mode']``) selects the preset."""
    data = dict(data or {})
    chosen = mode or data.pop("mode", None) or "indoor"
    data.pop("mode", None)
    return _merge(preset(chosen), data, "live")


def load_live_config(path: Path | None = None, mode: str | None = None) -> LiveConfig:
    if path is None:
        return preset(mode or "indoor")
    with open(path, "r", encoding="utf-8") as f:
        document = yaml.safe_load(f) or {}
    section = document.get("live", {})
    if not isinstance(section, dict):
        raise ValueError(f"Missing or invalid 'live' section in config: {path}")
    return live_config_from_dict(section, mode)
