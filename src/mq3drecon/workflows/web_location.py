"""Build a walkable web location from a recorded session (``mq3drecon build-location``).

Output (``specs/WEB_LOCATION_EXPORT.md``)::

    <workspace>/locations/<id>/location.json, visual.glb, collider.glb, thumbnail.jpg
    <workspace>/locations/index.json
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

LOCATION_FORMAT = "mq3drecon-web-location"
LOCATION_VERSION = 1


@dataclass(frozen=True)
class LocationResult:
    location_dir: Path
    location: dict


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", text.lower()).strip("-") or "location"


def _thumbnail(project_dir: Path, out_path: Path, size: int = 512) -> bool:
    frames = sorted((project_dir / "left_camera_mruk_rgba").glob("*.jpg"))
    if not frames:
        return False
    image = cv2.imread(str(frames[len(frames) // 2]))
    if image is None:
        return False
    scale = size / max(image.shape[:2])
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return bool(cv2.imwrite(str(out_path), image, [cv2.IMWRITE_JPEG_QUALITY, 85]))


def _train_splats(seed_points, seed_colors, views, out: Path, enabled: bool, iterations: int) -> tuple[str | None, dict]:
    """Train and write ``scene.splat`` if gsplat is available; the location is still valid without it.

    Seeds are only vertices some color camera observed: never-seen geometry would yield untrained gray splats.
    """
    if not enabled or not views:
        return None, {}
    try:
        from mq3drecon.processing.web_location.msvc_env import prepare_gsplat_build

        prepare_gsplat_build()
        from mq3drecon.processing.web_location.splat_io import write_splat
        from mq3drecon.processing.web_location.splat_train import SplatSettings, train_splats
    except ImportError as exc:
        logger.info("Splat training skipped (%s); install torch + gsplat to enable it", exc)
        return None, {}
    result = train_splats(views, seed_points, seed_colors, SplatSettings(iterations=iterations))
    write_splat(out / "scene.splat", result.means, result.scales, result.quats_wxyz, result.rgb, result.opacity)
    return "scene.splat", {
        "splats": int(len(result.means)), "splat_iterations": iterations, "splat_seconds": round(result.seconds, 1),
        "splat_train_psnr": round(result.train_psnr, 2), "splat_holdout_psnr": round(result.holdout_psnr, 2),
    }


def run_serve_locations(
    *, workspace: Path | None = None, port: int = 8080, adb: bool = True, adb_path: Path | None = None, open_headset: bool = False,
) -> None:
    """Serve the gallery and viewer on localhost until Ctrl+C; expose it to the headset with adb reverse."""
    import time

    from mq3drecon.web_viewer.server import adb_reverse_and_open, make_server, serve_forever_in_thread
    from mq3drecon.workflows.live_reconstruction import default_workspace

    locations_dir = (Path(workspace) if workspace is not None else default_workspace()) / "locations"
    server = make_server(locations_dir, port=port)
    serve_forever_in_thread(server)
    print(f"[serve] http://localhost:{port}/  (locations: {locations_dir})")
    if adb:
        serials = adb_reverse_and_open(port, adb_path=adb_path, open_browser=open_headset)
        print(f"[serve] headset: adb reverse tcp:{port} on {serials or 'no device'}"
              + ("; opened the headset browser" if open_headset and serials else "; open http://localhost:%d in the Quest browser" % port))
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


def update_index(locations_dir: Path) -> list[dict]:
    entries = []
    for path in sorted(Path(locations_dir).glob("*/location.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        entries.append({
            "id": data["id"], "name": data["name"], "kind": data["kind"], "created_utc": data["created_utc"],
            "thumbnail": f"{data['id']}/{data['assets']['thumbnail']}" if data["assets"].get("thumbnail") else None,
            "triangles": data["stats"]["visual_triangles"],
        })
    entries.sort(key=lambda e: e["created_utc"], reverse=True)
    (Path(locations_dir) / "index.json").write_text(json.dumps({"locations": entries}, indent=2), encoding="utf-8")
    return entries


def build_location(
    project_dir: Path,
    *,
    workspace: Path | None = None,
    name: str | None = None,
    kind: str = "indoor",
    visual_triangles: int = 400_000,
    collider_triangles: int = 100_000,
    textured: bool = True,
    textured_triangles: int = 150_000,
    texture_size: int = 4096,
    splats: bool = True,
    splat_iterations: int = 15000,
    device: str = "cuda:0",
) -> LocationResult:
    from mq3drecon.processing.web_location.mesh_build import fuse_session
    from mq3drecon.processing.web_location.mesh_export import (
        decimate, estimate_floor_y, remove_debris, spawn_from_camera, to_open3d, write_glb,
    )
    from mq3drecon.workflows.live_reconstruction import default_workspace

    if kind not in ("indoor", "outdoor"):
        raise ValueError("kind must be 'indoor' or 'outdoor'")
    project_dir = Path(project_dir)
    workspace = Path(workspace) if workspace is not None else default_workspace()
    name = name or project_dir.name
    location_id = _slug(name)
    out = workspace / "locations" / location_id
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Location already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)

    built = fuse_session(project_dir, voxel_size_m=0.02 if kind == "indoor" else 0.04, device=device)
    from dataclasses import replace as dc_replace

    from mq3drecon.processing.web_location.color_fill import fill_uncolored

    # Unseen areas (e.g. an unscanned ceiling) get a smooth blend of the surrounding seen colors, not flat gray.
    colors = fill_uncolored(built.mesh.vertices, built.mesh.triangles, built.mesh.colors, built.mesh.color_valid, harmonic=True)
    mesh = remove_debris(to_open3d(dc_replace(built.mesh, colors=colors)))
    collider = decimate(mesh, collider_triangles)
    floor_y, floor_source = estimate_floor_y(decimate(mesh, visual_triangles), built.trajectory[:, 1])
    spawn = spawn_from_camera(built.first_view.camera_to_world, floor_y)
    texture_stats: dict = {"textured": False}
    views = []
    if textured:
        from mq3drecon.processing.web_location.textured_visual import build_textured_visual, load_bake_views

        views = load_bake_views(project_dir)
    baked = None
    if views:
        from mq3drecon.processing.web_location.glb import write_glb as write_binary_glb
        from mq3drecon.processing.web_location.uv_atlas import make_manifold

        visual, manifold_stats = make_manifold(decimate(mesh, textured_triangles))
        try:
            baked = build_textured_visual(
                np.asarray(visual.vertices), np.asarray(visual.triangles), np.asarray(visual.vertex_colors), views,
                texture_size=texture_size, device=device,
            )
        except Exception as error:  # keep the location (vertex colors + splats) rather than losing the build
            print(f"[build-location] texture bake failed, falling back to vertex colors: {error}", flush=True)
            texture_stats = {"textured": False, "texture_error": str(error)[:300]}
    if baked is not None:
        write_binary_glb(out / "visual.glb", baked.vertices, baked.triangles, uvs=baked.uvs, texture_jpeg=baked.texture_jpeg)
        texture_stats = {"textured": True, "texture_size": texture_size, "bake_views": baked.views,
                         "textured_fraction": round(baked.textured_fraction, 4),
                         "manifold_triangles_removed": manifold_stats["triangles_removed"]}
    else:
        visual = decimate(mesh, visual_triangles)
        write_glb(visual, out / "visual.glb", with_colors=True)
    write_glb(collider, out / "collider.glb", with_colors=False)
    seen = built.mesh.color_valid
    splat_file, splat_stats = _train_splats(built.mesh.vertices[seen], built.mesh.colors[seen], views, out, splats, splat_iterations)
    has_thumbnail = _thumbnail(project_dir, out / "thumbnail.jpg")
    vertices = np.asarray(visual.vertices)
    location = {
        "format": LOCATION_FORMAT,
        "version": LOCATION_VERSION,
        "id": location_id,
        "name": name,
        "kind": kind,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source_session": str(project_dir),
        "units": "meters",
        "frame": "glTF: Y-up, right-handed",
        "floor_y": floor_y,
        "floor_source": floor_source,
        "spawn": {"position": list(spawn.position), "yaw_deg": spawn.yaw_deg},
        "bounds": {"min": vertices.min(axis=0).tolist(), "max": vertices.max(axis=0).tolist()},
        "assets": {"visual": "visual.glb", "collider": "collider.glb", "thumbnail": "thumbnail.jpg" if has_thumbnail else None, "splat": splat_file},
        "stats": {
            "visual_triangles": len(visual.triangles), "collider_triangles": len(collider.triangles),
            "voxel_size_m": built.voxel_size_m, "frames": built.frames, **texture_stats, **splat_stats,
        },
    }
    (out / "location.json").write_text(json.dumps(location, indent=2), encoding="utf-8")
    update_index(workspace / "locations")
    return LocationResult(location_dir=out, location=location)
