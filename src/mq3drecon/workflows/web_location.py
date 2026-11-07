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


# Half-resolution uint8 views, ~2.2 MB each at 1728² capture. Fidelity captures (5-15 min, up to 4 color frames
# per second per eye after compaction) reach thousands; beyond SplatSettings.gpu_image_budget_gb they are kept
# in CPU memory (6000 views ~ 13 GB of the laptop's 32 GB).
SPLAT_MAX_VIEWS = 6000


def splat_iterations_for(train_views: int) -> int:
    """Default schedule: 15k iterations (tuned on ~500 views), growing with the view count so that each view
    is still visited ~10 times; capped at 60k (~50 min on the laptop GPU)."""
    return int(min(60_000, max(15_000, 10 * train_views)))


def load_splat_views(project_dir: Path):
    """Every color frame of the session (up to ``SPLAT_MAX_VIEWS``) at half resolution, for splat training."""
    from mq3drecon.processing.web_location.textured_visual import load_bake_views

    return load_bake_views(project_dir, max_views=SPLAT_MAX_VIEWS, downscale=2)


def _train_splats_spirula(project_dir: Path, seed_points, seed_colors, out: Path, spirula_path, iterations: int) -> tuple[str | None, dict]:
    """Train with the external Spirula Studio binary; writes ``scene.ply`` (3DGS PLY with SH3, which Spark renders).

    Returns ``(None, {})`` when the binary is missing or fails, so the caller falls back to gsplat.
    """
    import shutil
    import tempfile

    from mq3drecon.processing.web_location.spirula import (
        find_spirula, holdout_psnr, read_splat_ply, train_with_spirula, write_nerfstudio_dataset,
    )
    from mq3drecon.processing.web_location.textured_visual import iter_bake_views

    exe = find_spirula(spirula_path)
    if exe is None:
        print("[build-location] Spirula not found (set spirula_path or MQ3DRECON_SPIRULA); using gsplat", flush=True)
        return None, {}
    try:
        with tempfile.TemporaryDirectory(prefix="mq3drecon-spirula-") as work:
            dataset = Path(work) / "dataset"
            split = write_nerfstudio_dataset(iter_bake_views(project_dir, max_views=SPLAT_MAX_VIEWS), seed_points, seed_colors, dataset)
            run = train_with_spirula(exe, dataset, Path(work) / "train", iterations=iterations)
            shutil.copy2(run.splat_ply, out / "scene.ply")
            shutil.copy2(run.log, out / "spirula_train.log")
            seconds, version = run.seconds, run.version
    except Exception as error:  # keep the location: fall back to our own trainer
        print(f"[build-location] Spirula training failed, using gsplat instead: {error}", flush=True)
        return None, {}
    splats = read_splat_ply(out / "scene.ply")
    try:
        psnr = holdout_psnr(splats, iter_bake_views(project_dir, max_views=SPLAT_MAX_VIEWS, downscale=2))
    except ImportError:
        psnr = float("nan")
    return "scene.ply", {
        "splats": int(len(splats["means"])), "splat_trainer": version, "splat_format": "ply-sh%d" % (int(round(np.sqrt(splats["sh"].shape[1]))) - 1),
        "splat_iterations": iterations, "splat_seconds": round(seconds, 1), "splat_holdout_psnr": round(psnr, 2),
        "splat_views": split["train_views"] + split["holdout_views"],
    }


def _train_splats(seed_points, seed_colors, views, out: Path, enabled: bool, iterations: int | None) -> tuple[str | None, dict]:
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
    # ``views`` are already at training resolution (half size), so no further downscale here.
    if iterations is None:
        iterations = splat_iterations_for(len(views) - len(views) // SplatSettings().holdout_every)
    result = train_splats(views, seed_points, seed_colors, SplatSettings(iterations=iterations, downscale=1))
    write_splat(out / "scene.splat", result.means, result.scales, result.quats_wxyz, result.rgb, result.opacity)
    asset, extra = "scene.splat", {"splat_format": "splat-dc"}
    if result.sh is not None:
        # Keep the view-dependent color: on held-out views of a real capture, dropping it cost ~3 dB PSNR.
        from mq3drecon.processing.web_location.splat_io import write_splat_ply

        write_splat_ply(out / "scene.ply", result.means, result.scales, result.quats_wxyz, result.sh, result.opacity)
        degree = int(round(np.sqrt(result.sh.shape[1]))) - 1
        asset, extra = "scene.ply", {"splat_format": f"ply-sh{degree}", "splat_dc_fallback": "scene.splat"}
    return asset, {**extra,
        "splats": int(len(result.means)), "splat_iterations": iterations, "splat_seconds": round(result.seconds, 1),
        "splat_train_psnr": round(result.train_psnr, 2), "splat_holdout_psnr": round(result.holdout_psnr, 2),
        **({"splat_camera_calibration": result.camera_calibration} if result.camera_calibration else {}),
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
    keeper = None
    if adb:
        from mq3drecon.streaming.adb import AdbReverseKeeper

        serials = adb_reverse_and_open(port, adb_path=adb_path, open_browser=open_headset)
        print(f"[serve] headset: adb reverse tcp:{port} on {serials or 'no device'}"
              + ("; opened the headset browser" if open_headset and serials else "; open http://localhost:%d in the Quest browser" % port))
        # A sleeping Quest drops wireless adb and the forward with it; restore both when it wakes.
        keeper = AdbReverseKeeper(
            [port], adb_path=adb_path, on_change=lambda text: print(f"[serve] {text}", flush=True),
            remember_file=locations_dir.parent / "headsets.json",  # reconnect a headset that is asleep at startup
        )
        keeper.tick()  # right away: a headset already offline at startup is reconnected now, not in 5 s
        keeper.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        if keeper is not None:
            keeper.stop()
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
    splat_iterations: int | None = None,  # None: splat_iterations_for(view count)
    splat_backend: str = "gsplat",
    spirula_path: str | Path | None = None,
    spirula_iterations: int = 30_000,
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
    # Splats get their own, larger view set at training resolution: they need many views to generalize
    # (halving 286 views cost 3 dB held-out PSNR on a real capture), while the bake only needs the best few.
    splat_file, splat_stats = None, {}
    if splats and splat_backend == "spirula":
        splat_file, splat_stats = _train_splats_spirula(
            project_dir, built.mesh.vertices[seen], built.mesh.colors[seen], out, spirula_path, spirula_iterations,
        )
    if splats and splat_file is None:  # gsplat, or the fallback when Spirula is unavailable
        splat_views = load_splat_views(project_dir)
        splat_file, splat_stats = _train_splats(built.mesh.vertices[seen], built.mesh.colors[seen], splat_views, out, splats, splat_iterations)
        if splat_stats:
            splat_stats["splat_views"] = len(splat_views)
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
