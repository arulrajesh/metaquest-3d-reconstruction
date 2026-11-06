"""Optional splat trainer: Spirula Studio, an external binary run as a separate process (GPL-3.0, not bundled).

On the real 1728x1728 recapture (2026-11-07) it beat our gsplat trainer on views neither trainer saw, at native
resolution: 29.87 dB PSNR and 36% of the photos' fine detail (its SH3 PLY) against 27.12 dB and 13% (our
DC-only ``.splat``). It combines MCMC/IGS+/MRNF-style densification with per-photo exposure correction
(PPISP + bilateral grid). Point ``spirula_path``, the ``MQ3DRECON_SPIRULA`` environment variable, or PATH at
``spirula.exe`` to use it (https://github.com/harry7557558/spirula-studio, releases).

The dataset handed over is a Nerfstudio ``transforms.json`` built from exactly the views, intrinsics and
headset poses our own trainer uses, so both are judged the same way. Its output, ``splat.ply``, is a standard
3DGS PLY with SH3 in our world frame (``--scene-center none --orientation-method none``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])


def find_spirula(explicit: str | Path | None = None) -> Path | None:
    for candidate in (explicit, os.environ.get("MQ3DRECON_SPIRULA"), shutil.which("spirula")):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


def write_nerfstudio_dataset(views, seed_points: np.ndarray, seed_colors: np.ndarray, out_dir: Path, *, holdout_every: int = 8) -> dict:
    """Images + ``transforms.json`` (training views) + ``eval_transforms.json`` (every ``holdout_every``-th view,
    index % n == n - 1, the same split as ``train_splats``) + ``sparse_pc.ply`` seed. ``views`` may be a generator."""
    import cv2

    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    train_frames, eval_frames = [], []
    for index, view in enumerate(views):
        name = f"images/{index:05d}.jpg"
        cv2.imwrite(str(out_dir / name), cv2.cvtColor(view.rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        h, w = view.rgb.shape[:2]
        k = view.intrinsic
        frame = {
            "file_path": name, "transform_matrix": (np.linalg.inv(view.world_to_camera) @ OPENCV_TO_OPENGL).tolist(),
            "fl_x": float(k[0, 0]), "fl_y": float(k[1, 1]), "cx": float(k[0, 2]), "cy": float(k[1, 2]), "w": int(w), "h": int(h),
        }
        (eval_frames if holdout_every and index % holdout_every == holdout_every - 1 else train_frames).append(frame)
    _write_point_ply(out_dir / "sparse_pc.ply", np.asarray(seed_points, np.float32), np.asarray(seed_colors, np.float32))
    common = {"camera_model": "OPENCV", "ply_file_path": "sparse_pc.ply"}
    (out_dir / "transforms.json").write_text(json.dumps({**common, "frames": train_frames}), encoding="utf-8")
    (out_dir / "eval_transforms.json").write_text(json.dumps({**common, "frames": eval_frames}), encoding="utf-8")
    return {"train_views": len(train_frames), "holdout_views": len(eval_frames)}


def _write_point_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    record = np.zeros(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    record["x"], record["y"], record["z"] = points[:, 0], points[:, 1], points[:, 2]
    rgb = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint8)
    record["red"], record["green"], record["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    Path(path).write_bytes(header.encode("ascii") + record.tobytes())


@dataclass(frozen=True)
class SpirulaRun:
    splat_ply: Path
    seconds: float
    version: str
    log: Path


def train_with_spirula(
    spirula: Path,
    dataset_dir: Path,
    out_dir: Path,
    *,
    iterations: int = 30_000,
    cap_max: int = 1_000_000,
    resolution_divisor: int = 2,
    preset: str = "3dgs",
    timeout_s: float = 4 * 3600,
) -> SpirulaRun:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "spirula_train.log"
    command = [
        str(spirula), "train", preset, "--data", str(dataset_dir), "--data-format", "nerfstudio",
        "--output-dir-prefix", str(out_dir), "--output-dir-name", "run",
        "--eval-mode", "all", "--scene-center", "none", "--orientation-method", "none",
        "--train-resolution-divisor", str(resolution_divisor), "--cap-max", str(cap_max), "--num-iterations", str(iterations),
        "--disable-viewer", "1", "--keep-viewer-alive", "0", "--eval-lpips", "0",
    ]
    start = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout_s)
    text = log.read_text(encoding="utf-8", errors="replace")
    if result.returncode != 0 or "Training complete" not in text:
        tail = "\n".join(text.splitlines()[-15:])
        raise RuntimeError(f"Spirula training failed (exit {result.returncode}); see {log}\n{tail}")
    run = out_dir / "run"
    checkpoints = sorted(run.glob("step-*.ckpt/splat.ply"))
    if not checkpoints:
        raise RuntimeError(f"Spirula wrote no splat.ply under {run}")
    _check_identity_transform(run / "scene_transform.json")
    return SpirulaRun(splat_ply=checkpoints[-1], seconds=time.time() - start, version=spirula_version(spirula), log=log)


def spirula_version(spirula: Path) -> str:
    """``Spirula Studio <version> (<commit>)`` from ``--help`` (training logs don't print it)."""
    import re

    try:
        text = subprocess.run([str(spirula), "--help"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return "Spirula Studio"
    found = re.search(r"Spirula Studio \S+ \([0-9a-f]+\)", text)
    return found.group(0) if found else "Spirula Studio"


def _check_identity_transform(path: Path) -> None:
    """The splats must be in our world frame; refuse a run whose recorded train frame differs."""
    if not path.exists():
        return
    transform = json.loads(path.read_text(encoding="utf-8")).get("world_from_train", {})
    matrix = np.asarray(transform.get("rotation", {}).get("matrix_3x3", np.eye(3)), float)
    scale = float(transform.get("scale", 1.0))
    translation = np.asarray(transform.get("translation", [0, 0, 0]), float)
    if not (np.allclose(matrix, np.eye(3), atol=1e-6) and abs(scale - 1.0) < 1e-6 and np.allclose(translation, 0.0, atol=1e-6)):
        raise RuntimeError(f"Spirula trained in a transformed frame ({path}); expected identity with --scene-center none")


def read_splat_ply(path: Path) -> dict[str, np.ndarray]:
    """Standard 3DGS binary PLY -> activated arrays: means, quats_wxyz (unit), scales (linear), opacity [0, 1],
    sh (N, K, 3) with the DC term first (K = (degree + 1)^2)."""
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header")
    end = raw.index(b"\n", end) + 1
    header = raw[:end].decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError(f"{path}: expected a binary little-endian PLY")
    types = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4"}
    count, fields, in_vertex = 0, [], False
    for line in header:
        parts = line.split()
        if parts[:2] == ["element", "vertex"]:
            count, in_vertex = int(parts[2]), True
        elif parts and parts[0] == "element":
            in_vertex = False
        elif in_vertex and parts and parts[0] == "property":
            fields.append((parts[2], types[parts[1]]))
    data = np.frombuffer(raw, dtype=np.dtype(fields), count=count, offset=end)
    rest = sorted((n for n in data.dtype.names if n.startswith("f_rest_")), key=lambda n: int(n.rsplit("_", 1)[1]))
    dc = np.stack([data[f"f_dc_{i}"] for i in range(3)], axis=1).astype(np.float32)
    sh = dc[:, None, :]
    if rest:  # stored channel-major: all R coefficients, then G, then B
        sh_rest = np.stack([data[n] for n in rest], axis=1).astype(np.float32).reshape(count, 3, -1).transpose(0, 2, 1)
        sh = np.concatenate([sh, sh_rest], axis=1)
    quats = np.stack([data[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float32)
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    return {
        "means": np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32),
        "quats_wxyz": quats,
        "scales": np.exp(np.stack([data[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float32)),
        "opacity": 1.0 / (1.0 + np.exp(-np.asarray(data["opacity"], np.float32))),
        "sh": sh,
    }


def holdout_psnr(splats: dict[str, np.ndarray], views, *, holdout_every: int = 8, device: str = "cuda") -> float:
    """Mean PSNR (with SH) on the held-out views (index % n == n - 1). Needs torch + gsplat."""
    import torch

    from mq3drecon.processing.web_location.msvc_env import prepare_gsplat_build

    prepare_gsplat_build()
    from gsplat import rasterization

    sh = torch.tensor(splats["sh"], device=device)
    degree = int(round(np.sqrt(sh.shape[1]))) - 1
    tensors = {k: torch.tensor(splats[k], device=device) for k in ("means", "quats_wxyz", "scales", "opacity")}
    psnrs = []
    for index, view in enumerate(views):
        if index % holdout_every != holdout_every - 1:
            continue
        h, w = view.rgb.shape[:2]
        image, _, _ = rasterization(
            means=tensors["means"], quats=tensors["quats_wxyz"], scales=tensors["scales"], opacities=tensors["opacity"],
            colors=sh, sh_degree=degree, width=w, height=h,
            viewmats=torch.tensor(view.world_to_camera[None], dtype=torch.float32, device=device),
            Ks=torch.tensor(view.intrinsic[None], dtype=torch.float32, device=device),
        )
        mse = torch.mean((image[0].clamp(0, 1) - torch.tensor(view.rgb, device=device, dtype=torch.float32) / 255.0) ** 2).item()
        psnrs.append(10.0 * np.log10(1.0 / max(mse, 1e-12)))
    return float(np.mean(psnrs)) if psnrs else float("nan")
