"""Gaussian splat file writers (numpy only).

``.splat`` (antimatter15 layout, read by Spark and most web viewers), 32 bytes per splat:
position float32×3, scale float32×3 (linear), RGBA uint8 (A = opacity), rotation uint8×4
(w, x, y, z quaternion mapped from [-1, 1] to [0, 255]). Splats are sorted by
descending size × opacity so viewers that stream the file show the important ones first.
Positions are in MQ3DRecon's Open3D world frame (Y-up, right-handed), like the GLBs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814


def sh_dc_to_rgb(sh0: np.ndarray) -> np.ndarray:
    return np.clip(0.5 + SH_C0 * np.asarray(sh0, np.float64), 0.0, 1.0)


def write_splat(
    path: Path,
    means: np.ndarray,
    scales: np.ndarray,
    quats_wxyz: np.ndarray,
    rgb: np.ndarray,
    opacity: np.ndarray,
) -> int:
    """Write a ``.splat`` file. ``scales`` are linear (not log); ``opacity`` in [0, 1]. Returns bytes written."""
    means = np.asarray(means, np.float32)
    scales = np.asarray(scales, np.float32)
    q = np.asarray(quats_wxyz, np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    rgba = np.concatenate([np.asarray(rgb, np.float64), np.asarray(opacity, np.float64).reshape(-1, 1)], axis=1)
    importance = np.prod(scales.astype(np.float64), axis=1) * rgba[:, 3]
    order = np.argsort(-importance)
    record = np.zeros(len(means), dtype=[("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    record["pos"] = means[order]
    record["scale"] = scales[order]
    record["rgba"] = np.clip(np.round(rgba[order] * 255.0), 0, 255).astype(np.uint8)
    record["rot"] = np.clip(np.round(q[order] * 128.0 + 128.0), 0, 255).astype(np.uint8)
    data = record.tobytes()
    Path(path).write_bytes(data)
    return len(data)


def read_splat(path: Path) -> dict[str, np.ndarray]:
    raw = np.frombuffer(Path(path).read_bytes(), dtype=[("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    return {
        "means": raw["pos"].copy(),
        "scales": raw["scale"].copy(),
        "rgba": raw["rgba"].astype(np.float32) / 255.0,
        "quats_wxyz": (raw["rot"].astype(np.float32) - 128.0) / 128.0,
    }


def write_splat_ply(
    path: Path,
    means: np.ndarray,
    scales: np.ndarray,
    quats_wxyz: np.ndarray,
    sh: np.ndarray,
    opacity: np.ndarray,
) -> int:
    """Standard 3DGS binary PLY with view-dependent color (what Spark, SuperSplat and most trainers read).

    ``sh`` is (N, K, 3) with the DC term first; stored as f_dc_0..2 and f_rest_* channel-major (all red
    coefficients, then green, then blue). ``scales`` are linear (stored as logs), ``opacity`` in [0, 1] (stored as
    logits). The ``.splat`` format keeps only the DC color: on held-out views of a real capture, dropping the
    view-dependent part cost ~3 dB PSNR. Returns bytes written.
    """
    means = np.asarray(means, np.float32)
    n = len(means)
    sh = np.asarray(sh, np.float32).reshape(n, -1, 3)
    rest = sh[:, 1:, :].transpose(0, 2, 1).reshape(n, -1)  # channel-major
    q = np.asarray(quats_wxyz, np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    alpha = np.clip(np.asarray(opacity, np.float64), 1e-6, 1 - 1e-6)
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(rest.shape[1])]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    record = np.zeros(n, dtype=[(name, "<f4") for name in names])
    record["x"], record["y"], record["z"] = means[:, 0], means[:, 1], means[:, 2]
    for c in range(3):
        record[f"f_dc_{c}"] = sh[:, 0, c]
    for i in range(rest.shape[1]):
        record[f"f_rest_{i}"] = rest[:, i]
    record["opacity"] = np.log(alpha / (1 - alpha))
    log_scales = np.log(np.maximum(np.asarray(scales, np.float64), 1e-12))
    for i in range(3):
        record[f"scale_{i}"] = log_scales[:, i]
    for i in range(4):
        record[f"rot_{i}"] = q[:, i]
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
    header += "".join(f"property float {name}\n" for name in names) + "end_header\n"
    data = header.encode("ascii") + record.tobytes()
    Path(path).write_bytes(data)
    return len(data)
