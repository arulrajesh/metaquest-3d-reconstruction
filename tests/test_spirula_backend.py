"""Spirula backend plumbing without the binary: dataset export, PLY reading, frame check, fallback."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from mq3drecon.processing.web_location.spirula import (
    OPENCV_TO_OPENGL, _check_identity_transform, find_spirula, read_splat_ply, write_nerfstudio_dataset,
)


def _view(i):
    w2c = np.eye(4)
    w2c[:3, 3] = [0.1 * i, 0.0, 0.0]
    return SimpleNamespace(rgb=np.full((8, 10, 3), 10 * i, np.uint8), intrinsic=np.array([[5.0, 0, 4.5], [0, 5.0, 3.5], [0, 0, 1]]),
                           world_to_camera=w2c)


def test_dataset_export_splits_holdout_like_train_splats_and_uses_opengl_poses(tmp_path):
    views = (_view(i) for i in range(16))  # a generator, as in the build
    split = write_nerfstudio_dataset(views, np.zeros((3, 3)), np.full((3, 3), 0.5), tmp_path, holdout_every=8)
    assert split == {"train_views": 14, "holdout_views": 2}
    train = json.loads((tmp_path / "transforms.json").read_text())
    held = json.loads((tmp_path / "eval_transforms.json").read_text())
    assert [f["file_path"] for f in held["frames"]] == ["images/00007.jpg", "images/00015.jpg"]  # index % 8 == 7
    first = train["frames"][1]  # view 1
    c2w = np.asarray(first["transform_matrix"]) @ np.linalg.inv(OPENCV_TO_OPENGL)
    assert np.allclose(np.linalg.inv(c2w)[:3, 3], [0.1, 0.0, 0.0])  # back to our world_to_camera
    assert (first["w"], first["h"], first["cx"]) == (10, 8, 4.5)
    assert (tmp_path / "images" / "00015.jpg").exists() and (tmp_path / "sparse_pc.ply").exists()
    assert train["ply_file_path"] == "sparse_pc.ply"


def _write_3dgs_ply(path, n=5, sh_rest=45):
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(sh_rest)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.zeros(n, dtype=[(name, "<f4") for name in names])
    data["x"] = np.arange(n)
    data["f_dc_0"], data["f_dc_1"], data["f_dc_2"] = 0.1, 0.2, 0.3
    for i in range(sh_rest):
        data[f"f_rest_{i}"] = i  # channel-major: 0..14 red, 15..29 green, 30..44 blue
    data["opacity"] = 0.0  # sigmoid -> 0.5
    data["scale_0"] = np.log(2.0)
    data["rot_0"] = 2.0  # unnormalized w
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n + "".join(f"property float {name}\n" for name in names) + "end_header\n"
    path.write_bytes(header.encode() + data.tobytes())


def test_read_splat_ply_activates_and_orders_sh_channel_major(tmp_path):
    _write_3dgs_ply(tmp_path / "s.ply")
    s = read_splat_ply(tmp_path / "s.ply")
    assert s["means"][:, 0].tolist() == [0, 1, 2, 3, 4]
    assert s["sh"].shape == (5, 16, 3)
    assert np.allclose(s["sh"][0, 0], [0.1, 0.2, 0.3])  # DC first
    assert np.allclose(s["sh"][0, 1], [0, 15, 30]) and np.allclose(s["sh"][0, 15], [14, 29, 44])
    assert np.allclose(s["opacity"], 0.5) and np.allclose(s["scales"][:, 0], 2.0)
    assert np.allclose(s["quats_wxyz"][0], [1, 0, 0, 0])


def test_transformed_training_frame_is_refused(tmp_path):
    good = {"world_from_train": {"scale": 1, "translation": [0, 0, 0], "rotation": {"matrix_3x3": np.eye(3).tolist()}}}
    (tmp_path / "good.json").write_text(json.dumps(good))
    _check_identity_transform(tmp_path / "good.json")
    bad = {"world_from_train": {"scale": 3.2, "translation": [0, 0, 0], "rotation": {"matrix_3x3": np.eye(3).tolist()}}}
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="transformed frame"):
        _check_identity_transform(tmp_path / "bad.json")


def test_missing_binary_falls_back(tmp_path, monkeypatch):
    monkeypatch.delenv("MQ3DRECON_SPIRULA", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert find_spirula(tmp_path / "nope.exe") is None
    from mq3drecon.workflows.web_location import _train_splats_spirula

    assert _train_splats_spirula(tmp_path, np.zeros((1, 3)), np.zeros((1, 3)), tmp_path, tmp_path / "nope.exe", 10) == (None, {})


def test_our_sh_ply_round_trips_through_the_reader(tmp_path):
    from mq3drecon.processing.web_location.splat_io import write_splat_ply

    rng = np.random.default_rng(0)
    n = 7
    means = rng.normal(size=(n, 3)).astype(np.float32)
    scales = rng.uniform(0.01, 0.2, size=(n, 3))
    quats = rng.normal(size=(n, 4))
    sh = rng.normal(size=(n, 16, 3)).astype(np.float32)
    opacity = rng.uniform(0.05, 0.95, size=n)
    write_splat_ply(tmp_path / "ours.ply", means, scales, quats, sh, opacity)
    back = read_splat_ply(tmp_path / "ours.ply")
    assert np.allclose(back["means"], means) and np.allclose(back["sh"], sh, atol=1e-6)
    assert np.allclose(back["scales"], scales, rtol=1e-5) and np.allclose(back["opacity"], opacity, atol=1e-5)
    assert np.allclose(np.abs((back["quats_wxyz"] * (quats / np.linalg.norm(quats, axis=1, keepdims=True))).sum(axis=1)), 1.0, atol=1e-5)
