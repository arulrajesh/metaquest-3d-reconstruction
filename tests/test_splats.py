import numpy as np
import pytest

from mq3drecon.processing.web_location.splat_io import read_splat, write_splat


def test_splat_file_round_trip_and_ordering(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    means, scales = rng.normal(size=(n, 3)).astype(np.float32), rng.uniform(0.01, 0.2, (n, 3)).astype(np.float32)
    quats = rng.normal(size=(n, 4))
    rgb, opacity = rng.random((n, 3)), rng.random(n)
    size = write_splat(tmp_path / "s.splat", means, scales, quats, rgb, opacity)
    assert size == 32 * n
    back = read_splat(tmp_path / "s.splat")
    order = np.argsort(-np.prod(scales.astype(np.float64), axis=1) * opacity)  # most important first
    assert np.allclose(back["means"], means[order])
    q = quats[order] / np.linalg.norm(quats[order], axis=1, keepdims=True)
    assert np.abs(back["quats_wxyz"] - q).max() < 1.0 / 128 + 1e-6


def test_splats_train_on_synthetic_room(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("gsplat")
    pytest.importorskip("warp")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for gsplat")
    from mq3drecon.processing.web_location.mesh_build import fuse_session
    from mq3drecon.processing.web_location.msvc_env import prepare_gsplat_build
    from mq3drecon.processing.web_location.splat_train import SplatSettings, train_splats
    from mq3drecon.processing.web_location.textured_visual import load_bake_views
    from mq3drecon.synthetic import SyntheticCaptureSpec, surface_distance, write_synthetic_capture

    prepare_gsplat_build()
    project = write_synthetic_capture(tmp_path / "room", SyntheticCaptureSpec(frames=32, color_size=320, depth_size=96, color_format="jpg"))
    built = fuse_session(project, voxel_size_m=0.04)
    result = train_splats(load_bake_views(project), built.mesh.vertices, built.mesh.colors,
                          SplatSettings(iterations=600, downscale=1, max_init_points=60_000))
    assert len(result.means) > 1000
    assert result.holdout_psnr > 22.0  # held-out views: the splats generalize, not just memorize
    # Splats sit on the room surfaces (Open3D world -> Unity world for the analytic scene).
    distance = surface_distance(result.means.astype(np.float64) * np.array([1.0, 1.0, -1.0]))
    assert np.median(distance) < 0.05
