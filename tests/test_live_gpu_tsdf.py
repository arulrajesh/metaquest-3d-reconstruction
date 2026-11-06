import numpy as np
import pytest

pytest.importorskip("warp")

from mq3drecon.models import Side
from mq3drecon.processing.live.exclusion import apply_exclusion, torso_capsule
from mq3drecon.processing.live.gpu_tsdf import GpuTsdfVolume, VolumeSpec
from mq3drecon.processing.live.native_depth import color_to_image, depth_to_metric
from mq3drecon.streaming.messages import ColorFrame, DepthFrame, SessionStart
from mq3drecon.streaming.replay import iter_project_messages
from mq3drecon.synthetic import SyntheticCaptureSpec, scene_color, surface_distance, write_synthetic_capture


@pytest.fixture(scope="module")
def synthetic_frames(tmp_path_factory):
    project = write_synthetic_capture(
        tmp_path_factory.mktemp("tsdf") / "room", SyntheticCaptureSpec(frames=24, color_size=160, depth_size=96)
    )
    intrinsics, colors, depths = None, {}, []
    for _, message in iter_project_messages(project):
        if isinstance(message, SessionStart):
            intrinsics = message.intrinsics
        elif isinstance(message, ColorFrame) and message.side == Side.LEFT:
            colors[message.timestamp_us] = message
        elif isinstance(message, DepthFrame) and message.side == Side.LEFT:
            depths.append(message)
    return intrinsics, colors, depths


def _integrate(volume, synthetic_frames, exclude_torso=False):
    intrinsics, colors, depths = synthetic_frames
    for frame in depths:
        depth = depth_to_metric(frame, depth_min_m=0.2, depth_max_m=5.0)
        if exclude_torso:
            depth, _ = apply_exclusion(depth, [torso_capsule(depth.view.position)])
        nearest = min(colors, key=lambda t: abs(t - depth.view.timestamp_us))
        color = color_to_image(colors[nearest], intrinsics[Side.LEFT])
        volume.integrate(depth.depth_m, depth.view.intrinsic, depth.view.world_to_camera,
                         color.rgb, color.view.intrinsic, color.view.world_to_camera)


def test_live_tsdf_reconstructs_synthetic_room_geometry_and_color(synthetic_frames):
    volume = GpuTsdfVolume(VolumeSpec.around((0, 1.3, 0), (4.4, 2.9, 5.4), 0.04, 0.16))
    _integrate(volume, synthetic_frames)
    mesh = volume.extract_mesh()

    assert len(mesh.triangles) > 10_000
    unity_points = mesh.vertices.astype(np.float64) * np.array([1.0, 1.0, -1.0])  # Open3D world -> Unity world
    distance = surface_distance(unity_points)
    assert np.median(distance) < 0.005
    assert np.percentile(distance, 95) < 0.02
    assert mesh.color_valid.mean() > 0.8
    color_error = np.abs(mesh.colors - scene_color(unity_points)).mean(axis=1)[mesh.color_valid]
    assert np.median(color_error) < 0.01  # a mirrored or offset color mapping fails this by a wide margin
    assert np.percentile(color_error, 95) < 0.05


def test_mesh_excludes_never_observed_space(synthetic_frames):
    volume = GpuTsdfVolume(VolumeSpec.around((0, 1.3, 0), (4.4, 2.9, 5.4), 0.04, 0.16))
    _integrate(volume, synthetic_frames)
    mesh = volume.extract_mesh()
    # No spurious "back faces" outside the room: nothing beyond the walls by more than a truncation band.
    unity_points = mesh.vertices.astype(np.float64) * np.array([1.0, 1.0, -1.0])
    assert (np.abs(unity_points[:, 0]) < 2.0 + 0.2).all()
    assert (np.abs(unity_points[:, 2]) < 2.5 + 0.2).all()


def test_torso_exclusion_removes_points_below_the_head():
    from mq3drecon.processing.live.native_depth import CameraView, MetricDepth

    k = np.array([[50.0, 0, 50.0], [0, 50.0, 50.0], [0, 0, 1]])
    # Camera at the head looking straight down (Open3D camera Z forward = world -Y).
    cw = np.eye(4)
    cw[:3, :3] = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, -1.0, 0]])
    cw[:3, 3] = [0.0, 1.6, 0.0]
    depth = MetricDepth(CameraView(Side.LEFT, 0, k, cw, np.linalg.inv(cw)), np.full((101, 101), 1.0, np.float32))
    cleaned, removed = apply_exclusion(depth, [torso_capsule(cw[:3, 3])])
    assert removed > 0
    assert cleaned.depth_m[50, 50] == 0.0  # straight down, 1 m below the head: inside the torso capsule
    assert cleaned.depth_m[0, 0] == 1.0  # far corner of the view: outside the 0.35 m radius


def test_tracking_jump_starts_a_new_segment(synthetic_frames):
    from dataclasses import replace as dc_replace

    from mq3drecon.config.live_config import preset
    from mq3drecon.processing.live.reconstructor import LiveReconstructor
    from mq3drecon.streaming.messages import Pose

    intrinsics, colors, depths = synthetic_frames
    config = preset("indoor")
    config = dc_replace(
        config,
        volume=dc_replace(config.volume, extent_m=(6.0, 4.0, 6.0)),
        integration=dc_replace(config.integration, queue_size=64),
        preview=dc_replace(config.preview, mesh_interval_s=60.0),
    )
    reconstructor = LiveReconstructor(config)
    try:
        reconstructor.begin_session(intrinsics)
        for color in colors.values():
            reconstructor.add_color(color)
        half = len(depths) // 2
        for frame in depths[:half]:
            reconstructor.add_depth(frame)
        for frame in depths[half:]:  # same frames, tracking origin shifted 3 m: a reset
            p = frame.create_pose.position
            shifted = dc_replace(frame, create_pose=Pose((p[0] + 3.0, p[1], p[2]), frame.create_pose.rotation))
            reconstructor.add_depth(shifted)
        snapshot = reconstructor.end_session()
    finally:
        reconstructor.close()
    assert snapshot.stats.tracking_jumps == 1
    assert len(snapshot.previous_segments) == 1 and len(snapshot.previous_segments[0].triangles) > 0
    assert len(snapshot.mesh.triangles) > 0


def _plane_view(distance=1.0, size=64, f=60.0):
    """A camera at the origin looking down +Z (Open3D convention) at a fronto-parallel wall."""
    k = np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]])
    return k, np.eye(4), np.full((size, size), distance, np.float32)


def test_kalman_tsdf_rejects_outliers_and_reports_information_gain():
    k, wc, wall = _plane_view()
    volume = GpuTsdfVolume(VolumeSpec.around((0, 0, 1.0), (1.2, 1.2, 0.6), 0.02, 0.08))
    first = volume.integrate(wall, k, wc)
    assert first.new_voxels > 0 and first.novelty > 0.9  # unobserved surface: nearly full variance reduction
    for _ in range(10):
        repeat = volume.integrate(wall, k, wc)
    assert repeat.new_voxels == 0 and repeat.novelty < 0.05 * first.novelty  # nothing new to learn: low information

    # A single spurious reading 6 cm in front of the wall in the middle of the view (e.g. a passer-by) is
    # rejected by the innovation gate for voxels already confident about the wall.
    spike = wall.copy()
    spike[24:40, 24:40] = 0.94
    score = volume.integrate(spike, k, wc)
    assert score.rejected > 0
    mesh = volume.extract_mesh()
    z = mesh.vertices[:, 2]
    assert np.abs(z - 1.0).max() < 0.02  # the wall stays put; no ghost at 0.94 m
    assert mesh.sigma_m is not None and np.median(mesh.sigma_m) < 0.005


def test_kalman_tsdf_removes_a_ghost_when_something_moves_out_of_view():
    # Something (a person, a box) stands 10 cm in front of the wall, then leaves. Once the camera sees
    # through to the wall again, free-space and gated readings must erase the old surface.
    # (A surface that moves *closer* hides the old one; no view can disprove it, which is inherent to TSDFs.)
    k, wc, wall = _plane_view()
    volume = GpuTsdfVolume(VolumeSpec.around((0, 0, 1.0), (1.2, 1.2, 0.6), 0.02, 0.08))
    for _ in range(8):
        volume.integrate(np.full_like(wall, 0.9), k, wc)
    for _ in range(30):
        volume.integrate(wall, k, wc)
    z = volume.extract_mesh().vertices[:, 2]
    assert len(z) > 0
    assert np.abs(z - 1.0).max() < 0.02  # only the wall remains; no ghost at 0.9 m
