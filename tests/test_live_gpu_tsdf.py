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
    mesh = volume.extract_mesh(min_weight=1.0)

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
    mesh = volume.extract_mesh(min_weight=1.0)
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
