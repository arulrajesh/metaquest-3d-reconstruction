import threading
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("open3d")

from mq3drecon.config.live_config import live_config_from_dict, preset
from mq3drecon.dataio import DataIO
from mq3drecon.models import Side
from mq3drecon.streaming.replay import replay_project
from mq3drecon.synthetic import SyntheticCaptureSpec, surface_distance, write_synthetic_capture
from mq3drecon.workflows.live_reconstruction import LIVE_MESH_FILE, run_live_capture


def test_live_config_presets_and_overrides():
    outdoor = preset("outdoor")
    assert outdoor.volume.extent_m == (48.0, 8.0, 48.0) and outdoor.volume.voxel_size_m == 0.06
    cfg = live_config_from_dict({"mode": "outdoor", "volume": {"voxel_size_m": 0.05}, "integration": {"sides": ["left", "right"]}})
    assert cfg.mode == "outdoor" and cfg.volume.voxel_size_m == 0.05 and cfg.volume.extent_m == (48.0, 8.0, 48.0)
    assert cfg.integration.sides == ("left", "right")
    with pytest.raises(ValueError, match="Unknown key"):
        live_config_from_dict({"volume": {"voxel": 1}})
    with pytest.raises(ValueError, match="mode must be"):
        preset("underwater")


def test_live_capture_records_and_reconstructs_a_streamed_session(tmp_path):
    project = write_synthetic_capture(tmp_path / "synthetic", SyntheticCaptureSpec(frames=20, color_size=160, depth_size=96))
    config = preset("indoor")
    config = replace(
        config,
        server=replace(config.server, port=0, discovery=False, adb=False),
        volume=replace(config.volume, extent_m=(6.0, 4.0, 6.0)),
        integration=replace(config.integration, queue_size=64),
        preview=replace(config.preview, viewer=False, mesh_interval_s=0.2),
    )
    stop, done, ports = threading.Event(), threading.Event(), []
    listening = threading.Event()
    results = []

    def on_listening(port):
        ports.append(port)
        listening.set()

    worker = threading.Thread(
        target=lambda: results.extend(run_live_capture(
            workspace=tmp_path / "ws", config=config, stop_event=stop,
            on_session_complete=lambda completed: done.set(), on_listening=on_listening,
        )),
        daemon=True,
    )
    worker.start()
    assert listening.wait(30)
    replay_project(project, "127.0.0.1", ports[0], speed=0)
    assert done.wait(60)
    stop.set()
    worker.join(30)

    assert len(results) == 1
    session = results[0]
    assert session.summary.color_frames == {"left": 20, "right": 20}
    assert session.summary.end_reason == "replay_complete"
    assert DataIO(session.project_dir).color.load_color_dataset(Side.LEFT, use_cache=False) is not None
    assert (session.project_dir / LIVE_MESH_FILE).exists()

    mesh = session.final_snapshot.mesh
    assert 5 <= session.final_snapshot.stats.frames_integrated <= 20  # only keyframes are fused
    assert (session.project_dir / "keyframes.csv").exists()  # compacted in place after the session
    distance = surface_distance(mesh.vertices.astype(np.float64) * np.array([1.0, 1.0, -1.0]))
    assert len(mesh.triangles) > 5_000 and np.median(distance) < 0.01


def test_headset_receives_keyframes_surface_mesh_and_status(tmp_path):
    from mq3drecon.streaming.feedback import KeyframeNotice, LiveStatusUpdate, SurfaceMesh
    from mq3drecon.streaming.replay import iter_project_messages
    from mq3drecon.streaming.sender import LiveSender
    from mq3drecon.synthetic.room_capture import ROOM_MAX, ROOM_MIN

    project = write_synthetic_capture(tmp_path / "synthetic", SyntheticCaptureSpec(frames=24, color_size=160, depth_size=96))
    config = preset("indoor")
    config = replace(
        config,
        server=replace(config.server, port=0, discovery=False, adb=False),
        volume=replace(config.volume, extent_m=(6.0, 4.0, 6.0)),
        integration=replace(config.integration, queue_size=64),
        preview=replace(config.preview, viewer=False, mesh_interval_s=0.2),
        feedback=replace(config.feedback, surface_interval_s=0.2, status_interval_s=0.2),
    )
    stop, done, ports, listening = threading.Event(), threading.Event(), [], threading.Event()
    worker = threading.Thread(target=lambda: run_live_capture(
        workspace=tmp_path / "ws", config=config, stop_event=stop,
        on_session_complete=lambda c: done.set(), on_listening=lambda p: (ports.append(p), listening.set()),
    ), daemon=True)
    worker.start()
    assert listening.wait(30)

    received = []
    with LiveSender("127.0.0.1", ports[0]) as sender:
        reader_stop = threading.Event()

        def read():
            while not reader_stop.is_set():
                message = sender.receive(timeout_s=0.2)
                if message is not None:
                    received.append(message)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        messages = [m for _, m in iter_project_messages(project)]
        for message in messages[:-1]:  # hold SESSION_END until feedback has had time to arrive
            sender.send(message)
        deadline = __import__("time").monotonic() + 20
        while __import__("time").monotonic() < deadline and not any(isinstance(m, SurfaceMesh) for m in received):
            __import__("time").sleep(0.1)
        sender.send(messages[-1])
        assert done.wait(60)
        reader_stop.set()
        reader.join(5)
    stop.set()
    worker.join(30)

    notices = [m for m in received if isinstance(m, KeyframeNotice)]
    meshes = [m for m in received if isinstance(m, SurfaceMesh)]
    statuses = [m for m in received if isinstance(m, LiveStatusUpdate)]
    assert notices and notices[0].depth_kept and notices[0].reason == "first"
    assert any(not n.depth_kept for n in notices) or len(notices) == 24
    assert meshes and statuses and statuses[-1].values["keyframes"] > 0
    positions, rgba, triangles = meshes[-1].unpack()
    assert len(triangles) > 100 and triangles.max() < len(positions)
    # Unity frame: inside the synthetic room's bounds (which are defined in Unity world coordinates).
    assert (positions.min(axis=0) > ROOM_MIN - 0.2).all() and (positions.max(axis=0) < ROOM_MAX + 0.2).all()
    assert rgba[:, 3].mean() > 100  # mostly confident surfaces
