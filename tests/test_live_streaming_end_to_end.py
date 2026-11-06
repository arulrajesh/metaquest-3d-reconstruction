import threading

import numpy as np
import pytest

from mq3drecon.config.project_path_config import ProjectPathConfig
from mq3drecon.dataio import DataIO
from mq3drecon.dataio.depth_data_io import DepthDataIO
from mq3drecon.models import Side
from mq3drecon.streaming.discovery import DiscoveryResponder, discover_receivers
from mq3drecon.streaming.messages import ColorFrame, DepthFrame, SessionEnd, SessionStart
from mq3drecon.streaming.receiver import LiveReceiver
from mq3drecon.streaming.replay import ReplayOptions, iter_project_messages, replay_project
from mq3drecon.streaming.session_recorder import RecordingOptions, SessionRecorder
from mq3drecon.synthetic import SyntheticCaptureSpec, write_synthetic_capture


@pytest.fixture(scope="module")
def synthetic_project(tmp_path_factory):
    return write_synthetic_capture(tmp_path_factory.mktemp("synthetic") / "room", SyntheticCaptureSpec(frames=8, color_size=64, depth_size=32))


class RecordingHandler:
    def __init__(self, root, options=None):
        self.root = root
        self.options = options
        self.recorder = None
        self.summary = None
        self.done = threading.Event()
        self.events = []

    def on_connect(self, peer):
        self.events.append("connect")

    def on_message(self, message):
        if isinstance(message, SessionStart):
            self.recorder = SessionRecorder(self.root / message.session_name, message, self.options)
        elif isinstance(message, ColorFrame):
            self.recorder.write_color(message)
        elif isinstance(message, DepthFrame):
            self.recorder.write_depth(message)
        elif isinstance(message, SessionEnd):
            self.summary = self.recorder.finalize(message.reason, message.dropped)

    def on_disconnect(self, peer, error):
        self.events.append(("disconnect", error))
        self.done.set()


def test_replay_messages_are_time_ordered_and_framed(synthetic_project):
    items = list(iter_project_messages(synthetic_project))
    assert isinstance(items[0][1], SessionStart) and isinstance(items[-1][1], SessionEnd)
    times = [t for t, _ in items]
    assert times == sorted(times)
    assert sum(isinstance(m, ColorFrame) for _, m in items) == 16
    assert sum(isinstance(m, DepthFrame) for _, m in items) == 16


def _stream(project, handler):
    receiver = LiveReceiver(handler, host="127.0.0.1", port=0)
    port = receiver.start()
    try:
        replay_project(project, "127.0.0.1", port, speed=0, options=ReplayOptions(depth_encoding="f32"))
        assert handler.done.wait(10)
    finally:
        receiver.stop()


def test_default_recording_is_compact_left_depth_png(synthetic_project, tmp_path):
    handler = RecordingHandler(tmp_path)
    _stream(synthetic_project, handler)
    assert handler.summary.depth_frames == {"left": 8}
    recorded = tmp_path / "room"
    assert sorted(p.suffix for p in (recorded / "left_depth").iterdir()) == [".png"] * 8
    assert not (recorded / "right_depth").exists()
    depth_io = DepthDataIO(ProjectPathConfig(recorded).depth)
    source_io = DepthDataIO(ProjectPathConfig(synthetic_project).depth)
    recorded_depth = depth_io.build_depth_dataset(Side.LEFT)
    source_depth = source_io.build_depth_dataset(Side.LEFT)
    a = depth_io.load_depth_map_by_index(Side.LEFT, recorded_depth, 3)
    b = source_io.load_depth_map_by_index(Side.LEFT, source_depth, 3)
    valid = (a > 0) & (b > 0) & (b < 5.0)
    assert np.abs(a[valid] - b[valid]).max() < 0.005  # 16-bit NDC: sub-centimeter within 5 m


def test_stream_to_receiver_records_a_loadable_mruk_project(synthetic_project, tmp_path):
    handler = RecordingHandler(tmp_path, RecordingOptions(depth_format="raw", depth_sides=(Side.LEFT, Side.RIGHT)))
    receiver = LiveReceiver(handler, host="127.0.0.1", port=0)
    port = receiver.start()
    try:
        replay_project(synthetic_project, "127.0.0.1", port, speed=0, options=ReplayOptions(depth_encoding="f32"))
        assert handler.done.wait(10)
    finally:
        receiver.stop()

    assert handler.events[0] == "connect" and handler.events[1] == ("disconnect", None)
    assert handler.summary.color_frames == {"left": 8, "right": 8}
    assert handler.summary.depth_frames == {"left": 8, "right": 8}
    assert handler.summary.stereo_pairs == 8

    recorded = tmp_path / "room"
    color = DataIO(recorded).color.load_color_dataset(Side.LEFT, use_cache=False)
    assert all(name.endswith(".jpg") for name in color.image_file_names)
    source = DataIO(synthetic_project).color.load_color_dataset(Side.LEFT, use_cache=False)
    assert np.allclose(color.transforms.positions, source.transforms.positions)
    rgb_recorded = DataIO(recorded).color.load_color_rgb_image(dataset=color, index=3).astype(float)
    rgb_source = DataIO(synthetic_project).color.load_color_rgb_image(dataset=source, index=3).astype(float)
    assert np.abs(rgb_recorded - rgb_source).mean() < 3.0  # JPEG q90, same orientation

    depth_io = DepthDataIO(ProjectPathConfig(recorded).depth)
    source_io = DepthDataIO(ProjectPathConfig(synthetic_project).depth)
    recorded_depth = depth_io.build_depth_dataset(Side.RIGHT)
    source_depth = source_io.build_depth_dataset(Side.RIGHT)
    assert np.array_equal(recorded_depth.timestamps, source_depth.timestamps)
    assert np.array_equal(
        depth_io.load_depth_map_by_index(Side.RIGHT, recorded_depth, 2),
        source_io.load_depth_map_by_index(Side.RIGHT, source_depth, 2),
    )


def test_recorder_refuses_to_overwrite(tmp_path):
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError):
        SessionRecorder(tmp_path / "s", SessionStart(session_name="s", intrinsics={}))


def test_discovery_on_localhost():
    responder = DiscoveryResponder(tcp_port=9870, discovery_port=0, host_name="laptop")
    port = responder.start()
    try:
        found = discover_receivers(timeout_s=1.0, discovery_port=port, target="127.0.0.1")
    finally:
        responder.stop()
    assert [(r.address, r.port, r.host_name) for r in found] == [("127.0.0.1", 9870, "laptop")]
