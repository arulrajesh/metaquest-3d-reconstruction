"""A Wi-Fi drop mid-capture must not split the session: the headset resumes into the same recording."""

import time
from dataclasses import replace

import pytest

from mq3drecon.streaming.messages import ColorFrame, DepthFrame, SessionEnd, SessionStart
from mq3drecon.streaming.replay import iter_project_messages
from mq3drecon.synthetic import SyntheticCaptureSpec, write_synthetic_capture
from mq3drecon.workflows.live_session import LiveSessionController


@pytest.fixture(scope="module")
def messages(tmp_path_factory):
    project = write_synthetic_capture(tmp_path_factory.mktemp("synthetic") / "room", SyntheticCaptureSpec(frames=8, color_size=64, depth_size=32))
    return [m for _, m in iter_project_messages(project)]


def _split(messages):
    start = next(m for m in messages if isinstance(m, SessionStart))
    end = next(m for m in messages if isinstance(m, SessionEnd))
    frames = [m for m in messages if isinstance(m, (ColorFrame, DepthFrame))]
    return start, frames, end


def test_resume_after_a_drop_continues_the_same_session(messages, tmp_path):
    start, frames, end = _split(messages)
    controller = LiveSessionController(tmp_path / "sessions", resume_grace_s=30.0)
    half = len(frames) // 2

    controller.on_connect("headset:1")
    controller.on_message(start)
    for frame in frames[:half]:
        controller.on_message(frame)
    controller.on_disconnect("headset:1", ConnectionResetError("wifi dropped"))

    controller.on_connect("headset:2")  # new socket, same app session
    controller.on_message(replace(start, resume=True))
    for frame in frames[half:]:
        controller.on_message(frame)
    controller.on_message(end)

    assert len(controller.completed) == 1
    summary = controller.completed[0].summary
    colors = sum(isinstance(f, ColorFrame) for f in frames)
    left_depths = sum(isinstance(f, DepthFrame) and f.side.name == "LEFT" for f in frames)  # recorder keeps left depth only
    assert sum(summary.color_frames.values()) == colors
    assert summary.depth_frames == {"left": left_depths}
    assert [p.name for p in (tmp_path / "sessions").iterdir()] == [start.session_name]  # no "_2" split


def test_resume_after_the_grace_period_starts_a_new_session_folder(messages, tmp_path):
    start, frames, end = _split(messages)
    controller = LiveSessionController(tmp_path / "sessions", resume_grace_s=0.05)
    controller.on_connect("headset:1")
    controller.on_message(start)
    controller.on_message(frames[0])
    controller.on_disconnect("headset:1", ConnectionResetError("gone"))
    deadline = time.time() + 5
    while not controller.completed and time.time() < deadline:
        time.sleep(0.02)
    assert controller.completed and controller.completed[0].summary.end_reason == "disconnected"

    controller.on_connect("headset:2")
    controller.on_message(replace(start, resume=True))
    controller.on_message(frames[1])
    controller.on_message(end)
    names = sorted(p.name for p in (tmp_path / "sessions").iterdir())
    assert names == [start.session_name, f"{start.session_name}_2"]
