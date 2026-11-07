import json

import numpy as np
import pytest

from mq3drecon.processing.live.keyframes import KeyframePolicy, KeyframeSelector, select_color_frames
from mq3drecon.processing.live.pose_filter import PoseFilterResult


def _motion(t, speed=0.3, angular=20.0, jump=False):
    return PoseFilterResult(t, 0.0, jump, speed, angular)


def test_depth_is_kept_for_useful_information_and_color_only_when_new_and_spaced():
    selector = KeyframeSelector(KeyframePolicy(depth_min_info_gain=1000, color_min_novelty=0.15, color_min_gap_s=0.5))
    first = selector.decide(0, novelty=1.0, info_gain=50_000, motion=_motion(0.0), focal_px=870, color_novelty=1.0)
    redundant = selector.decide(100_000, novelty=0.01, info_gain=20, motion=_motion(0.1), focal_px=870, color_novelty=0.0)
    informative_soon = selector.decide(200_000, novelty=0.1, info_gain=5000, motion=_motion(0.2), focal_px=870, color_novelty=0.5)
    informative_later = selector.decide(800_000, novelty=0.1, info_gain=5000, motion=_motion(0.8), focal_px=870, color_novelty=0.5)
    known_color = selector.decide(1_500_000, novelty=0.1, info_gain=5000, motion=_motion(1.5), focal_px=870, color_novelty=0.05)
    assert (first.depth_kept, first.color_wanted, first.reason) == (True, True, "first")
    assert not redundant.depth_kept and redundant.reason == "redundant"
    assert informative_soon.depth_kept and not informative_soon.color_wanted  # color too soon after the last
    assert informative_later.depth_kept and informative_later.color_wanted
    assert known_color.depth_kept and not known_color.color_wanted  # surface already has good color


def test_tracking_jump_starts_segment_and_extreme_blur_blocks_color():
    selector = KeyframeSelector()
    selector.decide(0, 1.0, 1e5, _motion(0.0), 870, 1.0)
    jump = selector.decide(500_000, 0.0, 0.0, _motion(0.5, jump=True), 870, 0.0)
    assert jump.depth_kept and jump.reason == "segment_start"
    whip = selector.decide(1_500_000, 0.5, 1e5, _motion(1.5, angular=400.0), 870, 1.0)
    assert whip.depth_kept and not whip.color_wanted and whip.reason.endswith("+blurry")


def test_color_selection_prefers_the_locally_sharpest_frame():
    times = [0, 100_000, 200_000, 300_000]
    chosen = select_color_frames([150_000], times, [1.0] * 4, KeyframePolicy(color_window_s=0.25), sharpness=[5, 50, 9, 7])
    assert chosen == [100_000]


@pytest.mark.filterwarnings("ignore")
def test_compaction_keeps_geometry_and_shrinks_the_session(tmp_path):
    pytest.importorskip("warp")
    from mq3drecon.synthetic import SyntheticCaptureSpec, write_synthetic_capture
    from mq3drecon.workflows.compaction import COMPACTION_JSON, KEYFRAMES_CSV, compact_session

    source = write_synthetic_capture(tmp_path / "room", SyntheticCaptureSpec(frames=30, color_size=160, depth_size=96))
    report = compact_session(source, tmp_path / "compact")
    assert report.bytes_after < report.bytes_before / 3
    assert 0 < report.frames_after["depth_left"] <= report.frames_before["depth_left"]
    assert report.frames_after["depth_right"] == 0
    assert 0 < report.frames_after["color_left"] < report.frames_before["color_left"]
    assert report.frames_after["color_left"] == report.frames_after["color_right"]
    assert (tmp_path / "compact" / KEYFRAMES_CSV).exists()
    saved = json.loads((tmp_path / "compact" / COMPACTION_JSON).read_text())
    assert saved["policy"]["depth_min_info_gain"] == KeyframePolicy().depth_min_info_gain
    assert source.exists()  # never deleted unless asked


@pytest.mark.filterwarnings("ignore")
def test_compaction_keeps_extra_color_views_for_splats(tmp_path):
    pytest.importorskip("warp")
    from mq3drecon.synthetic import SyntheticCaptureSpec, write_synthetic_capture
    from mq3drecon.workflows.compaction import compact_session

    # 60 frames at the synthetic rate: dense enough that a 0.25 s grid adds views beyond the TSDF color keyframes.
    source = write_synthetic_capture(tmp_path / "room", SyntheticCaptureSpec(frames=60, color_size=96, depth_size=64))
    tsdf_only = compact_session(source, tmp_path / "tsdf_only", policy=KeyframePolicy(splat_color_gap_s=None))
    with_grid = compact_session(source, tmp_path / "with_grid", policy=KeyframePolicy(splat_color_gap_s=0.25))
    assert with_grid.frames_after["color_left"] > tsdf_only.frames_after["color_left"]
    assert with_grid.frames_after["color_left"] <= with_grid.frames_before["color_left"]
    assert with_grid.frames_after["color_left"] == with_grid.frames_after["color_right"]


def test_compaction_keeps_every_frame_in_a_detail_window_and_the_room_model(tmp_path):
    pytest.importorskip("warp")
    import pandas as pd

    from mq3drecon.synthetic import SyntheticCaptureSpec, write_synthetic_capture
    from mq3drecon.workflows.compaction import compact_session

    source = write_synthetic_capture(tmp_path / "room", SyntheticCaptureSpec(frames=60, color_size=96, depth_size=64))
    times = sorted(pd.read_csv(source / "left_camera_mruk_frame_metadata.csv").timestamp_us_realtime.astype(int))
    window = (times[20], times[40])  # "detail here": 21 left frames
    (source / "markers.json").write_text(json.dumps([{"kind": "detail", "start_us": window[0], "end_us": window[1]}]))
    (source / "room_model.json").write_text(json.dumps({"version": 1, "anchors": []}))
    compact_session(source, tmp_path / "compact", policy=KeyframePolicy(splat_color_gap_s=None))
    kept = set(pd.read_csv(tmp_path / "compact" / "left_camera_mruk_frame_metadata.csv").timestamp_us_realtime.astype(int))
    assert all(t in kept for t in times[20:41])
    assert (tmp_path / "compact" / "room_model.json").exists() and (tmp_path / "compact" / "markers.json").exists()
