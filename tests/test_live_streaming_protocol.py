import io
import struct

import numpy as np
import pytest

from mq3drecon.models import Side
from mq3drecon.streaming.bounded_queue import DropOldestQueue
from mq3drecon.streaming.framing import MAGIC, PREFIX, StreamProtocolError, encode_frame, read_frame
from mq3drecon.streaming.messages import (
    Bye,
    ColorFrame,
    DepthFrame,
    Heartbeat,
    Hello,
    MessageType,
    Pose,
    SessionEnd,
    SessionStart,
    decode_message,
    encode_message,
)
from mq3drecon.streaming.payloads import decode_color_rgba, decode_depth_ndc, encode_color_jpeg, encode_depth_ndc
from mq3drecon.streaming.session_recorder import nearest_timestamp_pairs

POSE = Pose(position=(1.0, 1.6, -0.5), rotation=(0.0, 0.3826834, 0.0, 0.9238795))


def _roundtrip(message):
    raw = read_frame(io.BytesIO(encode_message(message)))
    return decode_message(raw)


def _depth_frame(encoding="f32", data=None):
    ndc = np.linspace(0.5, 0.99, 12, dtype=np.float32).reshape(3, 4)
    return DepthFrame(
        side=Side.RIGHT, timestamp_ms=1_781_849_907_015, ovr_timestamp=12.5, width=4, height=3,
        encoding=encoding, create_pose=POSE, fov_left=1.0, fov_right=1.1, fov_top=0.9, fov_down=0.8,
        near_z=0.1, far_z=float("inf"), data=data if data is not None else encode_depth_ndc(ndc, encoding),
    )


def test_all_message_types_round_trip():
    color = ColorFrame(
        side=Side.LEFT, frame_index=7, timestamp_us=1_781_849_907_000_250, unix_time_ms=1_781_849_907_000,
        width=2, height=2, encoding="rgba", row_order="bottom_up", pose=POSE, pose_ok=True, data=bytes(range(16)),
    )
    messages = [
        Hello(device="Quest 3", app_version="1.6.0-live"),
        SessionStart(session_name="20261106_120000", intrinsics={Side.LEFT: {"focalLength": {"x": 1.0, "y": 2.0}}}),
        color,
        _depth_frame(),
        Heartbeat(time_ms=5, queued={"color": 1}, dropped={"depth": 2}),
        SessionEnd(reason="user_stop", dropped={"color": 3}),
        Bye(reason="app_quit"),
    ]
    for message in messages:
        assert _roundtrip(message) == message


def test_infinite_far_plane_survives_json():
    decoded = _roundtrip(_depth_frame())
    assert np.isinf(decoded.far_z)


def test_stream_of_messages_then_clean_eof():
    stream = io.BytesIO(encode_message(Hello("a", "b")) + encode_message(Bye()))
    assert decode_message(read_frame(stream)) == Hello("a", "b")
    assert decode_message(read_frame(stream)) == Bye()
    assert read_frame(stream) is None


def test_bad_magic_version_and_truncation_are_protocol_errors():
    good = encode_message(Hello("a", "b"))
    with pytest.raises(StreamProtocolError, match="Bad magic"):
        read_frame(io.BytesIO(b"XXXX" + good[4:]))
    wrong_version = PREFIX.pack(MAGIC, 99, int(MessageType.HELLO), 2, 0) + b"{}"
    with pytest.raises(StreamProtocolError, match="Unsupported protocol version 99"):
        read_frame(io.BytesIO(wrong_version))
    with pytest.raises(StreamProtocolError, match="mid-message"):
        read_frame(io.BytesIO(good[:-3]))


def test_unknown_type_and_bad_payload_sizes_are_rejected():
    with pytest.raises(StreamProtocolError, match="Unknown message type"):
        decode_message(read_frame(io.BytesIO(encode_frame(200, {}))))
    with pytest.raises(StreamProtocolError, match="Depth payload"):
        _roundtrip(_depth_frame(data=b"\x00" * 7))


def test_partial_socket_reads_are_reassembled():
    data = encode_message(_depth_frame())

    class Trickle(io.RawIOBase):
        def __init__(self):
            self.pos = 0

        def read(self, size=-1):
            chunk = data[self.pos:self.pos + min(3, size)]  # never more than requested, like a socket
            self.pos += len(chunk)
            return chunk

    assert decode_message(read_frame(Trickle())) == _depth_frame()


@pytest.mark.parametrize("encoding,tolerance", [("f32", 0.0), ("u16_ndc", 1.0 / 65535), ("f16", 1e-3)])
def test_depth_encodings(encoding, tolerance):
    ndc = np.array([[0.0, 0.5, 0.98, 1.0]], dtype=np.float32)
    frame = _depth_frame(encoding=encoding, data=encode_depth_ndc(ndc, encoding))
    frame = DepthFrame(**{**frame.__dict__, "width": 4, "height": 1})
    assert np.allclose(decode_depth_ndc(frame), ndc, atol=tolerance)


def test_jpeg_color_round_trip_keeps_orientation():
    rgba = np.zeros((8, 16, 4), np.uint8)
    rgba[:4, :, 0] = 255  # red top half
    frame = ColorFrame(Side.LEFT, 0, 1, 0, 16, 8, "jpeg", "top_down", POSE, True, encode_color_jpeg(rgba, 95))
    decoded = decode_color_rgba(frame)
    assert decoded[1, 8, 0] > 200 and decoded[6, 8, 0] < 50


def test_drop_oldest_queue():
    q = DropOldestQueue[int](2)
    assert q.put(1) and q.put(2)
    assert not q.put(3)
    assert q.dropped == 1
    assert [q.get(0), q.get(0), q.get(0)] == [2, 3, None]
    q.close()
    assert not q.put(4)


def test_nearest_timestamp_pairs_is_one_to_one():
    assert nearest_timestamp_pairs([0, 100, 200], [5, 104, 400]) == [(0, 0), (1, 1), (2, 2)]
    assert nearest_timestamp_pairs([0, 10], [9]) == [(1, 0)]
    left = list(range(0, 10_000, 100))
    right = [t + 7 for t in left]
    assert nearest_timestamp_pairs(left, right) == [(i, i) for i in range(len(left))]
