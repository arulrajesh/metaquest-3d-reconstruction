import json

import cv2
import numpy as np
import pytest

from mq3drecon.dataio import DataIO
from mq3drecon.dataio.mruk_frame_codec import read_mruk_frame_rgba, validate_mruk_frame_file
from mq3drecon.models import Side
from mq3drecon.utils.depth_utils import mirror_principal_point_x


def _gradient(width=8, height=6):
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, 0] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]  # red increases downward
    rgba[:, :, 3] = 255
    return rgba


def _write_jpg(path, rgba_top_down):
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(rgba_top_down, cv2.COLOR_RGBA2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
    assert ok
    path.write_bytes(buffer.tobytes())


def test_jpg_frames_are_top_down_and_rgba_frames_are_flipped(tmp_path):
    image = _gradient()
    _write_jpg(tmp_path / "1.jpg", image)
    np.flipud(image).tofile(tmp_path / "2.rgba")  # raw buffers are bottom-up

    jpg = read_mruk_frame_rgba(tmp_path / "1.jpg", 8, 6)
    raw = read_mruk_frame_rgba(tmp_path / "2.rgba", 8, 6)

    assert jpg.shape == raw.shape == (6, 8, 4)
    assert np.array_equal(raw, image)
    assert jpg[0, 0, 0] < 30 and jpg[-1, 0, 0] > 225  # same orientation as the source, no flip


def test_jpg_frame_size_mismatch_is_an_error(tmp_path):
    _write_jpg(tmp_path / "1.jpg", _gradient())
    with pytest.raises(ValueError, match="metadata says"):
        read_mruk_frame_rgba(tmp_path / "1.jpg", 16, 16)
    (tmp_path / "empty.jpg").write_bytes(b"")
    with pytest.raises(ValueError, match="Empty"):
        validate_mruk_frame_file(tmp_path / "empty.jpg", 8, 6)


def test_mruk_color_dataset_accepts_jpg_frames(tmp_path):
    (tmp_path / "session_info.json").write_text(json.dumps({"sessionFormatVersion": 2, "captureBackend": "MRUK"}))
    frame_dir = tmp_path / "left_camera_mruk_rgba"
    frame_dir.mkdir()
    _write_jpg(frame_dir / "1000.jpg", _gradient())
    (tmp_path / "left_camera_mruk_intrinsics.json").write_text(json.dumps({
        "backend": "MRUK", "imageFormat": "RGBA32",
        "focalLength": {"x": 5.0, "y": 5.0}, "principalPoint": {"x": 4.0, "y": 3.0},
        "resolution": {"width": 8, "height": 6},
    }))
    (tmp_path / "left_camera_mruk_frame_metadata.csv").write_text(
        "frame_index,file_name,unix_time_ms,timestamp_us_realtime,width,height,format,camera_position,"
        "pose_pos_x,pose_pos_y,pose_pos_z,pose_rot_x,pose_rot_y,pose_rot_z,pose_rot_w,pose_ok,is_playing,"
        "is_updated_this_frame,current_resolution_width,current_resolution_height,pixel_count,"
        "colors_array_length,rgba_byte_count,get_texture_ok,get_colors_ok,error\n"
        "1,1000.jpg,1,1000,8,6,JPEG,Left,0,0,0,0,0,0,1,true,true,true,8,6,48,48,192,true,true,\n"
    )

    data_io = DataIO(tmp_path)
    dataset = data_io.color.load_color_dataset(Side.LEFT, use_cache=False)
    rgb = data_io.color.load_color_rgb_image(dataset=dataset, index=0)

    assert list(dataset.image_file_names) == ["1000.jpg"]
    assert rgb.shape == (6, 8, 3)
    assert rgb[-1, 0, 0] > 225


def test_mirror_principal_point_matches_open3d_intrinsics_helper():
    pytest.importorskip("open3d")
    from mq3drecon.models.camera_dataset import CameraDataset
    from mq3drecon.models.transforms import CoordinateSystem, Transforms
    from mq3drecon.processing.reconstruction.utils.o3d_utils import compute_o3d_intrinsic_matrices

    dataset = CameraDataset(
        directory_relative_path="x", image_file_names=np.array(["a"]), timestamps=np.array([1]),
        fx=np.array([100.0]), fy=np.array([100.0]), cx=np.array([30.0]), cy=np.array([40.0]),
        transforms=Transforms(CoordinateSystem.UNITY, np.zeros((1, 3)), np.array([[0, 0, 0, 1.0]])),
        widths=np.array([200]), heights=np.array([100]),
    )
    assert compute_o3d_intrinsic_matrices(dataset)[0, 0, 2] == pytest.approx(mirror_principal_point_x(30.0, 200))
