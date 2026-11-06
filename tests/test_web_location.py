import json
import threading
import urllib.request

import numpy as np
import pytest

from mq3drecon.processing.web_location.color_fill import fill_uncolored
from mq3drecon.processing.web_location.glb import read_glb, write_glb


def test_glb_round_trip_is_exact_and_binary(tmp_path):
    rng = np.random.default_rng(0)
    vertices = rng.random((500, 3)).astype(np.float32)
    triangles = rng.integers(0, 500, (900, 3))
    colors = rng.random((500, 3))
    size = write_glb(tmp_path / "m.glb", vertices, triangles, colors)
    data = (tmp_path / "m.glb").read_bytes()
    assert len(data) == size and data[:4] == b"glTF" and b"BIN\x00" in data and b"base64" not in data
    v, t, c = read_glb(tmp_path / "m.glb")
    assert np.array_equal(v, vertices) and np.array_equal(t, triangles) and np.abs(c - colors).max() <= 0.5 / 255 + 1e-6


def test_color_fill_reaches_only_nearby_uncolored_vertices():
    # A strip of 10 vertices; only vertex 0 is colored. Three rings reach vertices 1-3, not 9.
    vertices = np.stack([np.arange(10.0), np.zeros(10), np.zeros(10)], axis=1)
    triangles = np.array([[i, i + 1, i + 1] for i in range(9)])
    colors = np.full((10, 3), 0.6)
    colors[0] = [1.0, 0.0, 0.0]
    filled = fill_uncolored(vertices, triangles, colors, np.arange(10) == 0, rings=3)
    assert np.allclose(filled[1:4], [1.0, 0.0, 0.0]) and np.allclose(filled[9], 0.6)


def test_floor_prior_rejects_furniture_and_uses_a_captured_floor():
    o3d = pytest.importorskip("open3d")
    from mq3drecon.processing.web_location.mesh_export import estimate_floor_y

    def plane(y):
        mesh = o3d.geometry.TriangleMesh.create_box(4.0, 0.001, 4.0)
        mesh.translate((-2.0, y, -2.0))
        return mesh.subdivide_midpoint(number_of_iterations=4)

    head = np.full(50, 1.75)
    assert estimate_floor_y(plane(0.55), head) == (0.0, "tracking_origin")  # a bed top is not the floor
    floor_y, source = estimate_floor_y(plane(0.04), head)
    assert source == "captured_floor" and abs(floor_y - 0.04) < 0.01
    assert estimate_floor_y(plane(0.55), np.full(50, 0.9))[1] == "lowest_horizontal_surface"  # no standing prior


def test_server_serves_viewer_and_locations_and_blocks_traversal(tmp_path):
    from mq3drecon.web_viewer.server import make_server

    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "location.json").write_text('{"ok": true}')
    (tmp_path.parent / "secret.txt").write_text("nope")
    server = make_server(tmp_path, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert b"VR Locations" in urllib.request.urlopen(base + "/").read()
        assert json.loads(urllib.request.urlopen(base + "/locations/a/location.json").read()) == {"ok": True}
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/locations/../secret.txt")
    finally:
        server.shutdown()


def test_build_location_from_synthetic_capture(tmp_path):
    pytest.importorskip("warp")
    pytest.importorskip("open3d")
    from mq3drecon.synthetic import SyntheticCaptureSpec, write_synthetic_capture
    from mq3drecon.workflows.web_location import build_location

    project = write_synthetic_capture(tmp_path / "room", SyntheticCaptureSpec(frames=24, color_size=160, depth_size=96, color_format="jpg"))
    result = build_location(project, workspace=tmp_path / "ws", name="Synthetic Room", visual_triangles=50_000, collider_triangles=10_000)
    loc = result.location
    assert loc["id"] == "synthetic-room" and loc["format"] == "mq3drecon-web-location"
    assert loc["floor_source"] == "captured_floor" and abs(loc["floor_y"]) < 0.03  # the synthetic floor is y = 0
    for name in ("visual.glb", "collider.glb", "location.json"):
        assert (result.location_dir / name).exists()
    index = json.loads((tmp_path / "ws" / "locations" / "index.json").read_text())
    assert [e["id"] for e in index["locations"]] == ["synthetic-room"]
