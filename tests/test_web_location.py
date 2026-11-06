import json
import threading
import urllib.request

import numpy as np
import pytest

from mq3drecon.processing.web_location.color_fill import fill_uncolored
from mq3drecon.processing.web_location.glb import read_glb, write_glb


def test_glb_round_trip_is_binary_with_linear_vertex_colors(tmp_path):
    from mq3drecon.processing.web_location.glb import srgb_to_linear

    rng = np.random.default_rng(0)
    vertices = rng.random((500, 3)).astype(np.float32)
    triangles = rng.integers(0, 500, (900, 3))
    colors = rng.random((500, 3))
    size = write_glb(tmp_path / "m.glb", vertices, triangles, colors)
    data = (tmp_path / "m.glb").read_bytes()
    assert len(data) == size and data[:4] == b"glTF" and b"BIN\x00" in data and b"base64" not in data
    back = read_glb(tmp_path / "m.glb")
    assert np.array_equal(back["vertices"], vertices) and np.array_equal(back["triangles"], triangles)
    assert np.abs(back["colors"] - colors).max() < 2e-3  # sRGB -> linear uint16 -> sRGB
    assert abs(srgb_to_linear(np.array([0.5]))[0] - 0.214) < 1e-3  # glTF vertex colors are linear


def test_textured_glb_embeds_jpeg_and_unlit_material(tmp_path):
    import cv2

    ok, jpeg = cv2.imencode(".jpg", np.full((64, 64, 3), 128, np.uint8))
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    uvs = np.array([[0, 0], [1, 0], [0, 1]], np.float32)
    write_glb(tmp_path / "t.glb", vertices, np.array([[0, 1, 2]]), uvs=uvs, texture_jpeg=jpeg.tobytes())
    back = read_glb(tmp_path / "t.glb")
    assert back["texture_jpeg"] == jpeg.tobytes() and np.array_equal(back["uvs"], uvs) and back["colors"] is None
    assert b"KHR_materials_unlit" in (tmp_path / "t.glb").read_bytes()


def test_texture_bake_reproduces_detail_vertex_colors_cannot(tmp_path):
    pytest.importorskip("warp")
    pytest.importorskip("open3d")
    from dataclasses import replace

    from mq3drecon.processing.web_location.mesh_build import fuse_session
    from mq3drecon.processing.web_location.mesh_export import decimate, remove_debris, to_open3d
    from mq3drecon.processing.web_location.textured_visual import build_textured_visual, load_bake_views
    from mq3drecon.synthetic import SyntheticCaptureSpec, scene_color, write_synthetic_capture

    project = write_synthetic_capture(tmp_path / "room", SyntheticCaptureSpec(
        frames=30, color_size=480, depth_size=128, color_format="jpg", texture="checker"))
    built = fuse_session(project, voxel_size_m=0.03)
    colors = fill_uncolored(built.mesh.vertices, built.mesh.triangles, built.mesh.colors, built.mesh.color_valid)
    mesh = decimate(remove_debris(to_open3d(replace(built.mesh, colors=colors))), 40_000)
    v, f, vc = np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.vertex_colors)
    visual = build_textured_visual(v, f, vc, load_bake_views(project), texture_size=1024)
    assert visual.textured_fraction > 0.8

    import cv2

    texture = cv2.cvtColor(cv2.imdecode(np.frombuffer(visual.texture_jpeg, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    centroids_uv = visual.uvs[visual.triangles].mean(axis=1)
    centroids = visual.vertices[visual.triangles].mean(axis=1).astype(np.float64)
    truth = scene_color(centroids * np.array([1.0, 1.0, -1.0]), "checker")  # Open3D world -> Unity world
    px = np.clip((centroids_uv * 1024).astype(int), 0, 1023)
    baked_error = np.median(np.abs(texture[px[:, 1], px[:, 0]] / 255.0 - truth).mean(axis=1))
    # vertex-color baseline on the same (pre-unwrap) mesh
    vertex_error = np.median(np.abs(vc[f].mean(axis=1) - scene_color(v[f].mean(axis=1) * np.array([1.0, 1.0, -1.0]), "checker")).mean(axis=1))
    assert baked_error < 0.03 and baked_error < 0.4 * vertex_error


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
    result = build_location(project, workspace=tmp_path / "ws", name="Synthetic Room", visual_triangles=50_000,
                            collider_triangles=10_000, textured_triangles=30_000, texture_size=1024, splats=False)
    loc = result.location
    assert loc["id"] == "synthetic-room" and loc["format"] == "mq3drecon-web-location"
    assert loc["floor_source"] == "captured_floor" and abs(loc["floor_y"]) < 0.03  # the synthetic floor is y = 0
    for name in ("visual.glb", "collider.glb", "location.json"):
        assert (result.location_dir / name).exists()
    assert loc["stats"]["textured"] and read_glb(result.location_dir / "visual.glb")["texture_jpeg"]
    index = json.loads((tmp_path / "ws" / "locations" / "index.json").read_text())
    assert [e["id"] for e in index["locations"]] == ["synthetic-room"]
