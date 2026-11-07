"""Room model from Space Setup anchors: geometry with openings, spawn at the entrance, exports, frame."""

import json
import math
import xml.etree.ElementTree as ET

import numpy as np

from mq3drecon.processing.web_location.room_model import (
    UNITY_TO_WORLD, build_room_geometry, entrance_spawn, export_room_model, load_room_model, parse_room_model,
    space_setup_floor_y, triangulate_polygon,
)

S = math.sqrt(0.5)
DOWN_TO_UP = [-S, 0.0, 0.0, S]  # -90 deg about X: local +Z (plane normal) -> world +Y


def _room_json(entrance=True):
    """4 x 3 m room (Unity: x -2..2, z -1.5..1.5), 2.5 m high, a door on the z = -1.5 wall, a table."""
    walls = [  # MRUK planes: local XY, normal local +Z facing into the room
        ("w-south", [0, 1.25, -1.5], [0, 0, 0, 1], 2.0),
        ("w-north", [0, 1.25, 1.5], [0, 1, 0, 0], 2.0),  # 180 deg about Y
        ("w-west", [-2, 1.25, 0], [0, S, 0, S], 1.5),  # +90 deg about Y: +Z -> +X
        ("w-east", [2, 1.25, 0], [0, -S, 0, S], 1.5),
    ]
    anchors = [{"uuid": u, "labels": ["WALL_FACE"], "pose": {"position": p, "rotation": q}, "plane_rect": [-h, -1.25, h, 1.25]}
               for u, p, q, h in walls]
    anchors.append({"uuid": "floor", "labels": ["FLOOR"], "pose": {"position": [0, 0, 0], "rotation": DOWN_TO_UP},
                    "plane_rect": [-2, -1.5, 2, 1.5], "plane_boundary": [[-2, -1.5], [2, -1.5], [2, 1.5], [-2, 1.5]]})
    anchors.append({"uuid": "door", "labels": ["DOOR_FRAME"], "pose": {"position": [0.5, 1.0, -1.5], "rotation": [0, 0, 0, 1]},
                    "plane_rect": [-0.45, -1.0, 0.45, 1.0]})
    anchors.append({"uuid": "table", "labels": ["TABLE"], "pose": {"position": [0, 0.75, 0], "rotation": [0, 0, 0, 1]},
                    "volume": {"min": [-0.6, -0.4, -0.75], "max": [0.6, 0.4, 0.0]}})
    return {"version": 1, "anchors": anchors, "entrance": {"anchor_uuid": "door"} if entrance else None}


def _area(part):
    v, f = part.vertices, part.faces
    return float(np.sum(np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)) / 2)


def test_walls_get_door_openings_floor_and_furniture():
    parts = build_room_geometry(parse_room_model(_room_json()))
    walls = [p for p in parts if p.kind == "wall"]
    assert len(walls) == 4
    areas = sorted(_area(w) for w in walls)
    assert math.isclose(areas[-1], 4 * 2.5, rel_tol=1e-6)  # the north wall, no opening
    assert any(math.isclose(a, 4 * 2.5 - 0.9 * 2.0, rel_tol=1e-6) for a in areas)  # south wall minus the door
    floor = next(p for p in parts if p.kind == "floor")
    assert math.isclose(_area(floor), 12.0, rel_tol=1e-6)
    assert np.allclose(floor.vertices[:, 1], 0.0)
    assert [p.kind for p in parts].count("entrance") == 1 and any(p.label == "TABLE" for p in parts)


def test_entrance_spawn_steps_into_the_room_and_faces_inward():
    model = parse_room_model(_room_json())
    (x, y, z), yaw = entrance_spawn(model, floor_y=space_setup_floor_y(model))
    # door at Unity z = -1.5 -> world z = +1.5; into the room is world -z
    assert math.isclose(x, 0.5, abs_tol=1e-6) and math.isclose(y, 0.0, abs_tol=1e-6) and math.isclose(z, 0.9, abs_tol=1e-6)
    assert math.isclose(yaw, 0.0, abs_tol=1e-6)  # yaw 0 looks along world -z (viewer convention)
    assert entrance_spawn(parse_room_model(_room_json(entrance=False)), 0.0) is None


def test_exports_are_readable(tmp_path):
    (tmp_path / "room_model.json").write_text(json.dumps(_room_json()))
    model = load_room_model(tmp_path)
    info = export_room_model(model, tmp_path)
    assert info["elements"]["wall"] == 4 and info["elements"]["furniture"] == 1
    obj = (tmp_path / "room_model.obj").read_text()
    assert "g wall_1" in obj and "g entrance_1" in obj and "mtllib room_model.mtl" in obj
    root = ET.parse(tmp_path / "room_model.dae").getroot()
    ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    assert len(root.findall(".//c:geometry", ns)) == len(build_room_geometry(model))
    assert (tmp_path / "room_model.glb").read_bytes()[:4] == b"glTF"


def test_ear_clipping_handles_an_l_shaped_room():
    outline = np.array([[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]], float)  # area 12
    for points in (outline, outline[::-1]):
        tris = triangulate_polygon(points)
        def twice_area(a, b, c):
            (ux, uy), (vx, vy) = points[b] - points[a], points[c] - points[a]
            return abs(ux * vy - uy * vx)

        area = sum(twice_area(a, b, c) / 2 for a, b, c in tris)
        assert len(tris) == 4 and math.isclose(area, 12.0)


def test_room_points_use_the_same_frame_as_camera_poses():
    from mq3drecon.processing.live.native_depth import unity_pose_to_open3d
    from mq3drecon.streaming.messages import Pose

    unity = np.array([0.3, 1.6, -2.0])
    camera_to_world, _ = unity_pose_to_open3d(Pose(position=tuple(unity), rotation=(0.0, 0.0, 0.0, 1.0)))
    assert np.allclose(camera_to_world[:3, 3], unity @ UNITY_TO_WORLD.T)


def test_room_model_and_markers_travel_over_the_wire_and_into_the_session(tmp_path):
    import io

    from mq3drecon.streaming.framing import read_frame
    from mq3drecon.streaming.messages import Marker, RoomModelUpdate, SessionEnd, SessionStart, decode_message, encode_message
    from mq3drecon.workflows.live_session import LiveSessionController

    room = _room_json()
    marker = Marker(kind="detail", data={"start_us": 10, "end_us": 2_000_010})
    for message in (RoomModelUpdate(room=room), marker):
        assert decode_message(read_frame(io.BytesIO(encode_message(message)))) == message

    controller = LiveSessionController(tmp_path / "sessions")
    controller.on_message(RoomModelUpdate(room=room))  # sent when the app loads the room, before recording
    controller.on_message(SessionStart(session_name="s1", intrinsics={}))
    controller.on_message(marker)
    controller.on_message(SessionEnd())
    session = tmp_path / "sessions" / "s1"
    model = load_room_model(session)
    assert model is not None and model.source_version == 1 and len(model.regions) == 1 and len(model.openings) == 1
    assert json.loads((session / "markers.json").read_text()) == [{"kind": "detail", "start_us": 10, "end_us": 2_000_010}]



# ---------------------------------------------------------------- schema v2: regions with their own floor and ceiling

def _v2(regions, vertices, openings=(), boxes=()):
    return parse_room_model({"version": 2, "vertices": vertices, "regions": regions, "openings": list(openings), "boxes": list(boxes)})


def _kind_area(parts, kind):
    return sum(_area(p) for p in parts if p.kind == kind)


SQUARE = [[0, 0], [4, 0], [4, 4], [0, 4]]  # 4 x 4 m room, Unity plan (x, z)


def test_false_ceiling_over_half_the_room_makes_a_bulkhead():
    vertices = [[0, 0], [2, 0], [4, 0], [4, 4], [2, 4], [0, 4]]
    model = _v2([
        {"loop": [0, 1, 4, 5], "floor": {"y": 0}, "ceiling": {"y": 2.7}},
        {"loop": [1, 2, 3, 4], "floor": {"y": 0}, "ceiling": {"y": 2.4}},  # false ceiling 30 cm lower
    ], vertices)
    parts = build_room_geometry(model)
    assert math.isclose(_kind_area(parts, "bulkhead"), 4 * 0.3, rel_tol=1e-6)  # along the shared 4 m edge
    # walls follow each region's own ceiling: x=0 side 4 x 2.7, x=4 side 4 x 2.4, front/back halves 2 x 2.7 + 2 x 2.4
    assert math.isclose(_kind_area(parts, "wall"), 4 * 2.7 + 4 * 2.4 + 2 * (2 * 2.7 + 2 * 2.4), rel_tol=1e-6)
    assert math.isclose(_kind_area(parts, "ceiling"), 16.0, rel_tol=1e-6) and _kind_area(parts, "step") == 0


def test_beam_is_a_ceiling_strip_pushed_down_with_two_sides():
    # beam 0.3 m wide across the room at z = 2, 0.25 m deep
    vertices = [[0, 0], [4, 0], [4, 1.85], [4, 2.15], [4, 4], [0, 4], [0, 2.15], [0, 1.85]]
    model = _v2([
        {"loop": [0, 1, 2, 7], "floor": {"y": 0}, "ceiling": {"y": 2.6}},
        {"loop": [7, 2, 3, 6], "floor": {"y": 0}, "ceiling": {"y": 2.35}},  # the beam
        {"loop": [6, 3, 4, 5], "floor": {"y": 0}, "ceiling": {"y": 2.6}},
    ], vertices)
    parts = build_room_geometry(model)
    bulkheads = [p for p in parts if p.kind == "bulkhead"]
    assert len(bulkheads) == 2 and math.isclose(_kind_area(parts, "bulkhead"), 2 * 4 * 0.25, rel_tol=1e-6)
    assert math.isclose(_kind_area(parts, "floor"), 16.0, rel_tol=1e-6)


def test_sunken_area_gets_a_step_riser_and_its_walls_go_down_with_it():
    vertices = [[0, 0], [4, 0], [4, 1], [4, 4], [0, 4], [0, 1]]
    model = _v2([
        {"loop": [0, 1, 2, 5], "floor": {"y": -0.17}, "ceiling": {"y": 2.5}},  # one step down
        {"loop": [5, 2, 3, 4], "floor": {"y": 0.0}, "ceiling": {"y": 2.5}},
    ], vertices)
    parts = build_room_geometry(model)
    assert math.isclose(_kind_area(parts, "step"), 4 * 0.17, rel_tol=1e-6)
    sunken_wall = max((p for p in parts if p.kind == "wall"), key=lambda p: -p.vertices[:, 1].min())
    assert math.isclose(sunken_wall.vertices[:, 1].min(), -0.17, abs_tol=1e-9)


def test_pillar_is_a_solid_region_with_four_walls_and_a_hole_in_the_floor():
    vertices = SQUARE + [[1.8, 1.8], [2.2, 1.8], [2.2, 2.2], [1.8, 2.2]]
    model = _v2([
        {"loop": [0, 1, 2, 3], "holes": [[4, 5, 6, 7]], "floor": {"y": 0}, "ceiling": {"y": 2.5}},
        {"loop": [4, 5, 6, 7], "solid": True, "floor": {"y": 0}, "ceiling": {"y": 2.5}},
    ], vertices)
    parts = build_room_geometry(model)
    assert len([p for p in parts if p.kind == "wall"]) == 8  # 4 room walls + 4 pillar faces
    assert math.isclose(_kind_area(parts, "wall"), 4 * 4 * 2.5 + 4 * 0.4 * 2.5, rel_tol=1e-6)
    assert math.isclose(_kind_area(parts, "floor"), 16.0 - 0.16, rel_tol=1e-6)


def test_sloped_ceiling_makes_trapezoid_walls():
    model = _v2([{"loop": [0, 1, 2, 3], "floor": {"y": 0}, "ceiling": {"y": 2.0, "ref": [0, 0], "slope": [0.25, 0.0]}}], SQUARE)
    parts = build_room_geometry(model)
    # ceiling rises from 2.0 m at x=0 to 3.0 m at x=4: walls 2x(4*2.5 trapezoid) + 4*2.0 + 4*3.0
    assert math.isclose(_kind_area(parts, "wall"), 2 * 4 * 2.5 + 4 * 2.0 + 4 * 3.0, rel_tol=1e-6)


def test_v2_entrance_spawn_uses_its_region_floor():
    model = _v2([{"loop": [0, 1, 2, 3], "floor": {"y": -0.17}, "ceiling": {"y": 2.5}}], SQUARE,
                openings=[{"edge": [0, 1], "offset": 1.5, "width": 0.9, "sill": 0, "height": 2.0, "kind": "door", "entrance": True}])
    (x, y, z), yaw = entrance_spawn(model)
    assert math.isclose(x, 1.95) and math.isclose(y, -0.17) and math.isclose(z, -0.6)  # world z = -(unity z 0.6)
    parts = build_room_geometry(model)
    assert math.isclose(_kind_area(parts, "wall"), 4 * 4 * (2.5 + 0.17) - 0.9 * 2.0, rel_tol=1e-6)  # walls 2.67 m tall
