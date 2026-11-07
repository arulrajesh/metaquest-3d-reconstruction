"""Basic geometric room model (SketchUp-style massing) from Meta Space Setup, sent by the headset.

The capture app loads the room with MRUK and sends ``ROOM_MODEL`` (stored as ``room_model.json``): every
anchor's semantic labels, Unity world pose, plane rectangle or outline (local XY, normal +Z) and volume bounds
(local), plus the entrance door the operator picked. Unity world is the frame of the camera poses, so the
room lines up with the scan after the same Unity -> MQ3DRecon conversion (flip Z).

From it the build writes walls with door and window openings, floor and ceiling, door/window panels and
furniture boxes as ``room_model.glb`` (web), ``.obj`` + ``.mtl`` and ``.dae`` (COLLADA, SketchUp's import
format), each element its own named group; and it uses the entrance for the visitor spawn and Space Setup's
floor for ``floor_y``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

ROOM_MODEL_FILE = "room_model.json"
UNITY_TO_WORLD = np.diag([1.0, 1.0, -1.0])  # Unity (left-handed, Y up) -> MQ3DRecon / Open3D world

WALL_LABELS = ("WALL_FACE", "INNER_WALL_FACE")
OPENING_LABELS = ("DOOR_FRAME", "WINDOW_FRAME")
FURNITURE_COLORS = {
    "TABLE": (0.80, 0.62, 0.42), "COUCH": (0.55, 0.62, 0.80), "BED": (0.72, 0.60, 0.80), "STORAGE": (0.70, 0.55, 0.40),
    "SCREEN": (0.25, 0.25, 0.28), "LAMP": (0.95, 0.85, 0.45), "PLANT": (0.45, 0.75, 0.45),
}
PART_COLORS = {
    "wall": (0.88, 0.88, 0.86), "floor": (0.62, 0.60, 0.57), "ceiling": (0.95, 0.95, 0.95),
    "door": (0.55, 0.38, 0.22), "window": (0.60, 0.80, 0.95), "furniture": (0.70, 0.70, 0.70), "entrance": (0.30, 0.80, 0.40),
}


def quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@dataclass(frozen=True)
class RoomAnchor:
    uuid: str
    labels: tuple[str, ...]
    rotation: np.ndarray  # (3, 3) local -> Unity world
    position: np.ndarray  # (3,) Unity world
    plane_rect: tuple[float, float, float, float] | None = None  # xmin, ymin, xmax, ymax in local XY
    plane_boundary: np.ndarray | None = None  # (K, 2) local XY
    volume: tuple[np.ndarray, np.ndarray] | None = None  # local min, max

    @property
    def label(self) -> str:
        return self.labels[0] if self.labels else "OTHER"

    def to_unity(self, local: np.ndarray) -> np.ndarray:
        return np.asarray(local, float) @ self.rotation.T + self.position

    def to_world(self, local: np.ndarray) -> np.ndarray:
        return self.to_unity(local) @ UNITY_TO_WORLD.T

    def from_unity(self, unity_points: np.ndarray) -> np.ndarray:
        return (np.asarray(unity_points, float) - self.position) @ self.rotation


@dataclass(frozen=True)
class RoomModel:
    anchors: list[RoomAnchor]
    entrance: dict | None = None  # {"anchor_uuid", "position": Unity world [x, y, z]}

    def with_labels(self, *labels: str) -> list[RoomAnchor]:
        return [a for a in self.anchors if any(label in a.labels for label in labels)]

    def anchor(self, uuid: str | None) -> RoomAnchor | None:
        return next((a for a in self.anchors if a.uuid == uuid), None) if uuid else None


def parse_room_model(data: dict) -> RoomModel:
    anchors = []
    for item in data.get("anchors", []):
        pose = item.get("pose", {})
        rect = item.get("plane_rect")
        boundary = item.get("plane_boundary")
        volume = item.get("volume")
        anchors.append(RoomAnchor(
            uuid=str(item.get("uuid", "")),
            labels=tuple(str(label) for label in item.get("labels", [])),
            rotation=quaternion_matrix(*[float(v) for v in pose.get("rotation", [0, 0, 0, 1])]),
            position=np.asarray(pose.get("position", [0, 0, 0]), float),
            plane_rect=tuple(float(v) for v in rect) if rect else None,  # type: ignore[arg-type]
            plane_boundary=np.asarray(boundary, float).reshape(-1, 2) if boundary else None,
            volume=(np.asarray(volume["min"], float), np.asarray(volume["max"], float)) if volume else None,
        ))
    return RoomModel(anchors=anchors, entrance=data.get("entrance"))


def load_room_model(project_dir: Path) -> RoomModel | None:
    path = Path(project_dir) / ROOM_MODEL_FILE
    if not path.exists():
        return None
    model = parse_room_model(json.loads(path.read_text(encoding="utf-8")))
    return model if model.anchors else None


# ---------------------------------------------------------------- geometry

@dataclass
class RoomPart:
    name: str
    kind: str  # wall, floor, ceiling, door, window, furniture, entrance
    vertices: np.ndarray  # (N, 3) MQ3DRecon world
    faces: np.ndarray  # (M, 3)
    color: tuple[float, float, float] = (0.8, 0.8, 0.8)
    label: str = ""


def _quad(corners: np.ndarray) -> np.ndarray:
    return np.array([[0, 1, 2], [0, 2, 3]])


def _box(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corners = np.array([[x, y, z] for z in (lo[2], hi[2]) for y in (lo[1], hi[1]) for x in (lo[0], hi[0])], float)
    faces = np.array([
        [0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6],  # -z, +z
        [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7],  # -y, +y
        [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5],  # -x, +x
    ])
    return corners, faces


def triangulate_polygon(points: np.ndarray) -> np.ndarray:
    """Ear clipping for a simple 2D polygon (either winding). Returns (K, 3) indices."""
    pts = np.asarray(points, float)
    n = len(pts)
    if n < 3:
        return np.zeros((0, 3), int)
    signed = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    order = list(range(n)) if signed > 0 else list(range(n))[::-1]  # make it counter-clockwise

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def inside(p, a, b, c):
        return cross(a, b, p) >= 0 and cross(b, c, p) >= 0 and cross(c, a, p) >= 0

    triangles = []
    guard = 0
    while len(order) > 3 and guard < 10 * n:
        guard += 1
        for k in range(len(order)):
            i, j, l = order[k - 1], order[k], order[(k + 1) % len(order)]
            a, b, c = pts[i], pts[j], pts[l]
            if cross(a, b, c) <= 1e-12:
                continue  # reflex or degenerate
            if any(inside(pts[m], a, b, c) for m in order if m not in (i, j, l)):
                continue
            triangles.append([i, j, l])
            order.pop(k)
            break
        else:
            break  # not simple: fall back to a fan for the rest
    if len(order) >= 3:
        triangles += [[order[0], order[k], order[k + 1]] for k in range(1, len(order) - 1)]
    return np.asarray(triangles, int)


def room_center(model: RoomModel) -> np.ndarray:
    """Unity-world centre of the room: the floor outline's centroid, else the walls' mean."""
    floors = model.with_labels("FLOOR")
    if floors and floors[0].plane_boundary is not None and len(floors[0].plane_boundary) >= 3:
        b = floors[0].plane_boundary
        return floors[0].to_unity(np.c_[b, np.zeros(len(b))]).mean(axis=0)
    walls = model.with_labels(*WALL_LABELS)
    if walls:
        return np.mean([w.position for w in walls], axis=0)
    return np.mean([a.position for a in model.anchors], axis=0)


def _inward_normal(anchor: RoomAnchor, center_unity: np.ndarray) -> np.ndarray:
    """Unity-world plane normal pointing into the room (MRUK planes face local +Z; do not rely on the sign)."""
    normal = anchor.rotation @ np.array([0.0, 0.0, 1.0])
    return normal if np.dot(center_unity - anchor.position, normal) >= 0 else -normal


def _openings_on(wall: RoomAnchor, openings: list[RoomAnchor]) -> list[tuple[float, float, float, float]]:
    """Door/window rectangles lying on ``wall``, in the wall's local XY, clipped to the wall."""
    xmin, ymin, xmax, ymax = wall.plane_rect  # type: ignore[misc]
    result = []
    for opening in openings:
        if opening.plane_rect is None:
            continue
        ox0, oy0, ox1, oy1 = opening.plane_rect
        corners = opening.to_unity(np.array([[ox0, oy0, 0], [ox1, oy0, 0], [ox1, oy1, 0], [ox0, oy1, 0]], float))
        local = wall.from_unity(corners)
        if np.max(np.abs(local[:, 2])) > 0.25:
            continue  # not on this wall
        lx0, ly0 = max(local[:, 0].min(), xmin), max(local[:, 1].min(), ymin)
        lx1, ly1 = min(local[:, 0].max(), xmax), min(local[:, 1].max(), ymax)
        if lx1 - lx0 > 0.05 and ly1 - ly0 > 0.05:
            result.append((lx0, ly0, lx1, ly1))
    return result


def build_room_geometry(model: RoomModel) -> list[RoomPart]:
    center = room_center(model)
    openings = model.with_labels(*OPENING_LABELS)
    entrance_uuid = (model.entrance or {}).get("anchor_uuid")
    parts: list[RoomPart] = []
    counters: dict[str, int] = {}

    def name(kind: str, label: str = "") -> str:
        key = (label or kind).lower().replace(" ", "_")
        counters[key] = counters.get(key, 0) + 1
        return f"{key}_{counters[key]}"

    for wall in model.with_labels(*WALL_LABELS):
        if wall.plane_rect is None:
            continue
        xmin, ymin, xmax, ymax = wall.plane_rect
        holes = _openings_on(wall, openings)
        xs = sorted({xmin, xmax, *[v for h in holes for v in (h[0], h[2])]})
        ys = sorted({ymin, ymax, *[v for h in holes for v in (h[1], h[3])]})
        vertices, faces = [], []
        for x0, x1 in zip(xs[:-1], xs[1:]):
            for y0, y1 in zip(ys[:-1], ys[1:]):
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                if any(h[0] <= cx <= h[2] and h[1] <= cy <= h[3] for h in holes):
                    continue  # this cell is a door or window opening
                base = len(vertices)
                vertices += [[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]]
                faces += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
        if faces:
            parts.append(RoomPart(name("wall"), "wall", wall.to_world(np.array(vertices, float)), np.array(faces), PART_COLORS["wall"], "WALL"))

    for label, kind in (("FLOOR", "floor"), ("CEILING", "ceiling")):
        for plane in model.with_labels(label):
            outline = plane.plane_boundary
            if outline is None or len(outline) < 3:
                if plane.plane_rect is None:
                    continue
                x0, y0, x1, y1 = plane.plane_rect
                outline = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
            faces = triangulate_polygon(outline)
            parts.append(RoomPart(name(kind), kind, plane.to_world(np.c_[outline, np.zeros(len(outline))]), faces, PART_COLORS[kind], label))

    for opening in openings:
        if opening.plane_rect is None:
            continue
        x0, y0, x1, y1 = opening.plane_rect
        inward = _inward_normal(opening, center)
        local_in = opening.rotation.T @ inward * 0.01  # 1 cm into the room, so it does not z-fight the wall
        corners = np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]], float) + local_in
        kind = "door" if "DOOR_FRAME" in opening.labels else "window"
        is_entrance = kind == "door" and opening.uuid == entrance_uuid
        parts.append(RoomPart(name("entrance" if is_entrance else kind), "entrance" if is_entrance else kind,
                              opening.to_world(corners), _quad(corners), PART_COLORS["entrance" if is_entrance else kind], opening.label))

    for item in model.anchors:
        if item.volume is None:
            continue
        corners, faces = _box(*item.volume)
        parts.append(RoomPart(name("furniture", item.label), "furniture", item.to_world(corners), faces,
                              FURNITURE_COLORS.get(item.label, PART_COLORS["furniture"]), item.label))
    return parts


# ---------------------------------------------------------------- spawn and floor

def entrance_spawn(model: RoomModel, floor_y: float, step_in_m: float = 0.6) -> tuple[tuple[float, float, float], float] | None:
    """(position, yaw_deg) in MQ3DRecon world: just inside the chosen entrance door, facing into the room."""
    entrance = model.entrance or {}
    door = model.anchor(entrance.get("anchor_uuid"))
    if door is None and entrance.get("position") is None:
        return None
    center = room_center(model)
    if door is not None:
        position = door.position.copy()
        inward = _inward_normal(door, center)
    else:
        position = np.asarray(entrance["position"], float)
        inward = center - position
    inward[1] = 0.0
    if np.linalg.norm(inward) < 1e-6:
        return None
    inward /= np.linalg.norm(inward)
    spot = (position + step_in_m * inward) @ UNITY_TO_WORLD.T
    facing = inward @ UNITY_TO_WORLD.T
    yaw = float(np.degrees(np.arctan2(-facing[0], -facing[2])))  # same convention as spawn_from_camera
    return (float(spot[0]), float(floor_y), float(spot[2])), yaw


def space_setup_floor_y(model: RoomModel) -> float | None:
    floors = model.with_labels("FLOOR")
    return float((floors[0].position @ UNITY_TO_WORLD.T)[1]) if floors else None


# ---------------------------------------------------------------- export

def _concat(parts: list[RoomPart]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices, faces, colors, offset = [], [], [], 0
    for part in parts:
        vertices.append(part.vertices)
        faces.append(part.faces + offset)
        colors.append(np.tile(part.color, (len(part.vertices), 1)))
        offset += len(part.vertices)
    return np.vstack(vertices), np.vstack(faces), np.vstack(colors)


def write_room_glb(parts: list[RoomPart], path: Path) -> int:
    from mq3drecon.processing.web_location.glb import write_glb

    vertices, faces, colors = _concat(parts)
    return write_glb(Path(path), vertices, faces, colors)


def write_room_obj(parts: list[RoomPart], path: Path) -> None:
    """Wavefront OBJ + MTL, one group per element (wall_1, door_2, table_1...), Y up, meters."""
    path = Path(path)
    mtl = path.with_suffix(".mtl")
    kinds = {}
    for part in parts:
        kinds.setdefault(f"{part.kind}_{part.label.lower() or part.kind}", part.color)
    mtl.write_text("".join(f"newmtl {k}\nKd {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\nd 1.0\n\n" for k, c in kinds.items()), encoding="utf-8")
    lines, offset = [f"# MQ3DRecon room model (Meta Space Setup), meters, Y up\nmtllib {mtl.name}\n"], 1
    for part in parts:
        lines.append(f"g {part.name}\nusemtl {part.kind}_{part.label.lower() or part.kind}\n")
        lines.append("".join(f"v {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in part.vertices))
        lines.append("".join(f"f {a + offset} {b + offset} {c + offset}\n" for a, b, c in part.faces))
        offset += len(part.vertices)
    path.write_text("".join(lines), encoding="utf-8")


def write_room_dae(parts: list[RoomPart], path: Path) -> None:
    """COLLADA 1.4.1 (SketchUp's import format): one node, geometry and material per element. Y up, meters."""
    effects, materials, geometries, nodes = [], [], [], []
    for i, part in enumerate(parts):
        r, g, b = part.color
        effects.append(f'<effect id="fx{i}"><profile_COMMON><technique sid="common"><lambert><diffuse>'
                       f'<color>{r:.3f} {g:.3f} {b:.3f} 1</color></diffuse></lambert></technique></profile_COMMON></effect>')
        materials.append(f'<material id="mat{i}" name="{escape(part.name)}"><instance_effect url="#fx{i}"/></material>')
        positions = " ".join(f"{v:.4f}" for v in part.vertices.reshape(-1))
        indices = " ".join(str(int(v)) for v in part.faces.reshape(-1))
        geometries.append(
            f'<geometry id="geo{i}" name="{escape(part.name)}"><mesh>'
            f'<source id="geo{i}-pos"><float_array id="geo{i}-pos-arr" count="{part.vertices.size}">{positions}</float_array>'
            f'<technique_common><accessor source="#geo{i}-pos-arr" count="{len(part.vertices)}" stride="3">'
            f'<param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>'
            f'<vertices id="geo{i}-vtx"><input semantic="POSITION" source="#geo{i}-pos"/></vertices>'
            f'<triangles material="m" count="{len(part.faces)}"><input semantic="VERTEX" source="#geo{i}-vtx" offset="0"/>'
            f'<p>{indices}</p></triangles></mesh></geometry>')
        nodes.append(f'<node id="node{i}" name="{escape(part.name)}"><instance_geometry url="#geo{i}"><bind_material><technique_common>'
                     f'<instance_material symbol="m" target="#mat{i}"/></technique_common></bind_material></instance_geometry></node>')
    xml = ('<?xml version="1.0" encoding="utf-8"?>\n'
           '<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">'
           '<asset><contributor><authoring_tool>MQ3DRecon</authoring_tool></contributor>'
           '<unit name="meter" meter="1"/><up_axis>Y_UP</up_axis></asset>'
           f'<library_effects>{"".join(effects)}</library_effects>'
           f'<library_materials>{"".join(materials)}</library_materials>'
           f'<library_geometries>{"".join(geometries)}</library_geometries>'
           f'<library_visual_scenes><visual_scene id="room" name="room">{"".join(nodes)}</visual_scene></library_visual_scenes>'
           '<scene><instance_visual_scene url="#room"/></scene></COLLADA>\n')
    Path(path).write_text(xml, encoding="utf-8")


def export_room_model(model: RoomModel, out_dir: Path) -> dict:
    """Write room_model.glb/.obj/.mtl/.dae into ``out_dir``; returns asset names and element counts."""
    parts = build_room_geometry(model)
    if not parts:
        return {}
    out_dir = Path(out_dir)
    write_room_glb(parts, out_dir / "room_model.glb")
    write_room_obj(parts, out_dir / "room_model.obj")
    write_room_dae(parts, out_dir / "room_model.dae")
    counts: dict[str, int] = {}
    for part in parts:
        counts[part.kind] = counts.get(part.kind, 0) + 1
    return {"assets": {"room_model": "room_model.glb", "room_model_obj": "room_model.obj", "room_model_dae": "room_model.dae"},
            "elements": counts}
