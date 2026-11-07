"""Basic geometric room model (SketchUp-style massing) from the headset: Space Setup, edited in the app.

The capture app sends ``ROOM_MODEL`` (stored as ``room_model.json``). Schema v2 (``specs/CAPTURE_CONTROLS.md``)
is a **floor plan divided into regions**, each with its own floor and ceiling plane:

- ``vertices``: plan points ``[x, z]`` in Unity world (meters; the frame of the camera poses);
- ``regions``: ``loop`` (vertex indices), optional ``holes``, ``solid`` (pillar: not room space), and ``floor`` /
  ``ceiling`` planes ``{"y", "ref": [x, z], "slope": [dy/dx, dy/dz]}``;
- ``openings``: doors and windows on a wall edge ``[i, j]``: ``offset`` from vertex ``i``, ``width``, ``sill`` above
  the floor, ``height``, ``kind``, ``entrance``;
- ``boxes``: ``label``, ``center`` (Unity world), ``yaw_deg`` (Unity Euler Y), ``size`` ``[w, h, d]``, ``support``.

Walls, step risers and ceiling bulkheads are *derived*: a wall along every edge with room space on one side only,
a riser where two neighbouring regions' floors differ, a bulkhead where their ceilings differ. That covers false
ceilings, beams (a narrow ceiling strip pushed down), sunken floors, platforms, pillars and sloped ceilings.
Schema v1 (a list of MRUK anchors, the first headset build) is converted to v2 on load.

The build writes ``room_model.glb`` (web), ``.obj`` + ``.mtl`` and ``.dae`` (COLLADA, SketchUp's import format), each
element its own named group, and uses the entrance for the visitor spawn and the main floor for ``floor_y``.
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
WELD_M = 0.02  # vertices of different regions closer than this are the same plan point

FURNITURE_COLORS = {
    "TABLE": (0.80, 0.62, 0.42), "COUCH": (0.55, 0.62, 0.80), "BED": (0.72, 0.60, 0.80), "STORAGE": (0.70, 0.55, 0.40),
    "SCREEN": (0.25, 0.25, 0.28), "LAMP": (0.95, 0.85, 0.45), "PLANT": (0.45, 0.75, 0.45),
}
PART_COLORS = {
    "wall": (0.88, 0.88, 0.86), "floor": (0.62, 0.60, 0.57), "ceiling": (0.95, 0.95, 0.95), "step": (0.55, 0.53, 0.50),
    "bulkhead": (0.90, 0.90, 0.92), "door": (0.55, 0.38, 0.22), "window": (0.60, 0.80, 0.95),
    "furniture": (0.70, 0.70, 0.70), "entrance": (0.30, 0.80, 0.40),
}


# ---------------------------------------------------------------- model

@dataclass(frozen=True)
class Plane:
    """Horizontal or sloped plane: y = y0 + slope_x * (x - ref_x) + slope_z * (z - ref_z), Unity plan (x, z)."""

    y: float
    ref: tuple[float, float] = (0.0, 0.0)
    slope: tuple[float, float] = (0.0, 0.0)

    def at(self, x: float, z: float) -> float:
        return self.y + self.slope[0] * (x - self.ref[0]) + self.slope[1] * (z - self.ref[1])


@dataclass(frozen=True)
class Region:
    loop: list[int]
    floor: Plane
    ceiling: Plane
    holes: list[list[int]] = field(default_factory=list)
    solid: bool = False


@dataclass(frozen=True)
class Opening:
    edge: tuple[int, int]
    offset: float
    width: float
    sill: float
    height: float
    kind: str = "door"
    entrance: bool = False


@dataclass(frozen=True)
class Box:
    label: str
    center: tuple[float, float, float]  # Unity world
    yaw_deg: float  # Unity Euler Y: the width axis is (cos, 0, -sin) of it
    size: tuple[float, float, float]  # width, height, depth
    support: str = "floor"


@dataclass(frozen=True)
class RoomModel:
    vertices: np.ndarray  # (N, 2) Unity plan (x, z)
    regions: list[Region]
    openings: list[Opening] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)
    source_version: int = 2

    def point(self, index: int) -> np.ndarray:
        return self.vertices[index]


def _plane(data: dict | None, default_y: float) -> Plane:
    data = data or {}
    return Plane(float(data.get("y", default_y)), tuple(float(v) for v in data.get("ref", (0.0, 0.0))),  # type: ignore[arg-type]
                 tuple(float(v) for v in data.get("slope", (0.0, 0.0))))  # type: ignore[arg-type]


def parse_room_model(data: dict) -> RoomModel:
    if "regions" not in data and "anchors" in data:
        return _from_anchors_v1(data)
    vertices = np.asarray(data.get("vertices", []), float).reshape(-1, 2)
    regions = [Region(loop=[int(i) for i in r["loop"]], floor=_plane(r.get("floor"), 0.0), ceiling=_plane(r.get("ceiling"), 2.5),
                      holes=[[int(i) for i in h] for h in r.get("holes", [])], solid=bool(r.get("solid", False)))
               for r in data.get("regions", [])]
    openings = [Opening(edge=(int(o["edge"][0]), int(o["edge"][1])), offset=float(o["offset"]), width=float(o["width"]),
                        sill=float(o.get("sill", 0.0)), height=float(o["height"]), kind=str(o.get("kind", "door")),
                        entrance=bool(o.get("entrance", False))) for o in data.get("openings", [])]
    boxes = [Box(label=str(b.get("label", "OTHER")), center=tuple(float(v) for v in b["center"]),  # type: ignore[arg-type]
                 yaw_deg=float(b.get("yaw_deg", 0.0)), size=tuple(float(v) for v in b["size"]),  # type: ignore[arg-type]
                 support=str(b.get("support", "floor"))) for b in data.get("boxes", [])]
    return RoomModel(vertices=vertices, regions=regions, openings=openings, boxes=boxes, source_version=int(data.get("version", 2)))


def load_room_model(project_dir: Path) -> RoomModel | None:
    path = Path(project_dir) / ROOM_MODEL_FILE
    if not path.exists():
        return None
    model = parse_room_model(json.loads(path.read_text(encoding="utf-8")))
    return model if any(not r.solid for r in model.regions) else None


# ---------------------------------------------------------------- schema v1 (MRUK anchors) -> v2

def quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _anchor_frame(item: dict) -> tuple[np.ndarray, np.ndarray]:
    pose = item.get("pose", {})
    return quaternion_matrix(*[float(v) for v in pose.get("rotation", [0, 0, 0, 1])]), np.asarray(pose.get("position", [0, 0, 0]), float)


def _plane_outline(item: dict) -> np.ndarray | None:
    """Unity world 3D outline of a plane anchor (boundary, else rectangle)."""
    rotation, position = _anchor_frame(item)
    boundary = item.get("plane_boundary")
    if boundary and len(boundary) >= 3:
        local = np.asarray(boundary, float).reshape(-1, 2)
    elif item.get("plane_rect"):
        x0, y0, x1, y1 = (float(v) for v in item["plane_rect"])
        local = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    else:
        return None
    return np.c_[local, np.zeros(len(local))] @ rotation.T + position


def _inside(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, z = point
    inside = False
    for (x0, z0), (x1, z1) in zip(polygon, np.roll(polygon, -1, axis=0)):
        if (z0 > z) != (z1 > z) and x < x0 + (z - z0) * (x1 - x0) / (z1 - z0):
            inside = not inside
    return inside


def _from_anchors_v1(data: dict) -> RoomModel:
    anchors = data.get("anchors", [])
    has = lambda item, label: label in item.get("labels", [])  # noqa: E731
    floors = [a for a in anchors if has(a, "FLOOR")]
    ceilings = [a for a in anchors if has(a, "CEILING")]
    walls = [a for a in anchors if has(a, "WALL_FACE")]
    vertices: list[list[float]] = []

    def vertex(x: float, z: float) -> int:
        for i, (vx, vz) in enumerate(vertices):
            if abs(vx - x) < WELD_M and abs(vz - z) < WELD_M:
                return i
        vertices.append([x, z])
        return len(vertices) - 1

    wall_tops = []
    for wall in walls:
        outline = _plane_outline(wall)
        if outline is not None:
            wall_tops.append(float(outline[:, 1].max()))
    regions = []
    for floor in floors:
        outline = _plane_outline(floor)
        if outline is None:
            continue
        floor_y = float(outline[:, 1].mean())
        plan = outline[:, [0, 2]]
        center = plan.mean(axis=0)
        ceiling_y = None
        for ceiling in ceilings:
            c_outline = _plane_outline(ceiling)
            if c_outline is not None and _inside(center, c_outline[:, [0, 2]]):
                ceiling_y = float(c_outline[:, 1].mean())
        if ceiling_y is None:
            ceiling_y = max(wall_tops) if wall_tops else floor_y + 2.5
        loop = [vertex(float(x), float(z)) for x, z in plan]
        loop = [v for k, v in enumerate(loop) if v != loop[k - 1]]  # drop welded duplicates
        regions.append(Region(loop=loop, floor=Plane(floor_y), ceiling=Plane(ceiling_y)))
    model = RoomModel(vertices=np.asarray(vertices, float).reshape(-1, 2), regions=regions, source_version=1)

    entrance_uuid = (data.get("entrance") or {}).get("anchor_uuid")
    openings = []
    for item in anchors:
        kind = "door" if has(item, "DOOR_FRAME") else "window" if has(item, "WINDOW_FRAME") else None
        outline = _plane_outline(item) if kind else None
        if outline is None:
            continue
        opening = _project_opening(model, outline, kind, item.get("uuid") == entrance_uuid)  # type: ignore[arg-type]
        if opening is not None:
            openings.append(opening)
    boxes = []
    for item in anchors:
        volume = item.get("volume")
        if not volume:
            continue
        rotation, position = _anchor_frame(item)
        lo, hi = np.asarray(volume["min"], float), np.asarray(volume["max"], float)
        extents = hi - lo
        center = rotation @ ((lo + hi) / 2) + position
        vertical = int(np.argmax(np.abs(rotation[1, :])))  # local axis closest to world up
        horizontal = [i for i in range(3) if i != vertical]
        width_axis = rotation[:, horizontal[0]]
        yaw = math.degrees(math.atan2(-width_axis[2], width_axis[0]))  # width axis = (cos, 0, -sin)(yaw)
        labels = item.get("labels") or ["OTHER"]
        boxes.append(Box(label=str(labels[0]), center=tuple(center), yaw_deg=yaw,  # type: ignore[arg-type]
                         size=(float(extents[horizontal[0]]), float(extents[vertical]), float(extents[horizontal[1]]))))
    return RoomModel(vertices=model.vertices, regions=regions, openings=openings, boxes=boxes, source_version=1)


def _project_opening(model: RoomModel, outline: np.ndarray, kind: str, entrance: bool) -> Opening | None:
    """Door/window outline (Unity 3D) -> an opening on the nearest parallel wall edge."""
    plan = outline[:, [0, 2]]
    center = plan.mean(axis=0)
    span = plan.max(axis=0) - plan.min(axis=0)
    best = None
    for (i, j), region in _edges(model):
        a, b = model.point(i), model.point(j)
        direction = b - a
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            continue
        direction /= length
        normal = np.array([-direction[1], direction[0]])
        distance = abs(float(np.dot(center - a, normal)))
        along = (plan - a) @ direction
        if distance > 0.3 or along.max() < 0 or along.min() > length:
            continue
        if best is None or distance < best[0]:
            best = (distance, (i, j), region, along)
    if best is None:
        return None
    _, (i, j), region, along = best
    start, end = max(0.0, float(along.min())), min(float(np.linalg.norm(model.point(j) - model.point(i))), float(along.max()))
    if end - start < 0.05:
        return None
    mid = model.point(i) + (model.point(j) - model.point(i)) / np.linalg.norm(model.point(j) - model.point(i)) * (start + end) / 2
    floor_y = region.floor.at(*mid)
    return Opening(edge=(i, j), offset=start, width=end - start, sill=max(0.0, float(outline[:, 1].min()) - floor_y),
                   height=float(outline[:, 1].max() - outline[:, 1].min()), kind=kind, entrance=entrance and kind == "door")


# ---------------------------------------------------------------- topology

def _loops(region: Region) -> list[list[int]]:
    return [region.loop, *region.holes]


def _edges(model: RoomModel):
    """Every directed edge (i, j) of every non-solid region with its region."""
    for region in model.regions:
        if region.solid:
            continue
        for loop in _loops(region):
            for k in range(len(loop)):
                yield (loop[k], loop[(k + 1) % len(loop)]), region


def _edge_owners(model: RoomModel) -> dict[tuple[int, int], list[tuple[Region, tuple[int, int]]]]:
    owners: dict[tuple[int, int], list[tuple[Region, tuple[int, int]]]] = {}
    for (i, j), region in _edges(model):
        owners.setdefault((min(i, j), max(i, j)), []).append((region, (i, j)))
    return owners


def _interior_normal(model: RoomModel, region: Region, i: int, j: int) -> np.ndarray:
    """Unit plan normal of edge (i, j) pointing into ``region``."""
    a, b = model.point(i), model.point(j)
    direction = (b - a) / max(np.linalg.norm(b - a), 1e-12)
    normal = np.array([-direction[1], direction[0]])
    probe = (a + b) / 2 + normal * 0.01
    inside = _inside(probe, model.vertices[region.loop]) and not any(_inside(probe, model.vertices[h]) for h in region.holes)
    return normal if inside else -normal


def region_area(model: RoomModel, region: Region) -> float:
    def area(loop):
        p = model.vertices[loop]
        return 0.5 * abs(float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])))
    return area(region.loop) - sum(area(h) for h in region.holes)


def region_at(model: RoomModel, x: float, z: float) -> Region | None:
    point = np.array([x, z])
    for region in model.regions:
        if region.solid:
            continue
        if _inside(point, model.vertices[region.loop]) and not any(_inside(point, model.vertices[h]) for h in region.holes):
            return region
    return None


def main_region(model: RoomModel) -> Region:
    return max((r for r in model.regions if not r.solid), key=lambda r: region_area(model, r))


# ---------------------------------------------------------------- triangulation

def triangulate_polygon(points: np.ndarray) -> np.ndarray:
    """Ear clipping for a simple 2D polygon (either winding). Returns (K, 3) indices."""
    pts = np.asarray(points, float)
    n = len(pts)
    if n < 3:
        return np.zeros((0, 3), int)
    signed = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    order = list(range(n)) if signed > 0 else list(range(n))[::-1]  # counter-clockwise

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
            if any(inside(pts[m], a, b, c) and not np.allclose(pts[m], a) and not np.allclose(pts[m], b) and not np.allclose(pts[m], c)
                   for m in order if m not in (i, j, l)):
                continue
            triangles.append([i, j, l])
            order.pop(k)
            break
        else:
            break  # not simple: fan the rest
    if len(order) >= 3:
        triangles += [[order[0], order[k], order[k + 1]] for k in range(1, len(order) - 1)]
    return np.asarray(triangles, int)


def triangulate_with_holes(outer: np.ndarray, holes: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Bridge each hole to the outer loop (rightmost hole vertex to the nearest outer vertex), then ear-clip.
    Returns (points, triangles) where points may repeat bridge vertices."""
    def ccw(p):
        signed = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
        return p if signed > 0 else p[::-1]

    polygon = [tuple(v) for v in ccw(np.asarray(outer, float))]
    for hole in sorted(holes, key=lambda h: -np.max(np.asarray(h)[:, 0])):
        h = np.asarray(hole, float)
        h = h[::-1] if 0.5 * np.sum(h[:, 0] * np.roll(h[:, 1], -1) - np.roll(h[:, 0], -1) * h[:, 1]) > 0 else h  # clockwise
        start = int(np.argmax(h[:, 0]))
        anchor = h[start]
        target = min(range(len(polygon)), key=lambda k: (polygon[k][0] - anchor[0]) ** 2 + (polygon[k][1] - anchor[1]) ** 2
                     if polygon[k][0] >= anchor[0] - 1e-9 else float("inf"))
        hole_seq = [tuple(h[(start + k) % len(h)]) for k in range(len(h))]
        polygon = polygon[:target + 1] + hole_seq + [hole_seq[0], polygon[target]] + polygon[target + 1:]
    points = np.asarray(polygon, float)
    return points, triangulate_polygon(points)


# ---------------------------------------------------------------- geometry

@dataclass
class RoomPart:
    name: str
    kind: str  # wall, floor, ceiling, step, bulkhead, door, window, furniture, entrance
    vertices: np.ndarray  # (N, 3) MQ3DRecon world
    faces: np.ndarray  # (M, 3)
    color: tuple[float, float, float] = (0.8, 0.8, 0.8)
    label: str = ""


def _world(points_unity: np.ndarray) -> np.ndarray:
    return np.asarray(points_unity, float) @ UNITY_TO_WORLD.T


def _between(a: np.ndarray, b: np.ndarray, bottom: tuple[float, float], top: tuple[float, float]) -> list[np.ndarray]:
    """Vertical faces between two lines over plan segment a->b: bottom/top heights at a and at b. Splits a crossing."""
    (b0, b1), (t0, t1) = bottom, top
    d0, d1 = t0 - b0, t1 - b1
    if max(abs(d0), abs(d1)) < 0.01:
        return []
    if d0 * d1 < 0:  # the lines cross: two triangles
        s = d0 / (d0 - d1)
        m = a + (b - a) * s
        ym = b0 + (b1 - b0) * s
        return [np.array([[a[0], b0, a[1]], [m[0], ym, m[1]], [a[0], t0, a[1]]]),
                np.array([[m[0], ym, m[1]], [b[0], b1, b[1]], [b[0], t1, b[1]]])]
    return [np.array([[a[0], b0, a[1]], [b[0], b1, b[1]], [b[0], t1, b[1]], [a[0], t0, a[1]]])]


def _polygons_to_part(name: str, kind: str, polygons: list[np.ndarray], color, label="") -> RoomPart | None:
    vertices, faces = [], []
    for poly in polygons:
        base = len(vertices)
        vertices += list(poly)
        faces += [[base, base + k, base + k + 1] for k in range(1, len(poly) - 1)]  # convex: fan
    if not faces:
        return None
    return RoomPart(name, kind, _world(np.asarray(vertices, float)), np.asarray(faces, int), color, label)


def _wall_polygons(model: RoomModel, region: Region, i: int, j: int) -> list[np.ndarray]:
    """Wall along edge i->j from the region's floor to its ceiling, minus its openings (cells between breaks)."""
    a, b = model.point(i), model.point(j)
    length = float(np.linalg.norm(b - a))
    if length < 1e-6:
        return []
    direction = (b - a) / length
    holes = []
    for opening in model.openings:
        if {opening.edge[0], opening.edge[1]} != {i, j}:
            continue
        start = opening.offset if opening.edge == (i, j) else length - opening.offset - opening.width
        mid = a + direction * (start + opening.width / 2)
        bottom = region.floor.at(*mid) + opening.sill
        holes.append((start, start + opening.width, bottom, bottom + opening.height))
    us = sorted({0.0, length, *[min(max(v, 0.0), length) for h in holes for v in h[:2]]})
    polygons = []
    for u0, u1 in zip(us[:-1], us[1:]):
        if u1 - u0 < 1e-6:
            continue
        p0, p1 = a + direction * u0, a + direction * u1
        f0, f1 = region.floor.at(*p0), region.floor.at(*p1)
        c0, c1 = region.ceiling.at(*p0), region.ceiling.at(*p1)
        cuts = sorted((h[2], h[3]) for h in holes if h[0] <= (u0 + u1) / 2 <= h[1])
        lower = (f0, f1)
        for bottom, top in cuts:  # piece below the opening, then continue above it
            polygons += _between(p0, p1, lower, (min(bottom, c0), min(bottom, c1)))
            lower = (max(top, f0), max(top, f1))
        polygons += _between(p0, p1, lower, (c0, c1))
    return polygons


def build_room_geometry(model: RoomModel) -> list[RoomPart]:
    parts: list[RoomPart] = []
    counters: dict[str, int] = {}

    def name(key: str) -> str:
        key = key.lower().replace(" ", "_")
        counters[key] = counters.get(key, 0) + 1
        return f"{key}_{counters[key]}"

    def add(part: RoomPart | None):
        if part is not None:
            parts.append(part)

    # Walls (room space on one side only), steps and bulkheads (two regions that differ).
    for _key, owners in _edge_owners(model).items():
        if len(owners) == 1:
            region, (i, j) = owners[0]
            add(_polygons_to_part(name("wall"), "wall", _wall_polygons(model, region, i, j), PART_COLORS["wall"], "WALL"))
        elif len(owners) >= 2:
            (ra, (i, j)), (rb, _) = owners[0], owners[1]
            a, b = model.point(i), model.point(j)
            floors = _between(a, b, (min(ra.floor.at(*a), rb.floor.at(*a)), min(ra.floor.at(*b), rb.floor.at(*b))),
                              (max(ra.floor.at(*a), rb.floor.at(*a)), max(ra.floor.at(*b), rb.floor.at(*b))))
            add(_polygons_to_part(name("step"), "step", floors, PART_COLORS["step"], "STEP"))
            ceilings = _between(a, b, (min(ra.ceiling.at(*a), rb.ceiling.at(*a)), min(ra.ceiling.at(*b), rb.ceiling.at(*b))),
                                (max(ra.ceiling.at(*a), rb.ceiling.at(*a)), max(ra.ceiling.at(*b), rb.ceiling.at(*b))))
            add(_polygons_to_part(name("bulkhead"), "bulkhead", ceilings, PART_COLORS["bulkhead"], "BULKHEAD"))

    # Floors and ceilings per region.
    for region in model.regions:
        if region.solid or len(region.loop) < 3:
            continue
        points, triangles = triangulate_with_holes(model.vertices[region.loop], [model.vertices[h] for h in region.holes])
        for plane, kind in ((region.floor, "floor"), (region.ceiling, "ceiling")):
            vertices = np.c_[points[:, 0], [plane.at(x, z) for x, z in points], points[:, 1]]
            add(RoomPart(name(kind), kind, _world(vertices), triangles.copy(), PART_COLORS[kind], kind.upper()))

    # Door and window panels, 1 cm into the room so they do not z-fight the wall.
    owners = _edge_owners(model)
    for opening in model.openings:
        owner = owners.get((min(opening.edge), max(opening.edge)))
        if not owner:
            continue
        region = owner[0][0]
        i, j = opening.edge
        a, b = model.point(i), model.point(j)
        direction = (b - a) / max(np.linalg.norm(b - a), 1e-12)
        inward = _interior_normal(model, region, i, j) * 0.01
        p0 = a + direction * opening.offset + inward
        p1 = a + direction * (opening.offset + opening.width) + inward
        y0 = region.floor.at(*(p0 + p1) / 2) + opening.sill
        quad = np.array([[p0[0], y0, p0[1]], [p1[0], y0, p1[1]], [p1[0], y0 + opening.height, p1[1]], [p0[0], y0 + opening.height, p0[1]]])
        kind = "entrance" if opening.entrance else opening.kind
        add(RoomPart(name(kind), kind, _world(quad), np.array([[0, 1, 2], [0, 2, 3]]), PART_COLORS.get(kind, PART_COLORS["door"]),
                     "DOOR_FRAME" if opening.kind == "door" else "WINDOW_FRAME"))

    for box in model.boxes:
        add(box_part(box, name(box.label)))
    return parts


def box_corners_unity(box: Box) -> np.ndarray:
    yaw = math.radians(box.yaw_deg)
    width_axis = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    depth_axis = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
    w, h, d = box.size
    c = np.asarray(box.center, float)
    return np.array([c + sx * w / 2 * width_axis + sy * h / 2 * np.array([0, 1.0, 0]) + sz * d / 2 * depth_axis
                     for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)])


def box_part(box: Box, part_name: str) -> RoomPart:
    faces = np.array([
        [0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4],
        [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5],
    ])
    return RoomPart(part_name, "furniture", _world(box_corners_unity(box)), faces, FURNITURE_COLORS.get(box.label, PART_COLORS["furniture"]), box.label)


# ---------------------------------------------------------------- spawn and floor

def entrance_spawn(model: RoomModel, floor_y: float | None = None, step_in_m: float = 0.6) -> tuple[tuple[float, float, float], float] | None:
    """(position, yaw_deg) in MQ3DRecon world: just inside the entrance door, on that region's floor, facing in."""
    door = next((o for o in model.openings if o.entrance), None)
    owners = _edge_owners(model).get((min(door.edge), max(door.edge))) if door else None
    if door is None or not owners:
        return None
    region = owners[0][0]
    i, j = door.edge
    a, b = model.point(i), model.point(j)
    direction = (b - a) / max(np.linalg.norm(b - a), 1e-12)
    inward = _interior_normal(model, region, i, j)
    spot = a + direction * (door.offset + door.width / 2) + inward * step_in_m
    inside = region_at(model, *spot) or region
    y = inside.floor.at(*spot) if floor_y is None else floor_y
    position = _world(np.array([spot[0], y, spot[1]]))
    facing = _world(np.array([inward[0], 0.0, inward[1]]))
    yaw = float(np.degrees(np.arctan2(-facing[0], -facing[2])))  # same convention as spawn_from_camera
    return (float(position[0]), float(position[1]), float(position[2])), yaw


def space_setup_floor_y(model: RoomModel) -> float | None:
    """Height of the main (largest) region's floor at its centroid."""
    regions = [r for r in model.regions if not r.solid]
    if not regions:
        return None
    region = main_region(model)
    center = model.vertices[region.loop].mean(axis=0)
    return float(region.floor.at(*center))


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
    """Wavefront OBJ + MTL, one group per element (wall_1, step_1, table_1...), Y up, meters."""
    path = Path(path)
    mtl = path.with_suffix(".mtl")
    kinds = {}
    for part in parts:
        kinds.setdefault(f"{part.kind}_{part.label.lower() or part.kind}", part.color)
    mtl.write_text("".join(f"newmtl {k}\nKd {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\nd 1.0\n\n" for k, c in kinds.items()), encoding="utf-8")
    lines, offset = [f"# MQ3DRecon room model (Space Setup, edited on the headset), meters, Y up\nmtllib {mtl.name}\n"], 1
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
            "elements": counts, "regions": sum(1 for r in model.regions if not r.solid), "schema": model.source_version}
