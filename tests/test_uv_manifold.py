"""Real scans give UVAtlas non-manifold meshes; make_manifold must clean them so unwrap succeeds."""

import numpy as np
import pytest

o3d = pytest.importorskip("open3d")

from mq3drecon.processing.web_location.uv_atlas import make_manifold, orient_consistently, same_vertex_set_mask, unwrap


def _grid(nx: int, ny: int, offset=(0.0, 0.0, 0.0)):
    xs, ys = np.meshgrid(np.arange(nx + 1, dtype=np.float64), np.arange(ny + 1, dtype=np.float64))
    vertices = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1) * 0.1 + np.asarray(offset)
    triangles = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            b, c, d = a + 1, a + nx + 1, a + nx + 2
            triangles += [[a, b, d], [a, d, c]]
    return vertices, np.asarray(triangles, np.int64)


def _broken_scan_mesh():
    """A clean grid plus the defects seen on real captures."""
    vertices, triangles = _grid(6, 6)
    n = len(vertices)
    # 1. A fin: a third triangle on an interior edge (edge shared by 3 triangles).
    a, b = triangles[10][0], triangles[10][2]
    vertices = np.vstack([vertices, [[0.25, 0.25, 0.2]]])
    triangles = np.vstack([triangles, [[a, b, n]]])
    # 2. A bowtie: a second grid touching the first only at its corner vertex.
    v2, t2 = _grid(2, 2, offset=(0.6, 0.6, 0.0))
    corner = 6 * 7 + 6  # (0.6, 0.6, 0) in the first grid
    base = len(vertices)
    t2 = t2 + base
    vertices = np.vstack([vertices, v2])
    t2[t2 == base] = corner  # weld the second grid's (0, 0) corner to the first grid's last corner
    triangles = np.vstack([triangles, t2])
    # 3. A duplicate and a degenerate triangle.
    triangles = np.vstack([triangles, triangles[:1], [[0, 0, 1]]])
    # 4. A two-faced "pillow": the same three vertices twice with opposite winding (real capture, 2026-11-06).
    p = len(vertices)
    vertices = np.vstack([vertices, [[2.0, 0.0, 0.0], [2.02, 0.0, 0.0], [2.0, 0.01, 0.0]]])
    triangles = np.vstack([triangles, [[p, p + 1, p + 2], [p + 1, p, p + 2]]])
    # 5. Winding flipped on a few interior triangles (what quadric decimation leaves behind).
    for index in (20, 21, 40):
        triangles[index] = triangles[index][[0, 2, 1]]
    return vertices, triangles


def _legacy(vertices, triangles):
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles.astype(np.int32)))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile([0.5, 0.4, 0.3], (len(vertices), 1)))
    return mesh


def test_fixture_has_the_real_scan_defects():
    # Don't feed the raw mesh to UVAtlas here: it either raises "Non-manifold mesh" (real capture,
    # 2026-11-06) or crashes the process with an access violation (this fixture's degenerate triangle).
    mesh = _legacy(*_broken_scan_mesh())
    assert len(mesh.get_non_manifold_edges()) > 0
    assert len(mesh.get_non_manifold_vertices()) > 0


def test_orient_consistently_fixes_flipped_triangles_and_keeps_majority_winding():
    vertices, triangles = _grid(4, 4)
    flipped = triangles.copy()
    flipped[[3, 7, 11]] = flipped[[3, 7, 11]][:, [0, 2, 1]]
    oriented, conflict = orient_consistently(flipped)
    assert not conflict.any()
    np.testing.assert_array_equal(np.sort(oriented, axis=1), np.sort(triangles, axis=1))
    # Majority (original) winding wins: every triangle's normal points +Z like the unflipped grid.
    v = vertices[oriented]
    assert np.all(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])[:, 2] > 0)


def test_same_vertex_set_mask_catches_mirrored_pairs_only():
    triangles = np.array([[0, 1, 2], [1, 0, 2], [2, 3, 4], [3, 4, 2], [4, 5, 6]])
    np.testing.assert_array_equal(same_vertex_set_mask(triangles), [True, True, True, True, False])


def test_make_manifold_lets_unwrap_succeed_and_keeps_colors():
    vertices, triangles = _broken_scan_mesh()
    mesh, stats = make_manifold(_legacy(vertices, triangles))
    assert stats["manifold"]
    assert mesh.is_orientable()
    assert not same_vertex_set_mask(np.asarray(mesh.triangles)).any()
    assert 0 < stats["triangles_removed"] < 16  # only the defects and their immediate neighbours go
    assert len(mesh.vertex_colors) == len(mesh.vertices)
    uv = unwrap(np.asarray(mesh.vertices), np.asarray(mesh.triangles), size=256, partitions=1)
    assert len(uv.triangles) == len(mesh.triangles)
    assert np.all((uv.uvs >= 0) & (uv.uvs <= 1))
