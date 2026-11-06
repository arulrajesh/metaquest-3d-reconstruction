"""Desktop live preview window (Open3D legacy visualizer), run on the main thread."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np
import open3d as o3d

from mq3drecon.processing.live.reconstructor import LiveSnapshot

_FRUSTUM_POINTS = np.array([[0, 0, 0], [-0.12, -0.09, 0.2], [0.12, -0.09, 0.2], [0.12, 0.09, 0.2], [-0.12, 0.09, 0.2]], dtype=np.float64)
_FRUSTUM_LINES = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]


def _update_mesh(target: o3d.geometry.TriangleMesh, snapshot: LiveSnapshot) -> None:
    target.vertices = o3d.utility.Vector3dVector(snapshot.mesh.vertices.astype(np.float64))
    target.triangles = o3d.utility.Vector3iVector(snapshot.mesh.triangles.astype(np.int32))
    target.vertex_colors = o3d.utility.Vector3dVector(snapshot.mesh.colors.astype(np.float64))
    target.compute_vertex_normals()


def _update_frustum(target: o3d.geometry.LineSet, camera_to_world: np.ndarray | None) -> None:
    if camera_to_world is None:
        return
    points = _FRUSTUM_POINTS @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    target.points = o3d.utility.Vector3dVector(points)
    target.lines = o3d.utility.Vector2iVector(_FRUSTUM_LINES)
    target.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.5, 0.0], (len(_FRUSTUM_LINES), 1)))


def _update_trajectory(target: o3d.geometry.LineSet, trajectory: np.ndarray) -> None:
    if len(trajectory) < 2:
        return
    target.points = o3d.utility.Vector3dVector(trajectory.astype(np.float64))
    target.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(trajectory) - 1)])
    target.colors = o3d.utility.Vector3dVector(np.tile([0.2, 0.6, 1.0], (len(trajectory) - 1, 1)))


def run_preview_window(
    get_snapshot: Callable[[], LiveSnapshot | None],
    stop_event: threading.Event,
    *,
    title: str = "MQ3DRecon live",
    on_tick: Callable[[], None] | None = None,
) -> None:
    """Show the growing model until the window is closed or ``stop_event`` is set. Closing sets ``stop_event``."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=800)
    options = vis.get_render_option()
    options.background_color = np.array([0.08, 0.08, 0.1])
    options.mesh_show_back_face = True
    mesh = o3d.geometry.TriangleMesh()
    frustum = o3d.geometry.LineSet()
    trajectory = o3d.geometry.LineSet()
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))
    shown_sequence = -1
    has_geometry = False
    try:
        while not stop_event.is_set():
            snapshot = get_snapshot()
            if snapshot is not None and snapshot.sequence != shown_sequence and len(snapshot.mesh.triangles):
                shown_sequence = snapshot.sequence
                if has_geometry:
                    for geometry in (mesh, frustum, trajectory):
                        vis.remove_geometry(geometry, reset_bounding_box=False)
                _update_mesh(mesh, snapshot)
                _update_frustum(frustum, snapshot.camera_to_world)
                _update_trajectory(trajectory, snapshot.trajectory)
                for geometry in (mesh, frustum, trajectory):
                    vis.add_geometry(geometry, reset_bounding_box=not has_geometry)
                has_geometry = True
            if not vis.poll_events():
                break
            vis.update_renderer()
            if on_tick is not None:
                on_tick()
            time.sleep(0.015)
    finally:
        stop_event.set()
        vis.destroy_window()
