"""Live capture workflow: ``mq3drecon live`` (receiver + recorder + GPU preview + auto-build).

Importable without Open3D/Warp; heavy modules are imported when the workflow runs.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from mq3drecon.config.live_config import LiveConfig, load_live_config
from mq3drecon.streaming.adb import setup_adb_connection
from mq3drecon.streaming.discovery import DiscoveryResponder
from mq3drecon.streaming.receiver import LiveReceiver
from mq3drecon.workflows.live_session import CompletedSession, LiveSessionController

logger = logging.getLogger(__name__)

LIVE_MESH_FILE = "live_mesh.ply"


def default_workspace() -> Path:
    """Safe default workspace (specs/LIVE_RECONSTRUCTION.md): ``~/MQ3DRecon``."""
    return Path.home() / "MQ3DRecon"


def save_live_mesh(completed: CompletedSession) -> Path | None:
    """Write the session's final live mesh (and any earlier tracking segments) next to its recording."""
    snapshot = completed.final_snapshot
    if snapshot is None:
        return None
    import numpy as np
    import open3d as o3d

    def write(mesh_arrays, path: Path) -> None:
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh_arrays.vertices.astype(np.float64)),
            o3d.utility.Vector3iVector(mesh_arrays.triangles.astype(np.int32)),
        )
        mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_arrays.colors.astype(np.float64))
        o3d.io.write_triangle_mesh(str(path), mesh)

    for index, segment in enumerate(snapshot.previous_segments):
        write(segment, completed.project_dir / f"live_mesh_segment_{index}.ply")
    if not len(snapshot.mesh.triangles):
        return None
    path = completed.project_dir / LIVE_MESH_FILE
    write(snapshot.mesh, path)
    return path


def _format_status(controller: LiveSessionController, reconstructor) -> str:
    status = controller.status()
    stats = reconstructor.stats
    link = status.connected_peer or "waiting for headset"
    session = status.session_name or "idle"
    dropped = sum(status.headset_dropped.values())
    return (
        f"[live] {link} | {session} | rx color {status.color_frames} depth {status.depth_frames} | "
        f"integrated {stats.frames_integrated} ({stats.integrate_ms:.1f} ms) | "
        f"dropped headset {dropped} live {stats.frames_dropped} | tris {stats.triangles:,}"
        + (f" | tracking jumps {stats.tracking_jumps}" if stats.tracking_jumps else "")
    )


def _announce(config: LiveConfig, port: int, workspace: Path) -> None:
    print(f"[live] Listening on TCP {port}; sessions are saved under {workspace / 'sessions'}")
    if config.server.adb:
        adb, setups = setup_adb_connection(
            port, adb_path=Path(config.server.adb_path) if config.server.adb_path else None,
            push=config.server.push_endpoint, package=config.server.app_package,
        )
        if adb is None:
            print("[live] adb not found: the headset will find this laptop by UDP discovery.")
        for setup in setups:
            print(
                f"[live] headset {setup.serial}: ip {setup.device_ip or '?'}, adb reverse {'ok' if setup.reverse_ok else 'failed'}, "
                f"endpoint {setup.endpoint_host or '?'}:{port} {'pushed' if setup.endpoint_pushed else 'not pushed (is the capture app installed?)'}"
            )


def run_live_capture(
    *,
    workspace: Path | None = None,
    config: LiveConfig | None = None,
    config_yml_path: Path | None = None,
    mode: str | None = None,
    show_viewer: bool | None = None,
    stop_event: threading.Event | None = None,
    on_session_complete: Callable[[CompletedSession], None] | None = None,
    on_listening: Callable[[int], None] | None = None,
) -> list[CompletedSession]:
    """Run until the viewer window closes, ``stop_event`` is set, or Ctrl+C. Returns completed sessions."""
    if config is not None and config_yml_path is not None:
        raise ValueError("Pass either config or config_yml_path, not both")
    cfg = config or load_live_config(config_yml_path, mode)
    workspace = Path(workspace) if workspace is not None else default_workspace()
    stop = stop_event or threading.Event()

    from mq3drecon.processing.live.reconstructor import LiveReconstructor
    from mq3drecon.workflows.live_feedback import FeedbackPublisher

    reconstructor = LiveReconstructor(cfg)

    def completed_callback(completed: CompletedSession) -> None:
        mesh_path = save_live_mesh(completed)
        print(f"[live] Session saved: {completed.project_dir}" + (f" (live mesh {mesh_path.name})" if mesh_path else ""))

        def postprocess() -> None:
            if cfg.build.compact_after_session:
                try:
                    from mq3drecon.workflows.live_postprocess import compact_in_place

                    compact_in_place(completed.project_dir, cfg)
                except Exception:
                    logger.exception("Compaction failed; the full recording is kept: %s", completed.project_dir)
            if on_session_complete is not None:
                on_session_complete(completed)

        threading.Thread(target=postprocess, name="mq3drecon-live-postprocess", daemon=False).start()

    controller = LiveSessionController(workspace / "sessions", reconstructor, completed_callback)
    receiver = LiveReceiver(controller, cfg.server.host, cfg.server.port, idle_timeout_s=cfg.server.idle_timeout_s)
    port = receiver.start()
    feedback = FeedbackPublisher(cfg, receiver, reconstructor, controller)
    reconstructor.on_keyframe = feedback.on_keyframe
    feedback.start()
    discovery = None
    if cfg.server.discovery:
        try:
            discovery = DiscoveryResponder(port, discovery_port=cfg.server.discovery_port)
            discovery.start()
        except OSError as exc:
            print(f"[live] UDP discovery disabled ({exc})")
            discovery = None
    _announce(cfg, port, workspace)
    if on_listening is not None:
        on_listening(port)

    last_print = [0.0]

    def tick() -> None:
        if time.monotonic() - last_print[0] >= 1.0:
            last_print[0] = time.monotonic()
            sys.stdout.write("\r" + _format_status(controller, reconstructor)[:180].ljust(180))
            sys.stdout.flush()

    viewer = cfg.preview.viewer if show_viewer is None else show_viewer
    try:
        if viewer:
            from mq3drecon.processing.live.viewer import run_preview_window

            run_preview_window(reconstructor.latest_snapshot, stop, on_tick=tick)
        else:
            while not stop.wait(0.2):
                tick()
    except KeyboardInterrupt:
        pass
    finally:
        print()
        feedback.stop()
        receiver.stop()
        if discovery is not None:
            discovery.stop()
        controller.finish_open_session("shutdown")
        reconstructor.close()
    return list(controller.completed)


def run_live_replay(project_dir: Path, *, host: str = "127.0.0.1", port: int = 9870, speed: float = 1.0) -> int:
    """Stream a recorded MRUK project to a running ``mq3drecon live``. Returns bytes sent."""
    from mq3drecon.streaming.replay import replay_project

    return replay_project(Path(project_dir), host, port, speed=speed)


def run_make_synthetic_capture(output_dir: Path, *, frames: int = 60, color_format: str = "rgba") -> Path:
    """Write a synthetic room capture (MRUK layout) for testing without a headset."""
    from mq3drecon.synthetic import SyntheticCaptureSpec, write_synthetic_capture

    return write_synthetic_capture(Path(output_dir), SyntheticCaptureSpec(frames=frames, color_format=color_format))
