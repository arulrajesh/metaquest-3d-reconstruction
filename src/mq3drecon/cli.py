"""Command-line entrypoint for package-backed MQ3DRecon workflows."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from mq3drecon.errors import MQ3DReconError
from mq3drecon.models.side import Side
from mq3drecon.workflows import (
    export_colmap_project,
    run_color_aligned_depth_to_png,
    run_foundation_stereo_depth,
    run_depth_to_linear,
    run_live_capture,
    run_live_replay,
    run_make_synthetic_capture,
    run_reconstruct_scene,
    run_rgba_to_png,
    run_visualize_camera_trajectories,
    run_yuv_to_rgb,
)

_EXPECTED_ERRORS = (MQ3DReconError, FileNotFoundError, ValueError, OSError)



def _parse_side(value: str) -> Side:
    try:
        return Side[str(value).upper()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError("side must be one of: left, right") from exc

def _existing_project_dir(value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Input directory does not exist: {path}")
    return path


def _add_project_and_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project-dir",
        "--project_dir",
        "-p",
        type=_existing_project_dir,
        required=True,
        help="Path to the project directory containing QRC data.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Optional path to the YAML config file for the pipeline. Uses built-in defaults when omitted.",
    )


def _run_yuv_to_rgb(args: argparse.Namespace) -> int:
    run_yuv_to_rgb(project_dir=args.project_dir, config_yml_path=args.config)
    return 0


def _run_depth_to_linear(args: argparse.Namespace) -> int:
    run_depth_to_linear(project_dir=args.project_dir, config_yml_path=args.config)
    return 0


def _run_rgba_to_png(args: argparse.Namespace) -> int:
    run_rgba_to_png(project_dir=args.project_dir)
    return 0


def _run_foundation_stereo_depth(args: argparse.Namespace) -> int:
    run_foundation_stereo_depth(
        project_dir=args.project_dir,
        model_path=args.model_path,
        config_yml_path=args.config,
    )
    return 0



def _run_color_aligned_depth_to_png(args: argparse.Namespace) -> int:
    result = run_color_aligned_depth_to_png(
        project_dir=args.project_dir,
        side=args.side,
        write_metric_png=args.metric,
        write_preview_png=not args.no_preview,
        depth_png_scale=args.depth_png_scale,
        depth_preview_min_m=args.depth_preview_min_m,
        depth_preview_max_m=args.depth_preview_max_m,
    )
    print(
        f"[Info] Exported {result.preview_png_count} preview PNG(s) "
        f"and {result.metric_png_count} metric PNG(s) for {result.side.name}."
    )
    return 0

def _run_reconstruct(args: argparse.Namespace) -> int:
    run_reconstruct_scene(project_dir=args.project_dir, config_yml_path=args.config)
    return 0


def _run_export_colmap(args: argparse.Namespace) -> int:
    export_colmap_project(
        project_dir=args.project_dir,
        output_dir=args.output_dir,
        use_colored_pointcloud=args.use_colored_pointcloud,
        use_optimized_color_dataset=args.use_optimized_color_dataset,
        interval=args.interval,
    )
    return 0


def _run_visualize_cameras(args: argparse.Namespace) -> int:
    run_visualize_camera_trajectories(project_dir=args.project_dir)
    return 0


def _run_live(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from mq3drecon.config.live_config import load_live_config

    config = load_live_config(args.config, args.mode)
    server = config.server
    if args.port is not None:
        server = replace(server, port=args.port)
    if args.adb is not None:
        server = replace(server, adb_path=str(args.adb))
    if args.no_adb:
        server = replace(server, adb=False)
    config = replace(config, server=server)
    sessions = run_live_capture(workspace=args.workspace, config=config, show_viewer=False if args.no_viewer else None)
    print(f"[Info] {len(sessions)} session(s) recorded.")
    return 0


def _run_live_replay(args: argparse.Namespace) -> int:
    sent = run_live_replay(args.project_dir, host=args.host, port=args.port, speed=args.speed)
    print(f"[Info] Replayed {args.project_dir} ({sent / 1e6:.1f} MB).")
    return 0


def _run_make_synthetic_capture(args: argparse.Namespace) -> int:
    path = run_make_synthetic_capture(args.output_dir, frames=args.frames, color_format=args.color_format)
    print(f"[Info] Synthetic capture written to {path}")
    return 0


def _run_compact_session(args: argparse.Namespace) -> int:
    from mq3drecon.workflows.compaction import compact_session

    report = compact_session(args.project_dir, args.output_dir, delete_source=args.delete_source)
    print(
        f"[Info] frames {report.frames_before} -> {report.frames_after}; "
        f"{report.bytes_before / 1e6:.0f} MB -> {report.bytes_after / 1e6:.0f} MB ({report.ratio:.0f}x smaller). "
        f"Decisions: {report.reasons}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mq3drecon", description="Meta Quest 3D reconstruction toolkit.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    yuv_parser = subparsers.add_parser("yuv-to-rgb", help="Convert captured YUV frames to RGB images.")
    _add_project_and_config_arguments(yuv_parser)
    yuv_parser.set_defaults(handler=_run_yuv_to_rgb)

    depth_parser = subparsers.add_parser("depth-to-linear", help="Convert depth frames to linear depth maps.")
    _add_project_and_config_arguments(depth_parser)
    depth_parser.set_defaults(handler=_run_depth_to_linear)

    rgba_parser = subparsers.add_parser("rgba-to-png", help="Convert MRUK RGBA frames to PNG images.")
    rgba_parser.add_argument(
        "--project-dir",
        "--project_dir",
        "-p",
        type=_existing_project_dir,
        required=True,
        help="Path to the project directory containing MRUK RGBA data.",
    )
    rgba_parser.set_defaults(handler=_run_rgba_to_png)

    stereo_parser = subparsers.add_parser(
        "foundation-stereo-depth",
        help="Generate rectified stereo depth maps with a FoundationStereo ONNX model.",
    )
    _add_project_and_config_arguments(stereo_parser)
    stereo_parser.add_argument(
        "--model-path",
        "--model_path",
        type=Path,
        default=None,
        help="Path to the FoundationStereo ONNX model. Required unless supplied by config.",
    )
    stereo_parser.set_defaults(handler=_run_foundation_stereo_depth)


    color_aligned_depth_parser = subparsers.add_parser(
        "color-aligned-depth-to-png",
        help="Export saved color-aligned depth .npy maps to PNG files for inspection.",
    )
    color_aligned_depth_parser.add_argument(
        "--project-dir",
        "--project_dir",
        "-p",
        type=_existing_project_dir,
        required=True,
        help="Path to the project directory containing color-aligned depth maps.",
    )
    color_aligned_depth_parser.add_argument(
        "--side",
        type=_parse_side,
        default=Side.LEFT,
        help="Camera side to export. Defaults to left.",
    )
    color_aligned_depth_parser.add_argument(
        "--metric",
        action="store_true",
        help="Also write 16-bit metric depth PNG files using --depth-png-scale units per meter.",
    )
    color_aligned_depth_parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable 8-bit visual preview PNG output.",
    )
    color_aligned_depth_parser.add_argument(
        "--depth-png-scale",
        type=float,
        default=1000.0,
        help="Scale factor for 16-bit metric PNG output. Defaults to millimeters.",
    )
    color_aligned_depth_parser.add_argument(
        "--depth-preview-min-m",
        type=float,
        default=0.1,
        help="Minimum depth in meters mapped to black for preview PNG output.",
    )
    color_aligned_depth_parser.add_argument(
        "--depth-preview-max-m",
        type=float,
        default=None,
        help="Maximum depth in meters mapped to white for preview PNG output. Defaults to a deterministic percentile.",
    )
    color_aligned_depth_parser.set_defaults(handler=_run_color_aligned_depth_to_png)

    reconstruct_parser = subparsers.add_parser("reconstruct", help="Run the reconstruction pipeline.")
    _add_project_and_config_arguments(reconstruct_parser)
    reconstruct_parser.set_defaults(handler=_run_reconstruct)

    colmap_parser = subparsers.add_parser("export-colmap", help="Export camera and image data to COLMAP format.")
    colmap_parser.add_argument(
        "--project-dir",
        "--project_dir",
        "-p",
        type=_existing_project_dir,
        required=True,
        help="Path to the project directory containing QRC data.",
    )
    colmap_parser.add_argument(
        "--output-dir",
        "--output_dir",
        "-o",
        type=Path,
        required=True,
        help="Path to the output directory where COLMAP model files will be saved.",
    )
    colmap_parser.add_argument(
        "--use-colored-pointcloud",
        "--use_colored_pointcloud",
        action="store_true",
        help="Include colored 3D point cloud if available.",
    )
    colmap_parser.add_argument(
        "--use-optimized-color-dataset",
        "--use_optimized_color_dataset",
        action="store_true",
        help="Use optimized color datasets if available.",
    )
    colmap_parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Sampling interval for image export. Use every N-th image.",
    )
    colmap_parser.set_defaults(handler=_run_export_colmap)

    visualize_parser = subparsers.add_parser("visualize-cameras", help="Visualize camera trajectories.")
    visualize_parser.add_argument(
        "--project-dir",
        "--project_dir",
        "-p",
        type=_existing_project_dir,
        required=True,
        help="Path to the project directory containing QRC data.",
    )
    visualize_parser.set_defaults(handler=_run_visualize_cameras)

    live_parser = subparsers.add_parser("live", help="Receive a live headset stream, record it, and show the model as it grows.")
    live_parser.add_argument("--workspace", "-w", type=Path, default=None, help="Workspace directory (default: ~/MQ3DRecon). Sessions go to <workspace>/sessions.")
    live_parser.add_argument("--config", "-c", type=Path, default=None, help="Optional YAML file with a 'live' section.")
    live_parser.add_argument("--mode", choices=["indoor", "outdoor"], default=None, help="Capture preset (default: indoor, or the config file's mode).")
    live_parser.add_argument("--port", type=int, default=None, help="TCP port to listen on (default 9870).")
    live_parser.add_argument("--adb", type=Path, default=None, help="Path to the adb executable to use.")
    live_parser.add_argument("--no-adb", action="store_true", help="Do not run adb (rely on UDP discovery).")
    live_parser.add_argument("--no-viewer", action="store_true", help="Run without the desktop preview window.")
    live_parser.set_defaults(handler=_run_live)

    replay_parser = subparsers.add_parser("live-replay", help="Stream a recorded MRUK project to a running 'mq3drecon live'.")
    replay_parser.add_argument("--project-dir", "-p", type=_existing_project_dir, required=True, help="Recorded MRUK project directory.")
    replay_parser.add_argument("--host", default="127.0.0.1", help="Receiver host (default 127.0.0.1).")
    replay_parser.add_argument("--port", type=int, default=9870, help="Receiver TCP port (default 9870).")
    replay_parser.add_argument("--speed", type=float, default=1.0, help="Playback speed; 0 sends as fast as possible.")
    replay_parser.set_defaults(handler=_run_live_replay)

    synthetic_parser = subparsers.add_parser("make-synthetic-capture", help="Write a synthetic room capture for testing without a headset.")
    synthetic_parser.add_argument("--output-dir", "-o", type=Path, required=True, help="New or empty output directory.")
    synthetic_parser.add_argument("--frames", type=int, default=60, help="Number of frames per camera.")
    synthetic_parser.add_argument("--color-format", choices=["rgba", "jpg"], default="rgba", help="Color frame format.")
    synthetic_parser.set_defaults(handler=_run_make_synthetic_capture)

    compact_parser = subparsers.add_parser("compact-session", help="Keep only keyframes worth keeping (Kalman information gain, blur) in a new compact session.")
    compact_parser.add_argument("--project-dir", "-p", type=_existing_project_dir, required=True, help="Recorded MRUK session to compact.")
    compact_parser.add_argument("--output-dir", "-o", type=Path, required=True, help="New or empty directory for the compact session.")
    compact_parser.add_argument("--delete-source", action="store_true", help="Delete the source session after a successful compaction.")
    compact_parser.set_defaults(handler=_run_compact_session)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler

    try:
        return handler(args)
    except _EXPECTED_ERRORS as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
