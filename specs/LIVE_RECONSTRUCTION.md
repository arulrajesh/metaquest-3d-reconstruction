# Live Reconstruction Specification

## Purpose

This document defines how `mq3drecon live` turns a streamed session into a
growing 3D model on the laptop, and where it writes its outputs.

It owns requirements for:

* the workspace layout and its safe default
* session lifecycle (start, resume, end, disconnect)
* per-frame depth and color conversion for live integration
* the GPU TSDF volume and mesh extraction
* operator exclusion and tracking-jump segmentation
* live configuration and presets

The wire protocol and session recording are specified in `LIVE_STREAMING.md`.

---

# Dependency Boundary

These modules must import without Open3D or Warp:

* `mq3drecon.config.live_config`;
* `mq3drecon.workflows.live_session` and `mq3drecon.workflows.live_reconstruction`;
* `mq3drecon.processing.live.native_depth`, `frame_matching`, and `exclusion`.

`processing.live.gpu_tsdf` and `reconstructor` require Warp. `processing.live.viewer` requires Open3D.

---

# Workspace

The workspace defaults to `~/MQ3DRecon`, a safe default under the user's home
directory as allowed by `CLI_BEHAVIOR.md`. `--workspace` overrides it.

| Path | Content |
| --- | --- |
| `<workspace>/sessions/<session_name>/` | Laptop recording (MRUK layout), `live_session.json`, `live_mesh.ply`, and `live_mesh_segment_<i>.ply` |

Session directory names are the sanitized `session_name`. An existing
non-empty directory is never reused; the name gets a `_2`, `_3`, … suffix.

---

# Session Lifecycle

* **Start.** `SESSION_START` opens a recorder and resets the live volume.
* **Resume.** `SESSION_START` with `resume: true` and the same name, while that session is open, continues it.
* **Supersede.** Any other `SESSION_START` while a session is open finalizes the open session with reason `superseded`.
* **End.** `SESSION_END` finalizes the recording and drains queued depth frames. The final mesh is extracted and saved as `live_mesh.ply`.
* **Disconnect.** A disconnect while a session is open waits `resume_grace_s` (default 30 s) for a resume, then finalizes with reason `disconnected`.
* **Shutdown.** Ending the command finalizes any open session with reason `shutdown`.

---

# Per-frame Conversion

Live integration must reuse the offline pipeline's interpretation of recordings:

* **Depth:** NDC depth goes through `utils.depth_utils.convert_depth_to_linear`. A buffer failing `is_ndc_depth_map_valid` is skipped. Pixels outside `[depth_min_m, depth_max_m]` become invalid (0).
* **Depth intrinsics:** `compute_depth_camera_params`, with the principal point x mirrored by `mirror_principal_point_x`.
* **Color intrinsics:** from the session's MRUK intrinsics, scaled when the streamed frame size differs from `resolution`, then mirrored the same way.
* **Poses:** Unity camera-to-world converted with `Transforms.convert_coordinate_system(OPEN3D, is_camera=True)`.
* **Color matching:** each depth frame uses the nearest color frame of the same side within `color_tolerance_ms`. Without one, geometry is still integrated without color.

---

# GPU TSDF Volume

* **Allocation.** The volume is dense, allocated at the first valid depth frame of a segment. It is centered on the camera position plus `(0, vertical_offset_m, 0)`, with size `extent_m` and voxel size `voxel_size_m`. Truncation is `trunc_voxels * voxel_size_m`.
* **Device.** It runs on `device`, default `cuda:0`. When CUDA is unavailable it must fall back to the CPU and produce identical results.
* **Integration** is projective and per voxel. The TSDF and weight are updated wherever `sdf >= -trunc`. Color is updated only where `|sdf| <= trunc`, by projecting the voxel into the color camera. Weights are capped at `max_weight`.
* **Mesh extraction** is marching cubes at the zero level. Triangles with a vertex on a grid edge whose endpoints are not both observed at `min_weight` or more are removed; this suppresses spurious surfaces against unobserved space. Vertices without a color observation get gray (0.6) and `color_valid = False`.

---

# Operator Exclusion

When `exclude_torso` is set, depth pixels whose world point lies inside a
vertical capsule below the head are invalidated before integration. The
capsule starts 0.15 m below the camera, is 1.75 m long, and has radius
`torso_radius_m` (default 0.35 m).

---

# Tracking-jump Segmentation

A depth frame is a **tracking jump** when the camera moved more than
`jump_distance_m` (default 0.3 m) **and** faster than `jump_speed_mps`
(default 2 m/s) since the previous integrated frame. This typically means the
tracking origin was reset, for example by a boundary re-setup.

On a jump:

* the current volume's mesh is kept as a previous segment, saved as `live_mesh_segment_<i>.ply`;
* a new volume starts;
* the jump is counted in `LiveStats.tracking_jumps`.

The recording itself is unaffected. Segments are in different coordinate frames
and must not be merged without registration.

---

# Back-pressure and Preview

Depth frames reach the GPU worker through a bounded drop-oldest queue
(`queue_size`). Drops are counted and never affect the recording.

Mesh snapshots for the preview are extracted at most every `mesh_interval_s`,
and only after new integration. The preview window runs on the main thread.
Closing it ends the command.

---

# Configuration

The YAML `live:` section mirrors `LiveConfig`: `server`, `volume`,
`integration`, `preview`, and `build`. Unknown keys are an error. `mode`
selects a preset before overrides are applied:

| Preset | `voxel_size_m` | `extent_m` | `vertical_offset_m` |
| --- | --- | --- | --- |
| `indoor` (default) | 0.04 | 12 × 5 × 12 | -0.3 |
| `outdoor` | 0.06 | 48 × 8 × 48 | -0.8 |
