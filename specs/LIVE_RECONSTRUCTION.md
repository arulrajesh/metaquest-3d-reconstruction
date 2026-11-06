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
* **Color intrinsics:** from the session's MRUK intrinsics, through `dataio/mruk_intrinsics.image_pinhole`.
  * MRUK's `focalLength` and `principalPoint` are in **sensor pixels** (`sensorResolution`, 1728×1728 on Quest 3).
  * A delivered frame (`resolution`, e.g. 1280×1280) is the sensor image center-cropped to the frame's aspect ratio and then scaled uniformly (Meta's `PassthroughCameraUtils` mapping).
  * Files without `sensorResolution` are taken to be in image pixels.
  * Until 2026-11-06 the values were scaled from `resolution`, which gave a focal length 35% too long and a principal point about 225 px off. Fixing it raised splat PSNR on a real capture from 22.5 to 28.6 dB.

  The principal point is used **as reported, not mirrored**. It was re-confirmed after the fix: raw gives 28.6 dB, mirrored 27.2 dB. A held-out-camera test on real Quest 3 data (2026-11-06) settled this: colors baked from the left camera matched the right camera's images 12% better with raw principal points on both cameras than with the offline pipeline's mirrored ones. Shifting both principal points together changed the error by only about 1%.
* **Poses:** Unity camera-to-world converted with `Transforms.convert_coordinate_system(OPEN3D, is_camera=True)`.
* **Color matching:** each depth frame uses the nearest color frame of the same side within `color_tolerance_ms`. Without one, geometry is still integrated without color.

---

# GPU TSDF Volume (Kalman)

The volume is dense: it covers `extent_m` around the camera position plus
`(0, vertical_offset_m, 0)`, is allocated at the first valid depth frame of a
segment, and runs on `device` (default `cuda:0`). It must fall back to the CPU,
with identical results, when CUDA is unavailable. Truncation is
`trunc_voxels * voxel_size_m`, and TSDF values are normalized by it.

**Each voxel is a 1-D Kalman filter** over its truncated signed distance, with
mean `tsdf`, information `info` (inverse variance; 0 = unobserved), and a
running outlier rate. A depth reading projects onto the voxel and gives
`obs = min(1, sdf / trunc)`. Readings with `sdf < -trunc` are ignored.

| Step | Rule |
| --- | --- |
| Measurement noise | `sigma(z) = sigma0 + k·z²` (defaults 4 mm and 3 mm/m²), multiplied by `edge_inflation` (10) where the depth gradient exceeds `edge_ratio · z` (0.05) or a neighbor is invalid (flying pixels). Normalized variance `r = (sigma / trunc)²`. |
| Process noise | Before each update, `info ← 1 / (1/info + q)`, with `q = 1e-5`. |
| Innovation gate | Applies to near-surface readings (`obs < 1`) of confident voxels (posterior sigma ≤ `confident_sigma_m`, 2 cm). The reading is rejected when `(obs − tsdf)² > gate_sigma² · (1/info + r)` (`gate_sigma` = 3), and the outlier rate is updated as `rate ← 0.9·rate + 0.1`. Accepted readings decay it: `rate ← 0.9·rate`. |
| Re-initialization | When the rate reaches `reinit_outlier_rate` (0.6), the voxel restarts from the current reading. This handles persistent change. |
| Update | `tsdf ← (tsdf·info + obs/r) / (info + 1/r)`, then `info ← min(info + 1/r, max_info)`. |
| Color | Only where `\|sdf\| ≤ trunc`: an information-weighted mean (weight `1/r`) of the color sampled by projecting the voxel into the color camera. |
| Frame score | `integrate` returns the near-surface information gain (Σ prior variance − posterior variance; an unobserved voxel's prior variance is 1), the counts of updated, new, and rejected voxels, and `novelty` = gain / updated voxels. |

Free-space readings (`obs = 1`) are never gated, so space that becomes empty
(a person walks away) is carved out. A surface that moves *closer* hides the old
one, which then remains; this is inherent to projective TSDFs.

**Mesh extraction** is marching cubes at the zero level. A vertex lies on one
grid edge and is kept only when both edge voxels have posterior sigma at most
`max_sigma_m` (default half a voxel) and an outlier rate of at most 0.3. These
are the points worth keeping. Each vertex reports its posterior sigma. Vertices
without color are gray (0.6) with `color_valid = False`.

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

# Keyframe-only Fusion

The live worker applies the keyframe policy of `SESSION_COMPACTION.md` causally:

1. Each depth frame is scored by a dry run, with the color camera of the nearest color frame for color novelty.
2. The keyframe selector decides.
3. Only kept frames are fused. A wanted color keyframe uses the locally sharpest color frame within `color_window_s`.

The live model therefore shows exactly what the compact recording will contain.

Tracking jumps come from the pose Kalman filter's innovation test, with a
minimum innovation of `integration.jump_min_m`.

---

# Headset Feedback

While a headset is connected, the laptop sends (`LIVE_STREAMING.md`):

* a `KEYFRAME` notice for every decision;
* the live mesh simplified by vertex clustering (`feedback.surface_cluster_m`, enlarged until at most `feedback.surface_max_triangles`), every `feedback.surface_interval_s`;
* `STATUS` every `feedback.status_interval_s`.

---

# After a Session

When `build.compact_after_session` is set (the default), the ended session is
compacted in a background thread (`SESSION_COMPACTION.md`):

* the compact copy is written as `<name>.compacting`;
* only after that succeeds is the full recording removed, or kept as `<name>_full` when `build.keep_full_recording` is set;
* the copy is then renamed to `<name>`;
* `live_mesh*.ply` and `live_session.json` are carried over.

If compaction fails, the full recording is kept.

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

`preview.max_sigma_voxels` (default 0.5) sets the live mesh's `max_sigma_m` as a fraction of the voxel size.

| Preset | `voxel_size_m` | `extent_m` | `vertical_offset_m` |
| --- | --- | --- | --- |
| `indoor` (default) | 0.04 | 12 × 5 × 12 | -0.3 |
| `outdoor` | 0.06 | 48 × 8 × 48 | -0.8 |
