# Plan: Live Streaming Reconstruction and Hosted VR Locations

Status: draft, 2026-11-06
Branch: `feature/live-streaming-reconstruction` (from `main` at `97d15f7`)

This is a planning document, not a specification. Behavioral requirements will
move into `specs/` as each milestone starts, per `specs/COMMON_RULES.md`.

---

## 1. Goal

1. **Live capture preview.** A Meta Quest 3 streams camera frames, depth, and
   poses to a connected laptop while the wearer walks around. The laptop builds
   and shows a growing 3D model in real time, so missed areas are visible
   while the operator is still on site.
2. **Hosted VR locations.** Each captured session becomes a "VR location"
   (indoor or outdoor) published on the user's website. Visitors open it in
   the Quest browser (WebXR), enter VR, and move around inside it.

The live model is a preview and a coverage check. The published location is
built offline from the same recorded session, at full quality.

---

## 2. System overview

```text
 Quest 3 (QuestRealityCapture fork)                 Laptop (this repo)
 ┌───────────────────────────────┐   TCP stream   ┌─────────────────────────────────────┐
 │ MRUK passthrough cams (L/R)   │  USB: adb      │ receiver ─┬─> live TSDF ─> live viewer│
 │ Depth API (L/R)               │  reverse, or   │           │                           │
 │ HMD pose                      │  Wi-Fi/hotspot │           └─> session recorder        │
 │ -> JPEG / f16 encode          │ ─────────────> │               (MRUK project on disk)  │
 │ -> bounded send queue         │                └──────────────────┬──────────────────┘
 └───────────────────────────────┘                                   │ offline
                                                                     v
          existing pipeline: stereo depth -> TSDF mesh (collider) + COLMAP export
                                                                     │
                                       splat training (gsplat / Brush) -> .spz
                                                                     │
                                                                     v
                         web location package (location.json + splat + collider GLB)
                                                                     │
                                                                     v
                             user's site: WebXR viewer, static HTTPS hosting
```

The work spans three codebases:

| Codebase | Owns |
| --- | --- |
| Fork of [QuestRealityCapture](https://github.com/t-34400/QuestRealityCapture) (Unity 6000.4.5f1, MRUK 203, MIT) | Headset-side streaming sender |
| This repo (`mq3drecon`) | Protocol, receiver, live reconstruction, live viewer, session recorder, web-location export |
| User's website | WebXR viewer page, hosting, location gallery |

---

## 3. Key decisions (recommended)

| # | Decision | Reason |
| --- | --- | --- |
| D1 | **Transport: plain TCP with length-prefixed, versioned binary messages.** The headset is the client and the laptop listens. | Simple, debuggable, and works unchanged over USB (`adb reverse tcp:9870 tcp:9870`, headset dials `127.0.0.1`) and Wi-Fi. WebRTC and ROS 2 add a lot of machinery for no gain here. |
| D2 | **USB first, Wi-Fi second.** Outdoors, use the laptop's mobile hotspot. | USB has ample bandwidth and no RF issues. It gets the pipeline working before network tuning. |
| D3 | **Encode on the headset:** JPEG for color (worker thread, `ImageConversion.EncodeNativeArrayToJPG`) and float16 for depth. | Raw MRUK RGBA is about 6.5 MB per eye-frame (see bandwidth table). |
| D4 | **Live depth defaults to Quest native depth.** Stereo depth (Fast-FoundationStereo ONNX) is an opt-in quality mode at a lower rate. | Native depth costs almost nothing. Stereo inference is too slow to run at full frame rate. |
| D5 | **Live reconstruction uses one persistent `VoxelBlockGrid` and the existing per-frame integrator.** It uses raw headset poses, with no pose optimization during the live session. | [`integrate_rgbd_frame`](../../src/mq3drecon/processing/reconstruction/rgbd_integration_common.py#L84) already integrates one frame with no file I/O. |
| D6 | **The live viewer is a non-blocking Open3D window.** | Open3D is already a dependency. The viewer sits behind a small interface so another sink (for example Rerun) can be added later. |
| D7 | **Every live session is saved as an MRUK-layout project.** | The whole offline pipeline (stereo depth, color optimization, tiled TSDF, COLMAP export) then runs on it unchanged. |
| D8 | **Published locations use Gaussian splats for appearance plus a simplified mesh for floor and collision.** | TSDF meshes from Quest depth look blobby and stop at about 5–10 m, so outdoor scenes would have no sky or far field. Splats trained on the camera images look photoreal and include the far field. The mesh gives visitors a walkable floor and walls. |
| D9 | **Develop and test the PC side against a replay sender before touching Unity.** | A Python sender streams an existing recording over the protocol. All PC work, and its tests, then need no headset. |

---

## 4. Bandwidth budget (estimates, verify in M0)

| Stream | Raw size | Sent as | At 10 fps |
| --- | --- | --- | --- |
| Color, per eye, 1280×1280 RGBA | 6.5 MB | JPEG q80, ~0.15–0.3 MB | both eyes: ~3–6 MB/s |
| Native depth, per eye (descriptor gives size, ~320×320) | ~0.4 MB float32 | float16, left eye only, ~0.2 MB | ~2 MB/s |
| Pose / heartbeat | bytes | JSON header only | negligible |
| **Total** | | | **~5–8 MB/s (40–65 Mbit/s)** |

- USB 3 cable: plenty of headroom.
- Wi-Fi 6 near the router: fine.
- Phone or laptop hotspot outdoors: marginal. Fall back to 5 fps or 640×640 color through config.

---

## 5. Wire protocol (draft; becomes `specs/LIVE_STREAMING.md`)

Each message:

```text
magic   4 bytes  "MQLS"
version u16      protocol version (start at 1)
type    u16      message type
hlen    u32      JSON header length
plen    u32      payload length
header  hlen     UTF-8 JSON
payload plen     binary
```

All integers are little-endian.

| Type | Header fields | Payload |
| --- | --- | --- |
| `HELLO` | session name, app version, device; per-side color intrinsics (same schema as `*_camera_mruk_intrinsics.json`); enabled streams and rates | none |
| `COLOR` | `side`, `timestamp_us_realtime`, `width`, `height`, pose (`pos xyz`, `rot xyzw`, Unity), `encoding: "jpeg"`, `row_order: "bottom_up"` | JPEG bytes |
| `DEPTH` | `side`, `timestamp_ms`, create pose, four FOV tangents, `near_z`, `far_z`, `width`, `height`, `encoding: "f16" \| "f32"` | raw NDC depth buffer |
| `HEARTBEAT` | headset time, send-queue depth, dropped-frame counters | none |
| `BYE` | reason | none |

Rules:

- **Same fields and units as the disk formats.** Header fields keep the units and conventions of the on-disk formats in `docs/DATA_FORMAT.md`: color timestamps are microseconds and depth timestamps are milliseconds. The recorder can then write files the existing loaders already read.
- **Color stays bottom-up.** Color rows are sent bottom-up, exactly as MRUK provides them. The PC applies the same vertical flip as [`MRUKImageDataIO.load_rgba`](../../src/mq3drecon/dataio/mruk_image_data_io.py#L74).
- **Bounded queues on both ends.** The headset drops the oldest frame per stream when its queue is full and reports drops in `HEARTBEAT`. The PC also uses bounded drop-oldest queues for live reconstruction. COMMON_RULES forbids unbounded in-memory accumulation.
- **Version mismatch is a typed error,** not a crash, and the receiver logs it.

---

## 6. Milestones

### M0: Feasibility spikes (do first; each answers one risk)

1. **Environment.**
   - Run `uv python install 3.11`, `uv venv`, and `uv pip install -e ".[full]"`. There is no system Python on this laptop.
   - Then check `open3d.core.cuda.is_available()`. Open3D's Windows pip wheels have historically been CPU-only.
   - If CUDA is missing, either accept CPU integration for native-depth live mode (320×320 depth is light) or run the PC side in WSL2 with mirrored networking.
   - `onnxruntime-gpu` works on native Windows either way.
2. **Sample capture.** Record a short indoor MRUK session with stock QuestRealityCapture, pull it with `adb pull`, and run the existing offline pipeline on it end to end.
3. **Laptop throughput** on the RTX 4080 Laptop GPU (12 GB):
   - frames per second for native-depth integration;
   - Fast-FoundationStereo ONNX latency at 640² and 1280²;
   - point-cloud extraction time as the model grows.
4. **Headset encode cost.** Measure JPEG encode time for a 1280×1280 frame on a Quest 3 worker thread, and the thermal and battery impact over 10 minutes.
5. **WebXR splat spike** (de-risks the end goal):
   - Load a public sample splat in a three.js + Spark page and in PlayCanvas, both in Quest Browser in `immersive-vr`.
   - Record the frame rate against splat count, and pick the viewer.
6. **Splat training spike.** Train a splat from the M0.2 capture using the existing `export-colmap` output, with known poses. Try Brush first (easiest on Windows), then gsplat or nerfstudio `splatfacto`.

Exit: a written note with the measured numbers, an updated bandwidth table, and the chosen viewer and trainer.

### M1: Protocol, replay sender, receiver (PC only)

- Specs: write `specs/LIVE_STREAMING.md` (wire format, queue and drop policy, errors) and add it to `specs/INDEX.md`. Add a decision record for D1 and D2 at `docs/decisions/0001-live-streaming-transport.md`.
- Code:
  - `mq3drecon/streaming/protocol.py`: pure Python, no Open3D.
  - `streaming/receiver.py`: TCP server thread with bounded queues.
  - `streaming/replay.py`: streams an MRUK project in real time.
- CLI: `mq3drecon live-replay --project-dir <dir> --host <h> --port <p> [--speed 1.0]`.
- Tests:
  - encode/decode round trip;
  - partial socket reads;
  - bad magic and version mismatch;
  - drop-oldest behavior;
  - replay of a tiny synthetic fixture yields the expected frame sequence.

### M2: Live reconstruction, viewer, session recorder (PC only)

- Specs: write `specs/LIVE_RECONSTRUCTION.md`. Update `RECORDING_FORMATS.md` so MRUK frames may be `.jpg` as well as `.rgba`; the loader dispatches on extension and keeps the same flip rule. Update `PACKAGING.md` with a new `live` extra (Open3D) and `CLI_BEHAVIOR.md`.
- Code, under `processing/live/`, each module kept under ~300 lines:

  | Module | Responsibility |
  | --- | --- |
  | `frame_pairing.py` | Matches left/right color by timestamp (tolerance from `stereoMaxTimeDeltaSeconds`, 0.02 s) and depth to its nearest color frame. |
  | `native_depth.py` | Converts NDC depth to metric depth with [`convert_depth_to_linear`](../../src/mq3drecon/utils/depth_utils.py#L42) and builds intrinsics with [`compute_depth_camera_params`](../../src/mq3drecon/utils/depth_utils.py#L4). It reuses a per-frame function extracted from [`DepthDataIO.build_depth_dataset`](../../src/mq3drecon/dataio/depth_data_io.py#L165), so live and offline depth handling cannot drift apart. |
  | `colorize.py` | Samples left-color pixels into the depth frame: unproject depth, transform with both poses, project into color. Needed because Open3D's `integrate` takes a single extrinsic and native depth comes from a different camera than color. |
  | `integrator.py` | Owns the `VoxelBlockGrid` ([`make_color_tsdf_vbg`](../../src/mq3drecon/processing/reconstruction/rgbd_integration_common.py#L51)). Converts poses with [`Transforms.convert_coordinate_system`](../../src/mq3drecon/models/transforms.py#L164). Takes snapshots on a timer: point cloud about every 1 s, mesh on demand. |
  | `session_recorder.py` | Writes the MRUK layout: intrinsics JSON, frame metadata CSV, `mruk_stereo_pairs.csv`, color frames, depth `.raw` files, and descriptor CSVs. |

- Workflow and CLI:
  - `workflows/live_reconstruction.py`.
  - `mq3drecon live --port 9870 --config config/pipeline_config_live.yml --record-dir <dir>`.
  - It shows the Open3D window: model, current camera frustum, trajectory, and stats (fps, latency, drops, voxel-block usage).
- Tests:
  - live integrator on the synthetic fixture produces a non-empty grid;
  - recorder output loads through the existing [`MRUKImageDataIO.build_color_dataset`](../../src/mq3drecon/dataio/mruk_image_data_io.py#L86) and depth loaders;
  - a lightweight-import test confirms the protocol, receiver, and recorder import without Open3D.
- Exit: replaying the M0.2 capture shows the model growing at 5 Hz or faster with under 1 s of latency. The recorded copy also reconstructs offline.

### M3: Headset sender (QuestRealityCapture fork)

- Fork the repo, clone it next to this one, and branch `feature/live-streaming`.
- New `RealityLog.Streaming` assembly:
  - `LiveStreamClient`: TCP connection, auto-reconnect, worker-thread encode and send, bounded per-stream queues.
  - `StreamingConfig`: a new `streaming` block in `recording_config.json` with `enabled`, `host`, `port`, per-stream fps, `jpegQuality`, `downscale`, and depth sides.
- Refactor `MrukCameraRecorder` and `DepthMapExporter` so each sampled frame goes to one or more frame sinks (disk, network). Existing disk recording behavior must stay byte-identical.
- HUD: connection state, send fps, drops, and queue depth.
- Order of work:
  1. USB with `adb reverse`;
  2. Wi-Fi;
  3. outdoor hotspot.
- Exit: walk a room with the headset and watch the model grow on the laptop. The laptop recording reconstructs offline.

### M4: Live quality mode (optional)

- Live stereo depth with `FoundationStereoOnnxModel` ([foundation_stereo_onnx.py](../../src/mq3drecon/processing/stereo_depth/foundation_stereo_onnx.py#L16)):
  - CUDA or TensorRT execution provider;
  - latest-wins scheduling at 2–5 Hz;
  - rectification maps from [`compute_stereo_rectification`](../../src/mq3drecon/processing/stereo_depth/rectification.py#L34), computed once per rig and cached by the left/right relative pose.
- Optional frame-to-model ICP to reduce visible drift in the preview.

### M5: Web location export (this repo)

- Spec: write `specs/WEB_LOCATION_EXPORT.md` covering the package layout, coordinate frame, and units (meters).
- `mq3drecon export-web-location --project-dir <dir> --out <dir>` produces:

  | File | Contents |
  | --- | --- |
  | `collider.glb` | TSDF mesh decimated to roughly 100–300k triangles, meshopt or Draco compressed. |
  | `location.json` | Name, indoor/outdoor, spawn pose, floor height, bounds, file references, capture date. |
  | `splat.spz` | Splat trained from the `export-colmap` output, seeded with the TSDF point cloud. Pruned to the splat budget measured in M0.5. |

- **Coordinate frame.** The Open3D world frame used here is Y-up and right-handed, like glTF. A test should confirm that the splat, the collider, and the spawn pose line up.
- **Privacy pass for outdoor or public captures.** Blur faces and license plates in the frames before training.

### M6: Site viewer (user's site repo)

- WebXR page built on the viewer chosen in M0.5:
  - loads `location.json`, the splat, and the collider;
  - **in VR:** teleport, snap turn, and optional smooth locomotion, all constrained by the collider;
  - **on desktop and phone:** WASD/orbit controls as a fallback;
  - a loading progress bar.
- Hosting:
  - static HTTPS (WebXR requires a secure context) behind a CDN;
  - correct MIME types, CORS, and range requests for large files;
  - a gallery page listing locations.
- Exit: open a link in Quest Browser, press "Enter VR", and walk around a captured indoor and outdoor location at a stable frame rate.

---

## 7. Risks

| Risk | Mitigation |
| --- | --- |
| Open3D on Windows may lack CUDA | M0.1 check; CPU is adequate for native depth; WSL2 for stereo mode |
| Wi-Fi bandwidth outdoors | USB tether; hotspot; drop to 5 fps or 640² via config |
| Tracking drift on long walks, and on open grass outdoors | Visible early in the live preview; keep sessions short with overlap; offline pose optimization |
| Native depth reaches only ~5 m, so the live preview misses the outdoor far field | Acceptable for coverage checking; splats cover the far field in the published location |
| Voxel-block capacity: each 16³ block with tsdf, weight, and color floats is ~80 KB, so 20k blocks is ~1.6 GB | Coarser voxels outdoors (5–10 cm); show block usage in the viewer; bound the region; tiled offline mode already exists |
| Quest heat and battery with cameras, encoding, and radio running together | Measure in M0.4; reduce fps |
| Splat rendering cost and download size in Quest Browser | Splat budget from M0.5; `.spz` compression; level-of-detail for large outdoor scenes |
| Model licensing: check the FoundationStereo / Fast-FoundationStereo model license before use on a commercial site | Native-depth TSDF and splat training work without stereo models |
| Privacy in public outdoor captures | M5 blur pass; capture policy on the site |

---

## 8. Open questions

1. What is the website built on, and where is it hosted? This affects M6.
2. ~~One visitor at a time, or multi-user?~~ Answered 2026-11-06: one visitor at a time; no shared sessions.
3. Visual bar: photoreal splats (planned) or a lighter textured mesh only?
4. The `origin` of this clone is `t-34400/metaquest-3d-reconstruction`. Should this work live in a personal fork, or go upstream as PRs?
5. Is there already an MRUK capture on hand for M0.2?
