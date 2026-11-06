# Plan: Live Streaming Reconstruction and Hosted VR Locations

Status: active, 2026-11-06. Progress is tracked in `PROGRESS.md`.
Branch: `feature/live-streaming-reconstruction` (from `main` at `97d15f7`) in the
fork `arulrajesh/metaquest-3d-reconstruction`. The headset side is on
`feature/live-streaming` in `arulrajesh/QuestRealityCapture`.

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
built from the same session after recording stops, at full quality.

### Operator flow (requirement, 2026-11-06): one streamlined experience

There is no "disconnect the headset and copy files" step. The laptop's stream
recording is the master copy of every session.

1. On the laptop, run `mq3drecon live`.
   - If a headset is on USB, the command sets up `adb reverse` itself.
   - It also broadcasts a discovery beacon on the local network.
2. Put on the headset and open the capture app. It connects without any IP
   entry: over USB first, otherwise to the laptop it finds on Wi-Fi.
3. Press record and walk. The model grows on the laptop screen.
4. Press stop. The laptop closes the session and automatically builds the web
   location package: collider mesh, `location.json`, COLMAP export, and a
   splat if a trainer is configured.

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
| D1 | **Transport: plain TCP with length-prefixed, versioned binary messages.** The headset is the client and the laptop listens. **adb is never on the data path.** | Needs no extra libraries on either end and has the least overhead. The framing does not depend on the transport, so a WebSocket transport (one protocol frame per binary WebSocket message) can be added if a browser ever becomes a client, for example a live view in a phone or headset browser. WebRTC and ROS 2 add a lot of machinery for no gain here. |
| D2 | **Headset and laptop on the same Wi-Fi; the headset uses wireless adb** (user setup, 2026-11-06). Frames go directly over Wi-Fi to the laptop's LAN address. | Direct TCP over Wi-Fi is much faster than tunnelling through wireless adb. |
| D3 | **Encode on the headset:** JPEG for color (worker thread, `ImageConversion.EncodeNativeArrayToJPG`) and float16 for depth. | Raw MRUK RGBA is about 6.5 MB per eye-frame (see bandwidth table). |
| D4 | **Live depth defaults to Quest native depth.** Stereo depth (Fast-FoundationStereo ONNX) is an opt-in quality mode at a lower rate. | Native depth costs almost nothing. Stereo inference is too slow to run at full frame rate. |
| D5 | **Live reconstruction is a dense TSDF on the laptop GPU (NVIDIA Warp), fed with raw headset poses.** There is no pose optimization during the live session. *(Revised 2026-11-06: originally Open3D's CPU `VoxelBlockGrid`.)* | Open3D's Windows wheel has no CUDA. Warp runs natively on Windows with CUDA, and its kernels also run on the CPU as a fallback. Measured on the RTX 4080: integrating 320² depth into 77 M voxels (40 × 6 × 40 m at 5 cm) takes 0.9 ms; marching cubes takes 125 ms. So the live preview shows a mesh, not just points. |
| D6 | **The live viewer is a non-blocking Open3D window.** | Open3D is already a dependency. The viewer sits behind a small interface so another sink (for example Rerun) can be added later. |
| D7 | **The laptop records every streamed session as an MRUK-layout project, and that recording is the master copy.** Headset-side disk recording is off by default when streaming. | Meets the no-copy requirement, and the whole offline pipeline runs on the recording unchanged. Color is stored as the JPEG bytes received, with no re-encode; `RECORDING_FORMATS.md` gains `.jpg` frames. Defaults favor fidelity: JPEG q90 and float32 depth for both eyes. |
| D10 | **Zero-configuration connection.** The headset tries these in order:<br>1. **endpoint file:** over wireless adb, the laptop works out its own LAN address as seen from the headset and pushes `live_stream_endpoint.json` into the capture app's data folder;<br>2. **`127.0.0.1`** through `adb reverse`, a slow fallback;<br>3. **UDP discovery probe** (`MQLS?` on port 9871), which needs no adb at all. | Nobody types an IP address on the headset. |
| D11 | **Auto-build when recording stops.** The final live TSDF becomes the collider straight away; heavier steps run after it. | Completes the one-step flow. |
| D8 | **Published locations use Gaussian splats for appearance plus a simplified mesh for floor and collision.** | TSDF meshes from Quest depth look blobby and stop at about 5–10 m, so outdoor scenes would have no sky or far field. Splats trained on the camera images look photoreal and include the far field. The mesh gives visitors a walkable floor and walls. |
| D9 | **Develop and test the PC side against a replay sender before touching Unity.** | A Python sender streams an existing recording over the protocol. All PC work, and its tests, then need no headset. |

---

## 3b. Laptop GPU stack, outdoor strategy, and ideas from QuestRoomScan (2026-11-06)

[QuestRoomScan](https://github.com/genesisinteractive/QuestRoomScan) does all
of its scanning on the headset. Here the headset stays a lightweight streamer,
and the heavy steps run on the laptop's RTX 4080.

| # | Decision | Reason |
| --- | --- | --- |
| D12 | **GPU stack:**<br>• Warp: TSDF integration, marching cubes, coverage, texture-bake kernels;<br>• `onnxruntime-gpu`: FoundationStereo depth;<br>• Brush or gsplat: splat training;<br>• Open3D: mesh I/O, decimation, plane RANSAC, and the desktop viewer only. | All run natively on Windows with CUDA, so there is no WSL2 in the operator flow. |
| D13 | **Outdoor capture is a first-class mode, not just a bigger room.** | Quest depth is reliable to only about 5 m and degrades in sunlight. Outdoor scenes are large, tracking drifts, and exposure swings between sun and cloud. See the outdoor table below. |
| D14 | **Operator exclusion zones.** Before integration, reject depth inside a torso capsule hanging from the head pose (radius 0.35 m) and inside hand/controller capsules. | Keeps the operator's body and hands out of the model (QuestRoomScan idea). |
| D15 | **Scan coverage metrics** from the TSDF: observed-free, observed-solid, and unknown voxels. "Leaks" are free–unknown faces, i.e. holes. Closure = surface / (surface + leaks). The live viewer tints holes red. | Tells the operator where to look next (QuestRoomScan idea). Later, a laptop-to-headset message can show this on the headset HUD. |
| D16 | **The build pipeline produces a textured mesh as the primary web asset**, with splats optional. | A textured GLB renders cheaply in Quest Browser; splats are heavier. The texturing method comes from QuestRoomScan. |

The D16 build steps run in this order once recording stops:

1. Select keyframes, gated on motion and sharpness (Laplacian variance, head angular velocity).
2. Re-integrate on the GPU at fine resolution. The voxel size is chosen automatically to fit a GPU memory budget. Apply exclusion zones; use stereo depth outdoors.
3. Clean the mesh: drop small components, snap to RANSAC planes, smooth normals.
4. Simplify the mesh.
5. UV-unwrap with xatlas.
6. Bake the texture on the GPU from keyframes. Score each texel's candidate views by how head-on the view is, times a working-distance weight peaking at 0.8–2 m. Test occlusion against depth. Blend the top K views with exposure equalization, then sharpen and level seams.
7. Export the collider and `location.json`. Train a splat if configured.

| D17 | **Keep only what is worth keeping (Kalman).** Frames are kept for useful information gain: variance reduction above the target precision, scored by a causal dry run. Color is kept for color novelty and local sharpness. Depth is stored as 16-bit PNG, left eye only. Sessions are compacted after recording. | User request, 2026-11-06. Real data: 1.8 GB → 13 MB with 98% geometry coverage. `specs/SESSION_COMPACTION.md`. |
| D18 | **Two-way link.** The laptop streams `KEYFRAME`, `SURFACE_MESH` (Unity frame, confidence-colored), and `STATUS` back to the headset. The operator sees keyframes and generated surfaces live. | User request, 2026-11-06 (moved from FUTURE.md). |
| D19 | **Raw MRUK color principal point**, not the offline pipeline's mirrored one. | Held-out-camera test on real data: 12% lower error. |

**Outdoor strategy (D13)**

| Problem outdoors | Handling |
| --- | --- |
| Quest depth reaches only ~5 m and degrades in sunlight | Stereo depth from the passthrough image pairs (FoundationStereo on the GPU). Passive stereo is unaffected by sunlight and useful to about 10 m: error ≈ z²·δd / (f·B), about 0.15 m at 5 m and 0.6 m at 10 m. |
| Far field and sky are missing | Gaussian splats trained on all frames. Walking gives a metres-wide baseline, which recovers distant structure. Optionally, a sky dome built from upward-looking frames. |
| Walkable collision | A **ground heightfield** fitted to near-field depth, since the ground under the visitor is always observed. Full 3D walls remain an indoor feature. |
| Scale versus memory | 5 cm dense live volume (about 600 MB for 40 × 6 × 40 m). The build voxel size is chosen to fit the budget, with block-sparse storage later if needed. |
| Tracking drift on long walks or featureless grass | Short overlapping sessions. Offline pose refinement with visual SfM (known poses as priors). The live preview shows seams early. |
| Exposure swings between sun and cloud | Exposure-equalized texture blending, and per-image appearance terms in splat training. |
| Sunlight on the headset lenses | Operator rule: never point the lenses at the sun or leave the headset face-up outdoors. Focused sunlight can burn the displays. |

---

## 4. Bandwidth budget (estimates, verify in M0)

| Stream | Raw size | Sent as | At 10 fps |
| --- | --- | --- | --- |
| Color, per eye, 1280×1280 RGBA | 6.5 MB | JPEG q90, ~0.3–0.45 MB | both eyes: ~6–9 MB/s |
| Native depth, per eye (descriptor gives size, ~320×320) | ~0.4 MB float32 | `u16_ndc` (16-bit NDC), both eyes | ~4 MB/s |
| Pose / heartbeat | bytes | JSON header only | negligible |
| **Total (default)** | | | **~10–13 MB/s (~80–105 Mbit/s)** |
| **Total (weak Wi-Fi profile)** | | JPEG q85, `u16_ndc`, 5 fps | **~4 MB/s** |

- USB 3 cable: plenty of headroom.
- Wi-Fi 6 near the router: fine.
- Phone or laptop hotspot outdoors: use the hotspot profile.
- Depth precision of `u16_ndc` (step 1/65535 in NDC, near plane 0.1 m): about 0.6 mm at 2 m and 4 mm at 5 m. float16 would be about 1 cm at 2 m and 12 cm at 5 m, so it is kept only as an option.

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
2. **Synthetic capture.** Add a generator that writes an MRUK-layout project of an analytic room (color, native NDC depth, poses).
   - It is built by inverting the existing offline pipeline's conventions, so data the offline pipeline reconstructs correctly is, by construction, data the live path must also reconstruct correctly.
   - It replaces "record and `adb pull`" for development.
   - The first real dataset arrives over the stream in M3.
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

M0 results so far (2026-11-06, this laptop):

- **Python:** 3.11.13 via `uv`; `.[full]` installs cleanly (open3d 0.19.0, numpy 2.4, opencv 5.0).
- **Open3D CUDA: not available** (`open3d.core.cuda.is_available()` is `False`). The Windows wheel is CPU-only, so live integration runs on `CPU:0` by default.
- **onnxruntime:** 1.31.1, CPU and Azure execution providers only. Live stereo (M4) needs `onnxruntime-gpu` or WSL2.
- **Test baseline on `main`:** 99 passed, 3 failed. The failures already exist upstream:
  - a stale `_load_rgbd_images` import;
  - the root `__all__` check;
  - the README `pip install` wording.

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
- **Boundaryless app.** Declare `<uses-feature android:name="com.oculus.feature.BOUNDARYLESS_APP" android:required="true"/>` in the Android manifest, as QuestRoomScan does. Passthrough apps may run without a Guardian boundary. Without it the operator keeps getting the "create roomscale / stationary boundary" prompt (observed 2026-11-06), and outdoors no boundary fits.
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

### M6: Site viewer, local hosting first (user decision, 2026-11-06)

Local version now; public hosting later.

- `mq3drecon serve-locations` serves the viewer and `<workspace>/locations/` on `localhost:8080`.
- It runs `adb reverse tcp:8080 tcp:8080` over wireless adb, so the Quest browser opens `http://localhost:8080`. WebXR needs a secure context; `localhost` counts as one, while a LAN IP would not.
- Optionally, it opens the headset browser through adb.
- The viewer files ship inside this package for now and move to the site repo when public hosting starts.

Original scope, for when public hosting starts:

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
6. ~~Hosting?~~ Answered 2026-11-06: local hosting for now; public hosting later.
7. ~~Connection?~~ Answered 2026-11-06: wireless adb, with headset and laptop on the same Wi-Fi.
3. Visual bar: photoreal splats (planned) or a lighter textured mesh only?
4. The `origin` of this clone is `t-34400/metaquest-3d-reconstruction`. Should this work live in a personal fork, or go upstream as PRs?
5. Is there already an MRUK capture on hand for M0.2?
