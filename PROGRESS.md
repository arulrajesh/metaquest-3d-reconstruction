# Progress

Plan: `PLAN.md`. Future release: `FUTURE.md`. Update this file at the end of every session and commit it.

Last updated: 2026-11-06

## Milestones

| Milestone | Status |
| --- | --- |
| M0 Feasibility | **Mostly done.** Environment, synthetic capture, Warp GPU spike, and real-headset data checks are done. Pending on-device: JPEG encode cost, heat, and WebXR frame rate. |
| M1 Protocol, replay, receiver | **Done.** `specs/LIVE_STREAMING.md`. |
| M2 Live reconstruction | **Done** (laptop side), with additions not in the original plan: **Kalman TSDF**, **keyframe-only fusion**, **compaction**, **two-way feedback** (`KEYFRAME`, `SURFACE_MESH`, `STATUS`). Specs: `LIVE_RECONSTRUCTION.md`, `SESSION_COMPACTION.md`. **Not yet checked by a person:** the Open3D viewer window in an interactive run. |
| M3 Headset sender (QuestRealityCapture fork) | **Code done, never built** (fork `feature/live-streaming`: `56cbc0e`, `3111e03`, `448dbc8`, `0ce6bde`). Includes the client (send and receive, endpoint resolution, reconnect and resume, motion gate), feedback renderer (overlay, keyframe markers, HUD), stereo-instanced shader, recorder, depth, and controller hooks, boundaryless manifest, and `Tools/build_and_install.ps1`. **Verified only by `Tools/StreamingCheck`:** the code compiles as C# 9 against Unity stand-ins, and the wire format is cross-checked with Python. **Blocked on the user installing Unity 6000.4.5f1** with Android Build Support. |
| M4 Live stereo quality mode | Deferred: needs `onnxruntime-gpu`. It is also the outdoor depth source (D13). |
| M5 Web location export | **v2: textured.** Builds write a **GPU-baked texture** (D16): Open3D UVAtlas unwrap in about 5 s for 150k triangles (xatlas took 496 s), a Warp bake (best views, BVH occlusion, top-3 blend, least-squares exposure gains), and a textured unlit GLB. The vertex-color fallback is kept. On the synthetic checker room the median color error is **9× lower** than vertex colors (0.006 vs 0.057), and the bake takes 2.5 s for 80 views at 2048². Checked rendering in desktop Chrome on synthetic data only. **Not yet run on a real capture.** **Splats are done too:** gsplat (PyTorch 2.9 cu129, compiled on Windows) trains from keyframes and headset poses, seeded from observed TSDF vertices. Export is `.splat`, shown in the viewer with Spark 2.3.1 (three.js 0.181.2), and M or a grip squeeze switches to the mesh. Synthetic checker room: **36.1 dB PSNR on held-out views**, 826k splats, about 137 s of training for 7000 iterations; the whole build takes about 165 s. **Not yet tried:** a real capture, and splat stereo/performance in the Quest browser. Each `.splat` is about 27 MB (SPZ compression and a splat cap are next). |
| M6 Local site viewer | **v1 done** (`a44ec28`): `serve-locations` (gallery, three.js WebXR viewer with teleport and snap turn, `adb reverse` 8080). Checked in desktop Chrome only; **not yet tried in the headset.** |

## Current state (read first)

- **The first real capture (`20261106_120328`) was rejected by the user ("garbage, looks nothing like reality") and must not be used.**
  - Why it was bad: stock app at 3 fps with motion blur; the boundary prompt reset tracking; only a 14 s usable window (82 depth and 21 color frames); vertex colors only.
  - The laptop copies were deleted.
  - The user was asked to delete the headset copy themselves (we may not permanently delete user data):
    `adb -s 192.168.0.86:5555 shell rm -rf /sdcard/Android/data/com.t34400.QuestRealityCapture/files/20261106_120328`
- **Next user action: install Unity 6000.4.5f1** with Android Build Support (OpenJDK, SDK and NDK) and sign in to Unity Hub. Then:
  1. Run `powershell -ExecutionPolicy Bypass -File QuestRealityCapture\Tools\build_and_install.ps1 -Device 192.168.0.86:5555`.
  2. Fix whatever the first real compile finds.
  3. Run `mq3drecon live` and do a **recapture with the streaming build**: boundaryless, slow and systematic, covering the floor and ceiling, watching the overlay.
- **Realism gap (the user's main complaint).** Even a good capture will not look real with vertex colors. Priority after the recapture:
  - (a) ~~GPU texture bake (D16)~~: **done**; it applies automatically to the recapture;
  - (b) ~~Gaussian splats~~: **done** with gsplat (the user chose it, 2026-11-06); they apply automatically to the recapture;
  - (c) FoundationStereo depth (M4) for better geometry.

## Done this session (2026-11-06)

**Setup**

- Forks under `arulrajesh` with remotes `origin` (fork) and `upstream`. The `.venv` has Python 3.11.13 with `.[full]` and `warp-lang` 1.19.
- Test baseline: the 3 failures already exist upstream (see `CLAUDE.md`). Current suite: **135 pass**.

**Streaming** (`src/mq3drecon/streaming/`)

- Versioned TCP protocol; `u16_ndc` depth; JPEG color.
- Receiver where the newest connection wins, plus an outbox for laptop-to-headset feedback.
- UDP discovery, and wireless-adb endpoint push plus `adb reverse`.
- Recorder: MRUK layout, JPEG frames, 16-bit PNG depth (left eye only by default).
- Replay with timestamp filters.

**Live reconstruction** (`src/mq3drecon/processing/live/`)

| Part | Detail |
| --- | --- |
| GPU TSDF | Warp, CPU fallback. Each voxel is a Kalman filter: Quest depth noise model, flying-pixel inflation, 3σ innovation gate, outlier rate, re-initialization, and free-space carving. |
| Meshes | Keep only points with low posterior σ. |
| `score()` | Dry run returning useful information gain (above the target precision) and color novelty. |
| Pose Kalman filter | Tracking-jump detection and blur prediction. |
| Keyframe selector | Depth kept for useful gain; color kept for color novelty, spacing, and local sharpness. |
| Reconstructor | Fuses keyframes only; segments the model on tracking jumps. |
| Exclusion | Torso capsule. |
| Feedback mesh | Coarse, vertex-clustered, confidence-colored. |

**Workflows**

- `mq3drecon live`: receiver, discovery, adb, Open3D viewer, feedback publisher. Compacts in place after each session.
- `mq3drecon live-replay`, `mq3drecon make-synthetic-capture`, `mq3drecon compact-session`.

**Real-data validation** (session `20261106_120328`, 14 s window before the tracking jump)

| Check | Result |
| --- | --- |
| Floor plane | Normal [0, 1, 0]; flatness residual 5.6 mm |
| Mesh | 5 mm median posterior σ |
| Color principal point | Raw MRUK principal point is correct: held-out right-camera error 0.0546 raw vs 0.0622 mirrored (D19) |
| Compaction | 1,799 MB → **13 MB**. Geometry coverage 98.4% within 3 cm; 74% vs 80% colored vertices. |
| Novelty trace | About 0.99 for the first frame, 0.2 while turning, 0 while standing still |

## Dev samples

- **None.** The rejected capture and everything derived from it were deleted. Use the synthetic room (`mq3drecon make-synthetic-capture`) until a good capture comes from the streaming build.

## Headset facts (Quest 3 `2G0YC1ZG3H0214`, Android 14)

| Item | Fact |
| --- | --- |
| Network | Wi-Fi 192.168.0.86; laptop 192.168.0.223, same subnet |
| Wireless adb | USB once, `adb tcpip 5555`, then `adb connect 192.168.0.86:5555`. Use the SDK adb (`%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe`, v36); the adb on PATH is Touch Portal's v28, too old. |
| Git Bash | Prefix headset paths in `adb shell` with `MSYS_NO_PATHCONV=1` |
| Capture app | Stock QuestRealityCapture v1.5.0 (`com.t34400.QuestRealityCapture`). Launch with `adb shell monkey -p com.t34400.QuestRealityCapture -c android.intent.category.LAUNCHER 1`. It records only **about 3 fps per eye**, because raw RGBA writes are slow. |
| Color camera | MRUK 1280², fx 870, principal point (643, 639) |
| Depth | 320² at about 8 fps; FOV tangents L 1.376 / R 0.839 / T 0.966 / D 1.428; far plane `Infinity` |
| Measured motion | Median head angular speed 35°/s while scanning |
| Boundary | Guardian prompts interrupt the scan and **reset the tracking origin**, which caused the 2.7 m pose jump. Fix in M3: the boundaryless manifest. |

## Next steps

1. **M3 C# client** (`QuestRealityCapture`, branch `feature/live-streaming`):
   - `LiveStreamClient`: endpoint resolution, send and receive threads, reconnect with resume.
   - Hooks in `MrukCameraRecorder`, `DepthMapExporter` (readback to callback), and `RecordingSessionController`.
   - Overlay renderer for `SURFACE_MESH` (confidence-colored, toggle), keyframe markers, and a wrist HUD from `STATUS`.
   - Boundaryless manifest plus INTERNET permission.
   - Headset-side gating: motion and blur from the pose filter, so redundant frames are not even sent.
   - The user must install Unity 6000.4.5f1 with Android Build Support to build it.
2. **First live stream from the headset.** Verify JPEG orientation (`row_order`), bandwidth, and latency.
3. **M5:** textured GLB from keyframes (D16), collider, `location.json`.
4. **M6:** local WebXR site and `serve-locations` (`localhost:8080` plus `adb reverse`).
5. **M4 / outdoor:** `onnxruntime-gpu` plus Fast-FoundationStereo for outdoor depth.

## Known issues and risks

- **Upstream offline pipeline (not changed here):** `compute_o3d_intrinsic_matrices` mirrors the MRUK color principal point, causing 6–8 px misregistration. Candidate upstream issue/PR.
- **Blur exposure is assumed** at 4 ms. Image sharpness is confounded by scene content, so only extreme predicted blur (> 12 px) is rejected; otherwise the locally sharpest frame is chosen.
- **TSDF limitation:** a surface that moves *closer* hides the old one, which remains. Surfaces that move away are carved correctly.
- **JPEG orientation from Unity is unverified.** It is configurable on the headset (`streaming.jpegRowOrder`).
- **Windows Firewall** prompts on the first listen on TCP 9870 and UDP 9871. Allow private networks.
- **Wireless adb** serials can be `ip:port` or mDNS names. The device IP falls back to `ip addr show wlan0`.

## Decision log

| Date | Decision |
| --- | --- |
| 2026-11-06 | One visitor per location. Local hosting first. Wireless adb on the same Wi-Fi; frames go over direct TCP (WebSocket possible later for browser clients). |
| 2026-11-06 | No copy-off-headset step: the laptop recording is the master copy, compacted after each session. |
| 2026-11-06 | Laptop GPU does the heavy steps (QuestRoomScan ideas); Warp replaces Open3D CPU for the TSDF. |
| 2026-11-06 | Outdoor is a first-class mode: stereo depth, splats for the far field, ground heightfield collider. |
| 2026-11-06 | Kalman filtering decides what is worth keeping (D17). Two-way feedback is in this release (D18). Raw MRUK principal point (D19). |
| 2026-11-06 | Controller annotations, object detection, voice narration, and Gravity Sketch-style interaction are next release (`FUTURE.md`). |
