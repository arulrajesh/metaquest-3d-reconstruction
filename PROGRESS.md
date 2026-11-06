# Progress

Plan: `PLAN.md`. Future release: `FUTURE.md`. Update this file at the end of every session and commit it.

Last updated: 2026-11-06

## Milestones

| Milestone | Status |
| --- | --- |
| M0 Feasibility | **Mostly done.** Environment, synthetic capture, Warp GPU spike, and real-headset data checks are done. Pending on-device: JPEG encode cost, heat, and WebXR frame rate. |
| M1 Protocol, replay, receiver | **Done.** `specs/LIVE_STREAMING.md`. |
| M2 Live reconstruction | **Done** (laptop side), with additions not in the original plan: **Kalman TSDF**, **keyframe-only fusion**, **compaction**, **two-way feedback** (`KEYFRAME`, `SURFACE_MESH`, `STATUS`). Specs: `LIVE_RECONSTRUCTION.md`, `SESSION_COMPACTION.md`. **Not yet checked by a person:** the Open3D viewer window in an interactive run. |
| M3 Headset sender (QuestRealityCapture fork) | **In progress.** C# protocol, queue, and frame types are written (`Assets/RealityLog/Scripts/Runtime/Streaming/`, uncommitted). Still to write: the client (send plus receive), hooks in the recorders and controller, overlay rendering, HUD, boundaryless manifest. **Unity is not installed, so nothing has been compiled.** |
| M4 Live stereo quality mode | Deferred: needs `onnxruntime-gpu`. It is also the outdoor depth source (D13). |
| M5 Web location export | Not started |
| M6 Local site viewer | Not started (local hosting via `localhost:8080` plus adb reverse; see PLAN M6) |

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

## Dev samples (`C:\Users\Arul\MQ3DRecon\dev_samples\`)

- `20261106_120328_compact/` (14 MB): compact real session. Use it for real-data checks.
- `20261106_120328_headset_meta/`: CSV and JSON metadata of the full headset session.
- The raw 1.8 GB window was deleted at the user's request ("throw away garbage").
- The full 7+ GB original is still on the headset: `/sdcard/Android/data/com.t34400.QuestRealityCapture/files/20261106_120328`. **Ask the user before deleting it.**

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
