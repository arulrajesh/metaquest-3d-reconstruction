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
| M6 Local site viewer | **v2:** `serve-locations` (gallery, three.js WebXR viewer, `adb reverse` 8080). v1 (`a44ec28`) had teleport and snap turn. **v2 adds joystick movement in the style of Google Earth VR** (user request, 2026-11-06), specified in `specs/WEB_LOCATION_EXPORT.md`. The left stick glides where the controller points, with wall sliding, 40 cm step-up and ground following. The right stick snap-turns, flies up/down, and lands on click. There is a comfort vignette (`&vignette=0` turns it off) and desktop WASD/QE. The collider now has a BVH (`three-mesh-bvh` 0.9.16), so a ray costs 0.03 ms instead of 20 ms; without it, the v1 teleport aim alone would have hurt frame rate on Quest. Checked in desktop Chrome: simulated walks stop 0.25 m from each wall and slide into corners, and WASD/QE move at the set speeds. **The VR stick path is not yet tried in the headset.** |

## Current state (read first)

- **The first streamed recapture is done (2026-11-06, 17:55): `~/MQ3DRecon/sessions/20261106_175535_2`.** The streaming app, live fusion and headset feedback all ran end to end.
  - **Stream:** direct Wi-Fi TCP (endpoint file) at about 5 color and 7.5 depth fps, with 0 headset drops. Fusion takes 20–50 ms per keyframe.
  - **Compaction:** 309 MB → 84 MB, keeping 143 color and 500 depth keyframes over 144 s.
  - **Session split, caused by me:** restarting the receiver mid-session (to move the app off adb reverse) split off the first 14 s into an unfinalized `20261106_175535`. The app resumed into `..._2`.
    - Lesson: never restart `mq3drecon live` while the app is recording.
    - Improvement to make: the receiver should resume into the existing session folder, not create `_2`.
  - Closing the Open3D preview window ends `mq3drecon live`. That happens after the session is saved.
  - **Floor:** the boundaryless app's tracking origin is about 0.6 m *below* the floor (head y between 1.4 and 2.4). The floor prior correctly declined, and the lowest substantial horizontal surface (0.63 m) is right.
  - **Real-mesh UV unwrap had three failures, all fixed (`8899b9e`, `6291301`):**
    - edge/vertex non-manifold (fins, bowties);
    - inconsistent winding after quadric decimation (the fused mesh is orientable, the decimated one isn't): BFS `orient_consistently`;
    - a mirrored two-triangle "pillow" (same vertex set, opposite winding) that passes Open3D's checks but fails UVAtlas: `same_vertex_set_mask`.
    - UVAtlas can also *crash the process* (access violation) on degenerate input. Never feed it uncleaned meshes, even in tests.
  - **Root cause of the poor look: MRUK color intrinsics were in sensor pixels (fixed, `302d26d`).** MRUK's `focalLength` and `principalPoint` refer to the 1728² sensor; frames are 1280². The code scaled from `resolution`, so the focal length was 35% too long and the principal point ~225 px off. It affected vertex colors, the texture bake, splats, compaction blur, and the upstream offline loader. The fix is `dataio/mruk_intrinsics.image_pinhole` (Meta's crop-and-scale mapping).
    - Splat PSNR on this session (train/holdout): **23.9/21.6 → 32.2/30.1 dB**.
    - The exported `.splat` (DC color only) at never-trained views scores 24–33 dB. Side-by-sides: scratchpad `uvdebug/holdout_v2.jpg`.
    - It now looks close to photoreal for the curtain, wardrobe and desk. Weaknesses: soft close-up detail (clothes) and smeared near-camera objects.
    - Ruled out on the way: stereo calibration (constant 6.38 cm baseline, each eye alone equally bad) and pose/exposure (`pose_opt`/`exposure_opt` gave +0.3/+0.45 dB, now opt-in).
    - The raw principal point was re-confirmed under the correct scaling (mirrored: −1.35 dB).
  - **Locations:**
    - **`recapture-2026-11-06`** (current, correct intrinsics): textured mesh (80% textured, 4096²) and 907k splats.
    - Superseded, kept until the user deletes them: `20261106_175535_2` (vertex colors) and `recapture-2026-11-06-textured` (wrong intrinsics).
  - **Next quality steps:**
    - SH in the export: `.splat` drops view-dependent color. SPZ or `.ply` would keep it.
    - More color keyframes for splats: compaction keeps ~1/s, tuned for TSDF.
    - Longer training: 7k iterations now, 30k is typical.
    - Fill grey (unseen) texture regions.
    - Floor holes.
- The first stock-app capture (`20261106_120328`) was rejected ("garbage") and deleted. The user deleted all old stock-app recordings from the headset (2026-11-06), freeing 15 GB.
- **The streaming APK builds (2026-11-06, fork `fb37a71`).** `Builds/QuestRealityCapture-live.apk` is 64 MB and installs as `com.arulrajesh.QuestRealityCaptureLive` ("QuestRealityCapture Live"). A first build takes about 15 min; later builds take about 3–8. `Tools/watch_build.ps1` shows a live progress bar. Issues found on the first real build:
  - Our only compile error: `Camera.main` resolved to the `RealityLog.Camera` namespace (fixed).
  - **Expected on a fresh `Library/`:** builtin ShaderGraph 17.4.0 fails its first compile ("GUID could not be found"). Unity's API updater then rewrites it to `UnityEngine.GUID`, and the recompile succeeds. **Don't revert those PackageCache edits.**
  - Package Manager sometimes fails with `EPERM` renaming `com.unity.collections` into `Library/PackageCache`, probably because antivirus briefly holds the folder.
    - Fix: copy `Editor/Data/Resources/PackageManager/BuiltInPackages/<pkg>` to `Library/PackageCache/<pkg>@<first 12 chars of <pkg>.sha1>`.
  - Unity needed `%LOCALAPPDATA%\Unity\Caches` to exist (now created).
  - The build restarts the adb server, which drops wireless adb. Afterwards, run `adb connect 192.168.0.86:5555`.
  - **Installed on the headset (2026-11-06).** The user granted spatial-data, headset-camera and camera permissions. The **left controller menu button** starts and stops recording.
- **Unity install, done (2026-11-06).**
  - Unity Hub 3.22.5 is installed by winget as an **MSIX** app at `C:\Program Files\WindowsApps\UnityTechnologies.UnityHub_3.22.5.65535_x64__2vrhnee42bhxm\app\Unity Hub.exe`.
  - The Hub CLI only works after `Remove-Item Env:ELECTRON_RUN_AS_NODE`. VS Code sets it to 1, which turns the Hub into plain Node ("Cannot find module '--headless'").
  - The editor install path is set to `C:\Users\Arul\Unity\Hub\Editor` (no admin needed). Install command: `& $hub -- --headless install --version 6000.4.5f1 --changeset cc83ebd631f8 --module android --childModules`.
  - **The user must sign in to Unity Hub once** (a free Unity ID) and activate a Personal license. Batch builds refuse to run without one, and Personal licenses can't be activated from the command line.
- **Capture routine:**
  1. Run `mq3drecon live --adb <SDK adb>`. It pushes the endpoint once the app has been launched at least once.
  2. Launch "QuestRealityCapture Live".
  3. Press the left menu button to record and again to stop. The build runs automatically.
  4. View at `localhost:8080`, served by `mq3drecon serve-locations --adb <SDK adb>`; this needs `adb reverse tcp:8080` while the headset is awake.
- **Realism gap (the user's main complaint):**
  - (a) GPU texture bake: **done**, and it now works on real meshes;
  - (b) Gaussian splats: done, but real-data quality needs pose and exposure optimization (in progress);
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
