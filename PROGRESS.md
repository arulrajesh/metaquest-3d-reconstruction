# Progress

Plan: `PLAN.md`. Future release: `FUTURE.md`. Update this file at the end of every session and commit it.

Last updated: 2026-11-07

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

- **Direction changed (user, 2026-11-07): fidelity first; live updating dropped** (`PLAN.md` §1b, M7).
  - In the headset, ours (SH3) and Spirula (SH3) were "both similar and equally bad". Hyperscape Capture is "amazing".
  - The user's hypothesis: too little detail is captured because there is too little feedback while capturing.
  - **Photo-to-model misalignment on `20261107_012159`** (scratchpad `uvdebug/misalign.py`, `misalign_ours.csv`):
    - Method: native 1728², DIS optical flow from the render of `recapture-2026-11-07-ours-sh3` to each photo, textured pixels only.
    - Median |flow| is 3.2 px on train views and 3.4 px on held-out views.

    | Cause | Size | Fix |
    | --- | --- | --- |
    | **Head turning speed** | Local residual 1.3 px below 10°/s, 2.2 px at 10–25°/s, **5.0 px at 25–50°/s**; sharpness halves. 45% of frames were above 25°/s. | Capture (slow down, with a headset cue) and blur-aware selection |
    | **Left vs right camera** | Constant mean shift, left (−1.1, +0.5) px, right (+2.1, +0.7) px: **~3 px apart horizontally**, the DC export's "doubled edges" | Software; per-eye correction being tested |
    | **Per-frame pose noise** | Global shift 1.35 px median, 4 px p90 | Software; fair per-frame refinement next |
    | **Time offset / rolling shutter** | None: fits give 0.1 ms and 1 ms with R² ≈ 0. `GetCameraPose()` already samples the head pose at the frame timestamp (MRUK source). | — |

    - The old "stereo ruled out" check (`uvdebug/splat_eyes.py`) ran **before** the intrinsics fix (19 dB holdout), so it did not count.
  - **Viewer: no fixed foveation by default** (`20e2a35`). three.js defaults to the strongest foveation, which blurs the periphery. `&fov=` and `&scale=` tune it; `&scale=1.25` is roughly native.
  - **Headset fidelity capture mode: built and installed (fork `b713f5f`, 2026-11-07 11:20). Not yet tried by the user.**
    - **Throughput.** The old pipeline topped out at 6.6 stereo pairs/s:
      - cameras sampled at 10 fps;
      - a 3 cm / 3° gate;
      - one JPEG encoder at about 75 ms per 1728² frame;
      - the network thread sent one color frame per wake.

      Now it samples at 30 fps, makes the send decision *before* `GetColors()` and the 12 MB copy, runs 3 encoder threads, uses a 1 cm / 1° gate (up to 15 pairs/s), and drains every encoded frame per wake.
    - **Blur.** No frames are sent above 30°/s. The HUD shows steady / "slower" (>20°/s) / "SLOW DOWN".
    - **Coverage by capture quality.** 5 cm voxels from Quest depth record two things: distinct slow viewpoints (moved ≥ 5 cm or 5°) and the closest sharp view.
      - Red: seen only. Yellow: partly captured. Green: ≥ 6 views, one of them within 0.8 m.
      - Finished areas fade out.
      - The laptop live-mesh overlay is off by default.
    - **To check on the headset:**
      - the real color rate (receiver line `rx color`);
      - encoder drops;
      - whether the dots read well (`pointSizeMeters` 0.03 in the scene);
      - CPU heat.
    - **Build quirk:** the first Gradle `:launcher:packageRelease` failed with `IncrementalSplitterRunnable` and no cause given; a plain re-run passed (probably a transient file lock).
      - Re-run command: Unity's `java -classpath ...gradle-launcher-9.1.0.jar org.gradle.launcher.GradleMain :launcher:packageRelease` in `Library/Bee/Android/Prj/IL2CPP/Gradle`.
      - Then copy `launcher/build/outputs/apk/release/launcher-release.apk` to `Builds/`.
  - **Per-eye rotation fix and frame dropping** (`uvdebug/eye_fix.py`, log `eye_fix.log`). Settings: 15k iterations, half res, the same 70 held-out views at native res.

    | Variant | Train views | Held-out PSNR | Fine detail |
    | --- | --- | --- | --- |
    | headset poses | 495 | 31.17 dB | 23% |
    | per-eye rotation from the mean flow | 495 | 31.47 dB | 26% |
    | fix + drop frames above 25°/s | 317 | 30.66 dB | 29% |
    | fix + drop the same number at random | 317 | 31.03 dB | 29% |

    - The fix cuts the left/right x disagreement from 3.4 to 0.9 px, but gains only +0.3 dB.
    - **Dropping fast frames after capture hurts**, being worse than random. They are better kept for coverage, and blur is better avoided at capture time.
    - Fewer views give sharper but less accurate models: views disagree, and averaging them blurs.
    - **Residual structure across all frames:** both cameras show a radial scale (+0.35/+0.55% left, +0.12/+0.37% right), which is 3–4 px at the edges. The right camera's photos are softer (sharpness 149 vs 179).
  - **Per-camera calibration refinement added** (`SplatSettings.camera_opt`, `7d09a16`, off by default).
    - Shared per camera (`BakeView.camera`): rig rotation plus focal x/y and principal point. The intrinsics are applied as a differentiable zoom/shift of the render, because gsplat has no `Ks` gradient.
    - Rig translation is off by default: on the synthetic test it drifted 9 mm with a matching +1% zoom (dolly vs zoom).
    - Test: a 2% focal error on one camera is recovered relative to the other (−2.06/−1.99%). A zoom shared by both cameras is barely observable and drifts; `camera_intrinsics_reference` pins one camera.
  - **Calibration refinement on the recapture** (`uvdebug/calib_fair.py`, log `calib_fair.log`; 70 held-out views at native res, 15k iterations):

    | Variant | Held-out PSNR | Fine detail |
    | --- | --- | --- |
    | headset calibration | 31.14 dB | 21% |
    | both cameras free | 31.70 dB | 24% |
    | **left intrinsics pinned (now the default, `0800a01`)** | **31.72 dB** | 23% |
    | plus rig translation | 31.57 dB | 23% |

    - All three refined variants learn the same thing: **both cameras pitched ~+0.16–0.19°** against MRUK's `lensOffset`, and the right camera's principal point **+1.6–1.8 px** in x.
    - There is no real radial scale (−0.04% relative). The +0.4% seen in the flow residual was not a focal error.
    - `location.json` stats now include `splat_camera_calibration`.
    - **End-to-end build with the new default works:** `recapture-2026-11-07-calibrated`, in the gallery.
      - 739k splats, 13 min of training.
      - Half-res holdout 31.97 dB, against 31.47 for the earlier `recapture-2026-11-07`.
      - Calibration matches the A/B (pitch +0.15/+0.16°).
  - **Training at scale (`750ea5b`):**
    - Images beyond `gpu_image_budget_gb` (4 GB) stay in CPU memory, and one is copied per step; this is tested.
    - `SPLAT_MAX_VIEWS` goes from 2000 to 6000.
    - Iterations come from `splat_iterations_for(views)`: 15k up to ~1500 views, then ~10 visits per view, capped at 60k.
    - Compaction still keeps the sharpest color frame per 0.25 s per eye: ~4 per second per eye, ~4800 views for 10 min.
  - **Receiver restarted on the new code (2026-11-07 ~13:20, idle):** `mq3drecon live --config ~/MQ3DRecon/live_fidelity.yaml --adb <SDK adb>`.
    - That config keeps the uncompacted recording (`<session>_full`) of the first fidelity captures, so the number of frames worth keeping per second can be measured.
    - Delete those copies once that's settled.
  - **Takeaway so far:** software fixes on existing data give about +0.6 dB. The remaining 3 px disagreement is mostly blur and pose noise in the frames themselves, so the next big lever is the new capture (denser, slower, closer), not more training tricks.

- **The first streamed recapture is done (2026-11-06, 17:55): `~/MQ3DRecon/sessions/20261106_175535_2`.** The streaming app, live fusion and headset feedback all ran end to end.
  - **Stream:** direct Wi-Fi TCP (endpoint file) at about 5 color and 7.5 depth fps, with 0 headset drops. Fusion takes 20–50 ms per keyframe.
  - **Compaction:** 309 MB → 84 MB, keeping 143 color and 500 depth keyframes over 144 s.
  - **Session split, caused by me:** restarting the receiver mid-session (to move the app off adb reverse) split off the first 14 s into an unfinalized `20261106_175535`. The app resumed into `..._2`.
    - Lesson: never restart `mq3drecon live` while the app is recording.
    - A Wi-Fi drop *within* one receiver process does resume into the same folder (30 s grace; tested in `tests/test_live_session_resume.py`, `169aba1`). Only a receiver restart splits a session.
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
  - **Done after that:**
    - Harmonic color fill for unseen areas (`66babe0`): placeholder-grey vertices 29k → 9.9k.
    - Splat defaults are now 15k iterations with densification capped at 5250 (`d6830a2`): +1.3 dB holdout at the same splat count, ~5.5 min.
    - **`recapture-2026-11-06-v3` is the current location**, built with both.
      - Splats: 918k (29 MB), holdout 31.3 dB.
      - Exported DC-only renders at never-trained views score 26–33 dB (`uvdebug/holdout_v3.jpg`).
      - Superseded, for the user to delete: `recapture-2026-11-06`.
  - **User verdict in the headset (2026-11-06): the splat is "barely passable... nowhere near photorealistic".** Research and measurements followed (full report in the chat, 2026-11-07):
    - **Hyperscape** renders its splats on Meta's servers and streams video to the Quest, after hours of cloud training. Ours renders ~900k splats in the Quest browser, without SH.
    - **Ruled out on our data:**
      - lens distortion (render error flat from image centre to corner: frames are rectified);
      - stereo calibration;
      - training resolution: full-res training is 5× slower, gives fewer splats and no PSNR gain (27.9 vs 27.9 at native res), and is equally soft.
    - **At native pixels the renders keep only ~10–25% of the photos' fine detail** (Laplacian variance). The smear looks like views disagreeing by a few pixels: pose or time-sync error, and/or too few close views.
    - **Capture levers (app rebuild):**
      - the camera API supports **1728×1728** (Horizon OS v209+), but we request 1280×1280 (`RealityLogScene.unity` `RequestedResolution`);
      - the camera streams up to 60 fps, but our gate sends ≤5 color fps (`LiveStreamSettings.colorMaxFps`), and compaction keeps ~1/s/eye;
      - frames are dim (mean 68/255): more light means less noise and blur.
    - **Training levers** (`splat_train.py` options, A/B on the recapture in progress): `pose_opt` with correct intrinsics, `absgrad`, `strategy="mcmc"` (gsplat ablation: 1M MCMC splats beat 3.2M default), bilateral-grid/PPISP exposure compensation, Quest-depth supervision (DN-Splatter style), and deblur/rolling shutter (3dgs-deblur, BAD-Gaussians, gsplat 3DGUT).
    - **Display levers:**
      - WebXR framebuffer scale (the Quest browser renders below native resolution);
      - SH in the export (Spark supports SH3 via `.ply`/`.spz`);
      - for local viewing, laptop rendering over Air Link with a native OpenXR splat viewer (e.g. RSR). This is the Hyperscape-like route and needs the user's decision.
    - The user is installing Hyperscape Capture (Beta) to compare. It's a store app, so the user installs it; it may be US-only.
    - **A/B results (2026-11-07, real recapture, 15k iterations; holdout PSNR / splats / native-res fine detail kept):**

      | Setting | Holdout PSNR | Splats | Fine detail |
      | --- | --- | --- | --- |
      | default | 31.00 | 891k | 30% |
      | pose_opt | 29.27 | — | 15% (worse) |
      | absgrad | 31.00 | 593k | 27% |
      | **mcmc 1M (now the default)** | **31.42** | **724k** | 26% |

      No training method recovers detail.
    - **The view-count test is decisive:** training on half the views (143) drops holdout PSNR **30.9 → 27.9 dB**, while train views get *sharper* (43% vs 32% detail, overfitting). More views is the strongest lever, and views also disagree at pixel level (motion blur or rolling shutter), so sharper frames matter too.
    - **Done in response:**
      - compaction keeps the sharpest color frame per 0.25 s for splats (`KeyframePolicy.splat_color_gap_s`);
      - the build trains splats on up to 2000 half-res views, uint8 on the GPU (`0179244`);
      - the live app requests 1728×1728 and sends 10 color pairs/s (fork `3e47724`). The APK rebuild is in progress.
    - **Next: install and recapture** in bright light, moving slowly, with close-up passes. Compare against Hyperscape.
  - **Recapture 2026-11-07 (`sessions/20261107_012159`).** 90 s with the new app: 1728×1728 frames, 6.6 color frames/s/eye (354 headset encode drops, harmless). Compaction kept 287+278 color and 408 depth keyframes, 197 MB.
    - The receiver had built a short test session and this one **concurrently**. They overflowed the GPU and neither finished in an hour. Fixed by serialising post-processing (`7fcf3a0`).
    - Rebuilt alone as **`recapture-2026-11-07`** in 14 min: 565 splat views, 729k splats, holdout 31.47 dB (half res, with SH).
  - **Spirula Studio trial (user approved, 2026-11-07).** Binary at `C:\Users\Arul\tools\spirula-2026.10.30\app\spirula.exe` (GPL-3.0, Vulkan).
    - Fed with `scratchpad uvdebug/export_nerfstudio.py`: Nerfstudio `transforms.json` of our poses and intrinsics, 1728² images, seed PLY, our holdout views excluded.
    - Command: `spirula train 3dgs --data-format nerfstudio --scene-center none --orientation-method none --train-resolution-divisor 2 --cap-max 1000000 --num-iterations 30000`. 8 min 50 s, 1M splats.
    - The output `splat.ply` is a standard 3DGS PLY (SH3) in our world frame (identity transform).
    - The process keeps serving its viewer after training, so kill it.
    - **Held-out comparison** (70 views, native 1728²; `uvdebug/compare_spirula.py`, side-by-sides in `uvdebug/compare_spirula.jpg`):

      | Model | Holdout PSNR | Fine detail kept |
      | --- | --- | --- |
      | ours, DC (current export) | 27.12 dB | 13% |
      | Spirula, DC | 26.84 dB | 24% |
      | **Spirula, SH3** | **29.87 dB** | **36%** |

      Visually, Spirula with SH3 is clearly sharpest (printer, curtain, grille, cot net); ours shows doubled edges.
    - **Two lessons:**
      1. View-dependent color (SH) is worth ~3 dB, and our `.splat` export drops it.
      2. Spirula's trainer (MCMC/IGS+/MRNF densification + PPISP/bilateral-grid exposure) beats ours.
    - **Test location `recapture-2026-11-07-spirula`**: Spirula's PLY (248 MB) served as `scene.ply`; Spark loads it with SH. **The user should compare it in the headset** against `recapture-2026-11-07`.
    - **Correction: the first comparison was unfair.** It set Spirula with SH against our DC-only export. Our exports now keep SH: `scene.ply` SH3 plus a `scene.splat` DC fallback (`b6da869`).
      - **Fair native-res held-out comparison** (70 views; `uvdebug/compare_splats.py`, images `uvdebug/compare_fair.jpg`):

        | Model | Holdout PSNR | Fine detail |
        | --- | --- | --- |
        | ours DC | 27.05 | 14% |
        | **ours SH3** | **31.31** | 23% |
        | Spirula SH3 | 29.87 | **36%** |

      - Visually, ours with SH3 loses the DC export's doubled edges and matches photo brightness, but is a bit soft. Spirula is crispest (printer, chair mesh, table texture) with slightly flatter, exposure-neutral brightness.
      - **Both are in the gallery for the user to judge in the headset:** `recapture-2026-11-07-ours-sh3` (181 MB PLY) and `recapture-2026-11-07-spirula` (248 MB PLY).
    - **Done: Spirula is an optional splat backend (`c49b4de`).**
      - Use `build-location --splat-backend spirula --spirula <exe>` or `live --splat-backend spirula`.
      - It falls back to gsplat if the binary is missing or fails, writes `scene.ply` (SH3), and scores held-out views with gsplat.
      - An end-to-end run on the real session is in progress (scratch workspace `ws_spirula_e2e`).
    - **Next:**
      - write a compressed SH format (SPZ) so the headset doesn't load 248 MB;
      - measure Quest frame rate at 1M SH3 splats.
  - **Next quality steps:**
    - SH in the export: `.splat` drops view-dependent color. SPZ or `.ply` would keep it, but test Spark loading in a browser first.
    - More color keyframes for splats: compaction keeps ~1/s, tuned for TSDF. This needs a new capture to measure.
    - Floor holes.
    - **First headset view (2026-11-06):** the user said the splat "looks ok" and the mesh is "horrible". VR is now splat-only by default (`6ba7908`). The grip toggles the mesh only with `&mesh=1`; the mesh remains the collider.
      - Still to hear: frame rate at ~900k splats, and how the joystick movement and comfort vignette feel.
    - `serve-locations` keeps wireless adb and the 8080 forward alive (`0387e26`, `bdb022e`). It reconnects remembered headsets (`~/MQ3DRecon/headsets.json`) every 15 s and re-applies the forward every 5 s.
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

M7, fidelity first (`PLAN.md` §1b). Live-mode work (M2–M4 extras) is paused.

1. **Laptop fixes on existing data:**
   - per-eye correction (`eye_fix.py` results);
   - fair per-frame pose refinement;
   - blur-aware frame selection.

   If they help, put them in the build.
2. **Headset fidelity capture mode** (fork):
   - on-headset coverage painting (views and closeness) plus a "slow down" cue;
   - dense frames (gate ~1 cm / 1°);
   - 5–15 min sessions;
   - no live fusion.

   Measure the encoder and Wi-Fi limits first.
3. **Training at scale:** thousands of views from CPU memory, then compare Spirula and ours on held-out views.
4. **Viewing:** Quest browser vs laptop rendering over Air Link. That is the user's decision, after 1–3.

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
| 2026-11-07 | User: "forget about the live updating and focus on fidelity". Hyperscape is the bar. M7 replaces live-mode work (`PLAN.md` §1b). |
