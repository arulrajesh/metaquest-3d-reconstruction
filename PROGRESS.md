# Progress

Plan: `PLAN.md`. Update this file at the end of every session and commit it.

Last updated: 2026-11-06

## Milestones

| Milestone | Status |
| --- | --- |
| M0 Feasibility | **Mostly done.** The environment is checked, the synthetic capture is done, and the Warp GPU spike passes. On-device measurements are pending: JPEG encode time, heat, and WebXR frame rate. |
| M1 Protocol, replay, receiver | **Done** (`0fff734`). Spec: `specs/LIVE_STREAMING.md`. 20 tests. |
| M2 Live reconstruction, viewer, recorder | Not started |
| M3 Headset sender (QuestRealityCapture fork) | Not started. The fork is cloned and on branch `feature/live-streaming`. |
| M4 Live stereo quality mode | Deferred: needs `onnxruntime-gpu` or WSL2 |
| M5 Web location export | Not started |
| M6 Local site viewer | Not started |

## Done

- Forked `t-34400/metaquest-3d-reconstruction` and `t-34400/QuestRealityCapture` to `arulrajesh`. Remotes are `origin` (the fork) and `upstream` (t-34400).
- `.venv` with Python 3.11.13 and `.[full]`.
  - Open3D 0.19 has **no CUDA** on Windows.
  - onnxruntime 1.31 is CPU only.
- Baseline tests on `main`: 99 pass, 3 fail. The 3 failures already exist upstream; see `CLAUDE.md`.
- MRUK frames can now be `.jpg` as well as `.rgba`.
  - All decoding goes through `dataio/mruk_frame_codec.py`.
  - The existing tests still pass.
- Shared helpers `is_ndc_depth_map_valid` and `mirror_principal_point_x` are in `utils/depth_utils.py`. `DepthDataIO` and `o3d_utils` now use them, so the live and offline paths cannot drift apart.
- The `src/mq3drecon/streaming/` package is done and tested end to end over localhost. It contains:
  - `framing`, `messages`, `payloads`;
  - `bounded_queue`, `receiver`, `sender`;
  - `discovery`, `adb`, `replay`, `session_recorder`.
- The synthetic room capture (`src/mq3drecon/synthetic/`) loads with the offline loaders. Back-projected depth lies exactly on the analytic surfaces.
- **Warp GPU spike** (warp-lang 1.19, CUDA 13, RTX 4080). Marching cubes is `warp.geometry.IsoSurfaceMarchingCubes(nx, ny, nz)`, whose device follows the field. `wp.MarchingCubes` is deprecated. Warp kernels must live in `.py` files, not `exec`/stdin.

  | Case | Integrate | Marching cubes |
  | --- | --- | --- |
  | 2 M voxels, 320² depth | < 0.1 ms | |
  | 101 M voxels @ 1 cm, 1280² depth | 2.1 ms | 178 ms |
  | 77 M voxels @ 5 cm | 0.9 ms | 125 ms |

- QuestRoomScan ideas and the outdoor strategy were added to `PLAN.md` §3b as decisions D12–D16.

## Next steps

1. M2 live reconstruction. Add `warp-lang` to a new `live` extra in `pyproject.toml` and `PACKAGING.md`. Then build:
   - `processing/live/native_depth.py` (NDC to metric, intrinsics, pose);
   - `colorize.py` (sample color into the depth frame);
   - `frame_matching.py`;
   - `exclusion.py` (capsules);
   - `gpu_tsdf.py` (Warp dense TSDF with color, marching cubes, point extraction);
   - `worker.py` (drop-oldest queue, snapshots);
   - `config/live_config.py`;
   - `workflows/live_reconstruction.py` (session controller, auto-build hook);
   - the `mq3drecon live` CLI;
   - the Open3D desktop viewer.

   Test on the synthetic room for geometry and color accuracy against `surface_distance` and `scene_color`.
2. Headset: when the user has wireless adb connected, install the QuestRealityCapture v1.5.0 APK (stock, MRUK) and record a 15 s sample. Pull it once as a development sample and validate the live path on real data.
3. M5 build pipeline per D16, then M6 local site with `serve-locations`, then M3 C# sender (needs Unity 6000.4.5f1 installed by the user).

## Known issues and risks

- Whether MRUK JPEGs come out upright is unverified on a device. Unity's `EncodeNativeArrayToJPG` is expected to write an upright image from a bottom-up buffer, so the headset sends `row_order: "top_down"`. Check this with the first real stream.
- Getting the headset's IP over wireless adb: serials can be `ip:port` or mDNS names. Fall back to `adb shell ip -f inet addr show wlan0`.
- Windows Firewall will prompt the first time Python listens on TCP 9870 or UDP 9871. Allow it on private networks.

## Decision log

| Date | Decision |
| --- | --- |
| 2026-11-06 | Single visitor per location. |
| 2026-11-06 | No copy-off-headset step: the laptop recording is the master copy. |
| 2026-11-06 | Wireless adb on the same Wi-Fi; frames go over direct TCP. A WebSocket transport is possible later for browser clients. |
| 2026-11-06 | Local hosting first. |
| 2026-11-06 | Default depth encoding is `u16_ndc`. |
| 2026-11-06 | Laptop GPU does the heavy steps (QuestRoomScan ideas). NVIDIA Warp for TSDF and meshing, replacing Open3D CPU. |
| 2026-11-06 | Outdoor capture is a first-class mode: stereo depth, splats for the far field, ground heightfield collider. |
