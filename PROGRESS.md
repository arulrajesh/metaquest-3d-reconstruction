# Progress

Plan: `PLAN.md`. Update this file at the end of every session and commit it.

Last updated: 2026-11-06

## Milestones

| Milestone | Status |
| --- | --- |
| M0 Feasibility | **Partly done.** Environment checks are done. The synthetic capture generator is in progress. On-device measurements are pending: JPEG encode time, heat, and WebXR splat frame rate. |
| M1 Protocol, replay, receiver | **In progress.** Code is written; tests and spec are pending. |
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
- `src/mq3drecon/streaming/` is written but untested:
  - `framing`, `messages`, `payloads`;
  - `bounded_queue`, `receiver`, `sender`;
  - `discovery`, `adb`, `replay`.

## Next steps

1. Write the synthetic room capture generator (`src/mq3drecon/synthetic/`). Its output must load with the existing MRUK and depth loaders, and back-projected depth must lie on the room surfaces.
2. Write tests for the streaming package:
   - framing round trip, bad magic, version mismatch, truncated stream;
   - payload encodings;
   - drop-oldest queue;
   - receiver plus sender over localhost;
   - discovery on localhost;
   - replay of a synthetic capture.
3. Write `specs/LIVE_STREAMING.md` and update `specs/RECORDING_FORMATS.md` for `.jpg` frames. Add both to `specs/INDEX.md`. Commit.
4. M2: the session recorder (`streaming/session_recorder.py`), `processing/live/*`, `config/live_config.py`, the `mq3drecon live` workflow and CLI, and the Open3D viewer.
5. M5 export, then the M6 local site with `serve-locations`, then the M3 C# sender.

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
