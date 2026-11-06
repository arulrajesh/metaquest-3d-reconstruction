# CLAUDE.md

This fork extends MQ3DRecon with **live streaming reconstruction** (Quest 3 → laptop over Wi-Fi) and **walkable VR locations** (WebXR, hosted locally for now).

- `PLAN.md` holds the goal, decisions, and milestones.
- `PROGRESS.md` holds what is done, what is next, and known issues.

## Session routine

1. Read `PROGRESS.md` first: the "Next steps" and "Known issues" sections.
2. Before changing behavior, read `specs/COMMON_RULES.md`, `specs/INDEX.md`, and the topic specs the change touches. The rules are:
   - **Order:** spec, then test, then implementation, in the same change set.
   - **Size:** modules around 300 lines; split before 500. Functions around 50 lines.
   - **CLI:** CLI handlers are thin and delegate to `mq3drecon.workflows`.
   - **Inputs:** input capture data is immutable. Outputs go to explicit locations.
   - **Import boundary:** lightweight modules must import without Open3D. These are `config`, `dataio`, `models`, `streaming`, `workflows`, and `processing/live/{native_depth,colorize,frame_matching}`. Import Open3D lazily inside functions.
3. **Commit after each working step** (tests pass). Push the branch to `origin` (the fork) at milestone checkpoints. Never push to `upstream`.
4. Update `PROGRESS.md` before ending a session, and commit it.

## Repos and remotes

| Path (under `C:\Users\Arul\vr world\`) | Remote `origin` | Branch |
| --- | --- | --- |
| `metaquest-3d-reconstruction/` (this repo, Python) | `arulrajesh/metaquest-3d-reconstruction` (`upstream` = t-34400) | `feature/live-streaming-reconstruction` |
| `QuestRealityCapture/` (Unity 6000.4.5f1, C#) | `arulrajesh/QuestRealityCapture` (`upstream` = t-34400) | `feature/live-streaming` |

`main` is the base branch. Upstream `develop` is stale: 92 commits behind `main`.

## Environment (Windows 11 laptop)

- **No system Python.** Use the repo venv, created with `uv`:
  - `uv venv --python 3.11 .venv`
  - `uv pip install --python .venv -e ".[full]" pytest`
- Run Python as `.venv/Scripts/python.exe ...`.
- Tests: `.venv/Scripts/python.exe -m pytest -q`.
  - **Baseline on `main`:** 3 tests already fail upstream. These are `test_color_aligned_rgbd_loader_uses_float_color_for_float_depth`, `test_package_root_exports_are_intentionally_minimal`, and `test_readme_documents_lightweight_and_full_pip_installs`. Do not count them as regressions.
- **Open3D 0.19 Windows wheel has no CUDA.** Live integration uses `CPU:0`.
- **PyTorch 2.9.0+cu129 and gsplat 1.5.3** are in the venv. Install PyTorch from the CUDA index (`--index-url https://download.pytorch.org/whl/cu129`), never plain PyPI.
  - gsplat JIT-compiled its CUDA kernels once (about 3.5 min) with VS 2022 Build Tools.
  - `processing/web_location/msvc_env.prepare_gsplat_build()` must run **before importing gsplat**. It loads vcvars64 and strips gsplat's GCC-only `-Wno-attributes` flag, which MSVC rejects.
- **onnxruntime is CPU-only.** Live stereo depth (M4) needs `onnxruntime-gpu` or WSL2.
- **Hardware:** RTX 4080 Laptop GPU 12 GB, 31.6 GB RAM.
- **adb:** two copies exist, one on PATH from Touch Portal and one at `%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe`. Mismatched versions kill each other's server; pass `--adb` to pin one.
- **Bash tool quirk:** a literal backslash-x-zero-zero escape in a command (heredoc or `-c`) reaches the file as a real NUL byte. Write files with such escapes using the Write/Edit tools, or build the bytes in code (`bytes([0])`).
- **Unity is not installed here,** so the C# in the QuestRealityCapture fork cannot be compiled locally. Review it carefully and say so when reporting.

## User setup and preferences (decided 2026-11-06)

- **One streamlined flow:** never a "copy files off the headset" step. The laptop's stream recording is the master copy.
- **Network:** the headset uses **wireless adb** on the **same Wi-Fi** as the laptop. Frames go over a direct TCP socket, never through adb. adb only pushes the endpoint file and runs `adb reverse` for localhost.
- **Visitors:** one at a time; no multi-user.
- **Hosting:** local for now (`localhost:8080` plus `adb reverse` for the Quest browser). Public hosting comes later.

## Live architecture cheat sheet

| Item | Value |
| --- | --- |
| Wire protocol | `src/mq3drecon/streaming/` (framing, messages, payloads); spec in `specs/LIVE_STREAMING.md` |
| Ports | TCP 9870 for frames; UDP 9871 for discovery (`MQLS?` probe); HTTP 8080 for the local site |
| Endpoint file pushed to headset | `/sdcard/Android/data/com.t34400.QuestRealityCapture/files/live_stream_endpoint.json` |
| Pixel conventions | Quest **depth** intrinsics use `cx → width - cx` (`utils/depth_utils.mirror_principal_point_x`), as the offline pipeline does. MRUK **color** intrinsics use the **raw** principal point: the offline pipeline's mirroring is wrong for MRUK color (real-data test, 2026-11-06). Poses go Unity → Open3D with `Transforms.convert_coordinate_system(OPEN3D, is_camera=True)`. `.rgba` frames are bottom-up, `.jpg` frames are top-down. |
| Synthetic test data | The synthetic room capture generator inverts those conventions; use it instead of headset data in tests |
