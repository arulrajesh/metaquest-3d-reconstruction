# Session Compaction Specification

## Purpose

This document defines how a recorded session is reduced to the frames worth
keeping, using Kalman information gain and blur, and what a compact session
contains.

The Kalman TSDF that produces the scores is specified in
`LIVE_RECONSTRUCTION.md`. File formats are specified in `RECORDING_FORMATS.md`.

---

# Principle

A frame is worth keeping when it adds **useful** information to the map. For
each near-surface voxel, the useful information is the variance reduction
**above the target precision** (default half a voxel):

```text
useful = max(P_before − P_target, 0) − max(P_after − P_target, 0)
```

* Re-observing surface that is already precise contributes zero. This is the redundant data.
* Firming up an uncertain or far surface, or seeing a new one, contributes positively.

Scoring is **causal**. Each frame is scored by a *dry run* (no state change)
against a volume holding only the frames kept so far, and it is integrated only
if kept. A dropped frame therefore never makes later frames look redundant.

Depth and color are judged separately. Depth is cheap and geometry needs
repeated readings. Color is the expensive texture source and only needs sharp,
well-spread views.

---

# Depth Keyframes

A left depth frame is kept when any of these holds, checked in order:

| Rule | Reason code |
| --- | --- |
| It is the first frame | `first` |
| The pose Kalman filter reports a tracking jump | `segment_start` |
| Useful information ≥ `depth_min_info_gain` (default 1000 voxel-equivalents) | `informative` |
| At least `max_gap_s` (2 s) since the last kept depth frame, and the operator is moving | `coverage_gap` |

Otherwise the frame is dropped (`redundant`). The default threshold was chosen
on real Quest 3 data: an active 14 s scan kept 76% of its depth frames and
covered 99% of the model built from all frames, within 3 cm.

---

# Color Keyframes

A color keyframe is wanted at a kept depth frame when **all** of these hold:

* the frame is not extremely blurred: the pose filter's angular speed × assumed exposure (4 ms) × focal length is at most `max_blur_px` (12 px);
* it is the first color keyframe, a segment start, or at least `color_min_gap_s` (0.5 s) after the previous one;
* and it either sees enough surface that still lacks good color (**color novelty** ≥ `color_min_novelty`, 0.15) or `max_gap_s` has passed while moving.

Color novelty is measured by the same dry run: it is the share of near-surface
voxels visible in the color camera whose color information is below the
coverage target (about two readings at 1.5 m).

For each wanted color keyframe, the **locally sharpest** left color frame
within `color_window_s` (0.25 s) is kept, using variance of the Laplacian at
1/4 resolution. Neighboring frames see nearly the same content, so the
comparison measures motion blur rather than texture. The right frame nearest in
time is kept with it, preserving stereo pairs.

On the same real 14 s scan: 21 color pairs were kept of 130, and 74% of
vertices were colored, against 80% with all frames.

---

# Compact Session Output

`compact_session(source, output)` writes a new session directory:

* the MRUK layout written by `SessionRecorder` with default `RecordingOptions`: JPEG color, 16-bit PNG depth, left depth eye only;
* `keyframes.csv`: every scored depth frame with `timestamp_us`, `depth_kept`, `color_wanted`, `reason`, `novelty`, `info_gain`, and `blur_px`;
* `compaction.json`: frame counts and bytes before and after, reason counts, and the policy used;
* pose logs (`hmd_poses.csv`, controller pose CSVs), copied unchanged.

The source is never modified. It is deleted only when `delete_source` is
requested explicitly, after the output is written.

Compaction is deterministic: identical input and policy give identical output.
