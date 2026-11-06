# Future features (next release)

Requested 2026-11-06. **Not in the current release.** The current release is
`PLAN.md` M0–M6. Each item below needs a spec before work starts, per
`specs/COMMON_RULES.md`.

The idea: the operator is already holding two controllers and can talk, so the
capture becomes **guided and annotated**, not just recorded. The laptop GPU
does the understanding.

---

## 1. Controller pointing: tell the scan where things are

| Gesture | Meaning | How it improves the result |
| --- | --- | --- |
| Point and pull the **right trigger** at a room corner | Room corner (floor–wall–wall or wall–wall–ceiling) | Fits a clean room layout of planar walls, floor, and ceiling that meet exactly at the marked corners. This gives a sharp collider and plane snapping even where depth fails: glass, black surfaces, far walls. |
| Trigger on an object's corners or edges | Object vertex | Gives an oriented bounding box or simple proxy per object, for collision, labels, and clean object meshes. |
| **A** | "Keyframe here" | Forces a high-priority keyframe, preferred by texture baking and splat training. Use it for signs, artwork, and details. |
| **B** | Undo last marker | |
| Right thumbstick left/right | Cycle the marker category (corner, object vertex, door, window, mirror/glass, "ignore this") | The category travels with every marker. |
| Hold the **right grip** while pointing | Paint a region (sweep) instead of a point, e.g. "this whole wall is glass" | |

**How the pointing ray hits a surface:** the controller pose is streamed. The
ray is hit-tested on the laptop against the live TSDF, and on the headset
against the MRUK scene mesh or depth for instant feedback. A marker stores the
ray, the hit point, the hit normal, the category, the time, and an optional
voice note.

**Prerequisites:**

- Stream controller poses. This also enables hand and forearm **exclusion capsules** (PLAN D14).
- Add a laptop-to-headset channel so markers and labels can be confirmed visually.

---

## 2. Object intelligence on the laptop GPU

- **Closed-set detection** (YOLO-class models) on keyframes, plus **open-vocabulary detection** (YOLO-World / Grounding-DINO-style) so the operator's words become detection prompts ("piano", "fireplace").
- **Segmentation** (SAM-class) per detection. Masks are lifted to 3D with depth and the TSDF and fused across views into **3D object instances** with labels and confidence.
- **Uses:**
  - labels and info hotspots in the WebXR location;
  - per-object meshes;
  - "ignore" classes: people, the operator's hands, pets, screens showing moving video;
  - better texture baking (no ghosting from moving objects).
- **Detection on device** (QuestRoomScan runs YOLO on device via Sentis) is unnecessary here; the 4080 is much faster. The headset only displays results.

---

## 3. Voice narration

- **Push-to-talk** (hold the left trigger) streams microphone audio to the laptop. This adds an `AUDIO` message to the protocol (Opus or 16 kHz PCM).
- **Transcription** with a Whisper-class model on the GPU, time-aligned with the pose stream and the active pointer ray.
- **An LLM turns narration into structured annotations:**
  - "this is the living room" → room name;
  - "that's a glass door" → material and plane hint at the pointed location;
  - "the painting above the fireplace is from 1890" → an info hotspot for visitors.
- Narration also feeds open-vocabulary detection prompts (section 2). An optional **narrated tour** can be played back in the published location.

---

## 4. Interaction principles (learned from Gravity Sketch's controller layout)

Source: [Gravity Sketch controllers layout](https://help.gravitysketch.com/hc/en-us/articles/9380640515485-controllers-layout).
These are patterns, not copied assets.

- **Two hands, two roles.**
  - The **tool hand** (dominant) does precise work: point, mark, undo.
  - The **menu/navigation hand** (non-dominant) holds menus, palettes, history, and world navigation.
  - Handedness is a setting (default right-handed). A quick gesture can swap it; Gravity Sketch bumps the controller bottoms together.
- **Half-pull to preview, full pull to commit.**
  - A half-pulled tool trigger shows where a marker would land, snapped to the detected corner, edge, or plane, plus its category.
  - A full pull places it, so misplaced markers become rare.
  - A half-pulled menu-hand trigger toggles **snap to plane or axis** for the next marker.
- **Undo is one button away** on the tool hand. The menu-hand thumbstick **scrubs history**: tilt back to step through the latest markers and keyframes.
- **Context-sensitive buttons.**
  - When the ray hovers an existing marker, the undo button deletes *that* marker.
  - When the paint grip is held, the thumbstick resizes the brush.
  - A contextual action appears only while its target exists.
- **Wrist-rotation palettes** on the menu hand, with no button press:
  - **Palm up:** session actions (pause, stop, mode, room name, save).
  - **Palm down:** view tools (coverage overlay, confidence heatmap, keyframe markers, X-ray of the live mesh).
  - These need dwell and hysteresis against accidental opens.
- **Grab to navigate.**
  - In review mode after capture, the menu-hand grip grabs the reconstructed model as a miniature dollhouse to move, rotate, and scale it.
  - Both grips together pinch-scale it.
  - This is the natural way to inspect a capture before publishing.
- **Continuous adjustments on thumbsticks**, not menus: brush radius, ray length, overlay opacity.

## 5. Full controller map (Quest Touch Plus; right-handed default, mirrored for left-handed)

| Control | Action |
| --- | --- |
| Menu hand (left) **☰ menu** | Start/stop recording (existing) |
| Menu hand **trigger**, half pull | Snap the next marker to a plane or axis |
| Menu hand **trigger**, hold | Push-to-talk narration |
| Menu hand **grip** | Capture: toggle the coverage overlay. Review: grab the dollhouse model. |
| Menu hand **X** | Pause/resume streaming without ending the session |
| Menu hand **Y** | Toggle the wrist HUD |
| Menu hand **thumbstick** | Tilt back = scrub marker/keyframe history; click = "where next?" arrow to the largest hole |
| Menu hand **wrist roll** | Palm up = session palette; palm down = view palette |
| Tool hand (right) **trigger** | Half pull = preview the marker (snapped); full pull = place it |
| Tool hand **grip** (hold) | Paint a region marker; thumbstick resizes the brush while held |
| Tool hand **A** | Force a keyframe |
| Tool hand **B** | Undo, or delete the hovered marker |
| Tool hand **thumbstick** | Cycle the marker category; click = cycle capture mode (room pass ↔ detail pass) |

The system (Meta) button is reserved by the OS.

---

## 6. Capture UI (inspired by guided capture apps such as Meta Hyperscape Capture)

These are patterns, not copied assets. Guided capture works in **phases with
visible progress**, and the operator always knows what to do next.

1. **Ready panel** (floating, before recording):
   - laptop link state and Wi-Fi strength;
   - mode (indoor/outdoor);
   - space check: boundaryless or boundary set up;
   - a big Start button.
2. **Room pass:** walk the perimeter and look around.
   - The laptop streams back a light **coverage overlay** that paints surfaces as they are captured.
   - A **closure percentage** (PLAN D15) is shown, and unscanned areas glow red.
   - Prompts: "Look up at the ceiling", "Turn around", "Cover the floor near the door".
3. **Detail pass:** walk closer to objects.
   - Guidance arrows point to low-detail or low-texture regions.
   - Markers, keyframes, and narration happen mostly here.
4. **Wrap-up panel:** duration, coverage, number of markers and notes, data size, and "Building on laptop… ready". When the build finishes, a preview of the location can be opened on the headset (local site, M6).
5. **Wrist HUD** (left controller) throughout:
   - recording time, link state, frames sent and dropped;
   - battery and thermal state;
   - a compact closure bar.

---

## 7. Protocol and pipeline additions this needs

Moved to the current release on 2026-11-06 (`PLAN.md`): laptop-to-headset
`KEYFRAME`, `SURFACE_MESH`, and `STATUS` feedback, so the operator sees
keyframes and the generated surfaces live.

Still future:

| Direction | Addition |
| --- | --- |
| Headset → laptop | `CONTROLLER_POSE` (both hands, ~30 Hz), `MARKER`, `KEYFRAME_REQUEST`, `AUDIO` |
| Laptop → headset | `COVERAGE` holes (centroids and areas for "where next?"), `GUIDANCE` (next prompt), `DETECTIONS` (labels with 3D positions), `MARKER_ACK` |
| Build | Room-layout fit from corner markers. Object instances from detection plus markers. An annotation layer in `location.json` (hotspots, rooms, materials). |
