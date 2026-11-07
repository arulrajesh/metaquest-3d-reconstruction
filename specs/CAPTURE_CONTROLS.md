# Capture controls and in-app room editing (plan)

Status: **plan, 2026-11-07, awaiting the user's review.** Not implemented. It turns `FUTURE.md` §1, §4 and §5 into a
concrete design for the headset capture app (`QuestRealityCapture` fork).

## Why

User feedback, 2026-11-07:

- "I can't actually move the objects with the grip trigger."
- "I want to be able to move the individual vertices while holding the left grip trigger."
- "There are so many buttons and almost everything is unused. I can have so much more control in the first scan itself."

Today Space Setup (Y) is the only way to fix the room, and it can't drag corners from inside our app. The room model
(walls, doors, windows, furniture) drives three things:

- the SketchUp-style export;
- the entrance spawn;
- the per-object coverage HUD.

So correcting it during the first pass pays off everywhere.

## Bindings today

| Input | Action |
| --- | --- |
| Left menu ☰, also **X** | Start/stop recording (the scene binds `Start` and `Three` with `Controller.All`) |
| **Y** | Open Space Setup, not while recording |
| Right **trigger** | Point at a door to set it as the entrance |
| **A** | "Detail here": 2 s of every frame, all kept |
| **B** | Show or hide the mini model |
| Both **grips**, both **thumbsticks**, left **trigger**, wrist roll | Unused |

## The editable room

- Space Setup is the starting point. MRUK anchors are read-only, so the app keeps its own **editable room**, made of:
  - **floor polygon:** corners on the floor plane;
  - **ceiling height**, plus optional per-wall heights later;
  - **walls:** derived from the floor polygon (one wall per edge), so moving a corner moves both walls that meet there, as in RoomPlan and SketchUp;
  - **openings:** door and window rectangles on a wall (offset along the wall, sill, width, height);
  - **boxes:** furniture with a label, centre, yaw, and size.
- **Persistence:** edits are saved on the headset per Space Setup room UUID, so the next session starts from them. "Reset to Space Setup" discards them.
- **Data flow:** the edited room is what `ROOM_MODEL` sends, with `"edited": true`. The laptop's export, spawn and floor logic need no other changes.
- **The coverage grid** stays on the Space Setup global mesh. It measures capture, not layout.

## Handles and feedback

- **Handles** are small spheres:
  - every floor corner and its ceiling twin;
  - every opening corner;
  - every box corner, plus the box's top-centre for moving it.
- **Pointing:** each hand has a short ray, about 4 m, that stops at the first surface (global mesh or depth). The handle nearest the ray, within about 8° or 15 cm, **highlights**.
  - Within reach, about 25 cm, the nearest handle to the controller tip wins, so close work is direct.
- **Dimensions** are shown SketchUp-style while dragging:
  - wall lengths along the floor edges;
  - box width × depth × height.
- **Snapping**, on by default; holding the other hand's trigger halfway turns it off:
  - floor corners stay on the floor plane;
  - a corner snaps to the real surface where its ray meets depth or the global mesh, to straight angles (90°/180°) with its neighbours, and to 1 cm;
  - a box sits on the floor or on the box under it;
  - an opening stays inside its wall.

## Controller map (both hands can grab; each hand keeps a role)

Following `FUTURE.md` §4: the **left hand** is the edit and menu hand, the **right hand** the pointing and tool hand. **Grip
is grab on both hands.** The left grip is the primary grab, as requested; the right grip also grabs, which helps for
two-handed resizing.

| Input | Room pass (not recording) | Scanning (recording) |
| --- | --- | --- |
| **Left grip** (hold) | Grab the highlighted handle and drag it: corner, opening corner, box corner, or box move | Same; edits never stop the recording |
| **Right grip** (hold) | Same as the left grip | Same |
| **Both grips** on one box | Two-handed: move, rotate (yaw) and scale the box | Same |
| **Both grips** on empty space | Grab the mini model: move, rotate, scale it, and grab handles in the miniature for far corners such as the ceiling | Same |
| Left **trigger** half-pull | Snapping off while held | Same |
| Left **thumbstick** left/right | Undo / redo | Same |
| Left **thumbstick** click | Show or hide the mini model (moved from B) | Same |
| **X** | Toggle the edit handles, to hide clutter (X is removed from the record toggle) | Same |
| **Y** | Re-run Space Setup (long press: reset edits to Space Setup) | Disabled |
| Left **menu ☰** | Start/stop recording | Same |
| Right **trigger** | Click a door: entrance. Click a wall: select it. Click a box: select it. | Same |
| **A** | **Add** at the ray: on a wall, a door (thumbstick to switch to window); on the floor, a box; on a floor edge, a new corner. | "Detail here" (unchanged) |
| **B** | **Delete** the highlighted handle's element: corner, opening or box | Undo the last detail marker |
| Right **thumbstick** left/right | Cycle the selected box's label (table, couch, bed, storage, screen, lamp, plant, other) or opening type (door/window) | Same |
| Right **thumbstick** up/down | Selected box: height. Selected wall: ceiling height. | Same |
| Right **thumbstick** click | Cycle the capture phase cue: room pass ↔ detail pass | Same |
| Wrist roll, palm up (left) | Session palette: room name, reset edits, save | Same |

The wrist HUD (left hand) shows the current mode and what the buttons do in it, as a two-line legend, so nothing needs memorising.

## Phases

1. **Room pass** (before recording):
   - Space Setup loads.
   - Walk the room and drag any wrong corners and boxes into place.
   - Add missing doors, windows and boxes, and pick the entrance.
2. **Scanning** (recording):
   - The coverage grid fades as surfaces are captured.
   - Editing stays possible.
   - A = detail.
3. **Wrap-up** (after stopping):
   - Wrist summary: duration, coverage %, edits, size.
   - The edited room is sent with the session and saved on the headset.

## Build order

1. **Editable room model:** data model, initialised from MRUK; persistence per room UUID; `ROOM_MODEL` sends it (`"edited": true`). The laptop needs no changes.
2. **Handles and grab:** rays, hover, either grip drags corners and boxes with plane constraints; undo/redo; X toggle; move the mini model toggle; free X from recording.
3. **Snapping and dimensions:** surface, angle and 1 cm snapping; wall length and box size labels.
4. **Add, delete and label:** A, B, right thumbstick; openings on walls; corner insertion on edges.
5. **Two-handed:** box rotate and scale; grab, rotate and scale the mini model; edit through the miniature.
6. **Wrist palette and wrap-up summary.**

Steps 1–2 alone answer the immediate request: drag the room's vertices with the left grip.

## Open questions for the user

1. Grab with **both** grips (recommended), or the left grip only?
2. Can the room be edited **while recording** (recommended: yes)?
3. Walls as a single floor polygon with one ceiling height (recommended, SketchUp-like), or per-wall heights now (sloped or split ceilings)?
