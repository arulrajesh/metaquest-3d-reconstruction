# Capture controls and in-app room editing (design)

Status: **design v2, 2026-11-07, awaiting the user's review.** Not implemented. It replaces v1, which was a
button-per-feature table; the user asked for real thought on the *editing experience* instead.

This document is the spec for the editing UI in the headset capture app (`QuestRealityCapture` fork). It follows
`FUTURE.md` §1, §4 and §5.

## The problem

User feedback, 2026-11-07:

- "I can't actually move the objects with the grip trigger."
- "I want to move the individual vertices while holding the left grip."
- "There are so many buttons and almost everything is unused."
- "Put some effort into thinking how we can better the editing experience… constrained movement vs free movement, while holding a specific button or trigger."

The task differs from Gravity Sketch's. Here the user isn't sculpting new shapes. They are **making a model agree
with the room they can already see** through passthrough. So the best constraint is reality itself, and the most
precise tool is the user's own hand, touching the real corner. Every design choice below follows from that.

## What Gravity Sketch does (sources: help centre, see the end)

Gravity Sketch's patterns, and where each one lands in this design:

| Gravity Sketch pattern | Here |
| --- | --- |
| **Grab (grip) does everything to an object:** move, rotate, scale. | Grip is the only edit gesture. |
| **The other hand's trigger is the constraint modifier.** A half-pull snaps to an axis or plane. In edit mode, holding it while grabbing a point *slides* the point along its neighbours instead of moving it freely. | The *other* hand's trigger is the constraint modifier (§2). |
| **The grabbing hand's own buttons act on what it holds.** The trigger duplicates it; the red button deletes it. With nothing held, the red button is undo. | Same, while grabbing (§4). |
| **Half-pull previews, full pull commits** (adding points). | Same, when placing (§5). |
| **Off-hand thumbstick scrubs history.** | Same (§6). |
| **The grab sphere grows with the thumbstick, so one grab catches several points.** | Grab sphere radius on the thumbstick; several handles are grabbed at once, e.g. both ends of a wall (§1). |
| **"Restrict face movement" to chosen directions**, and grid snapping. | Each element type has its own default degrees of freedom; snapping uses room axes (§2–3). |

## 1. One gesture: grip grabs

- **Each hand grabs with its own grip.** Left or right does not matter; whichever hand is grabbing is the *grab hand*, and the other is the *modifier hand*. This keeps the user's request (left grip) without forcing handedness.
- **Near:** a **grab sphere** sits at the controller tip, radius 3 cm by default. The grab hand's thumbstick up/down changes it, from 1 to 25 cm, while not grabbing. Everything inside is grabbed together. This is the precise way: touch the real corner and grab.
- **Far:** with nothing in the sphere, a short **ray** (up to 5 m) picks the handle nearest to it. Use it for ceiling corners and far walls.
- **What gets grabbed:** handles appear only within about 1.5 m of a controller tip. The scanning view stays clean (user feedback: overlays in front of the eyes while scanning are unpleasant). X shows all handles.

  | Handle | Grabbing it… |
  | --- | --- |
  | ● floor corner | moves that corner; the two walls meeting there follow |
  | ● ceiling corner | sets the ceiling height (all walls) |
  | ▬ wall midpoint (bar) | moves the whole wall, staying parallel; both corners follow |
  | ● opening corner | resizes the door or window on its wall |
  | ▬ opening body | slides the door or window along its wall |
  | ● box corner | resizes the box from the opposite corner |
  | ■ box face (centre) | push/pull that face (SketchUp-style), only along its normal |
  | ◆ box top (centre) | moves the box on whatever it rests on, and turns it about vertical |
  | the mini model (on the left wrist) | the same handles, in miniature: ceiling corners and far walls without walking or stretching |

## 2. Constrained by default, free on demand (modifier hand trigger)

Rooms are architectural, so **the default movement is already constrained to what makes sense for that element**.
The modifier hand's trigger changes the constraint while held:

| Modifier hand trigger | Constraint | Typical use |
| --- | --- | --- |
| released (default) | **Natural:** floor corners move on the floor plane; walls move along their normal; boxes slide on their support surface and turn about vertical; openings stay on their wall; box faces move along their normal | Almost everything |
| **half-pull** (held) | **Axis lock:** after 2 cm of motion the movement locks to the dominant **room axis** (along or across the room's main walls, or vertical). A coloured guide line shows it. Release the trigger to unlock. | Squaring a corner, lining up a wall, raising something straight up |
| **full pull** (held) | **Free:** full 6-DoF, ignoring the natural plane. A floor corner may leave the floor; a box may tilt. | The rare odd case: sloped ceiling edges, a leaning shelf |

- **Room axes, not world axes.** Axis lock and angle snapping use the room's own frame, aligned to its longest wall, so "straight" means parallel to the walls. Headset world axes are arbitrary.
- **Visual cue:**
  - natural constraint: a faint patch of the active plane around the handle;
  - axis lock: a long coloured line (room-X red, room-Z blue, vertical green);
  - free: no guide.

## 3. Snapping: reality first

Snapping is on while grabbing. Candidates are listed in priority order, each with a small label and a **haptic tick** when
it engages:

1. **Reality.** The grabbed point snaps to the real surface under it: the Quest depth or the Space Setup global mesh, ray-cast along the constrained direction.
   - Corners snap to where two real wall surfaces meet (intersection of nearby depth planes).
   - Box faces snap flush to the real object's surface.
2. **Alignment with other elements** (SketchUp-style inference): a dashed line appears when the point lines up with another corner along a room axis, sits on an edge's extension, or makes a wall parallel or perpendicular to another.
3. **Angles:** 90° and 45° between walls, and box yaw to the room axes.
4. **Increments:** lengths and offsets to 1 cm, or 5 cm with a wider grab sphere.

- **Bypass:** the grab hand's thumbstick click, held, ignores all snapping for that drag.
- **Live dimensions** while dragging:
  - lengths of both walls at a corner;
  - box W × D × H;
  - door width × height and its distance to the nearest corner;
  - the gap to the real surface when not snapped.

## 4. While holding: the grab hand's own buttons act on what it holds

| Grab hand, while grabbing | Action |
| --- | --- |
| **Trigger** | **Duplicate** the held box or opening; the copy stays in the hand (Gravity Sketch). For chairs, windows in a row. |
| **Lower face button** (A or X) | **Delete** what is held (Gravity Sketch's red button while grabbing) |
| **Thumbstick flick** | **Nudge** 1 cm along the current constraint, for fine placement without hand shake |
| **Thumbstick hold** (push and keep) | **Precision:** movement scaled 1:4 while held, for steady far-ray edits |
| **Both grips on one box** | Two-handed: hand distance resizes along the axis between the hands, twisting turns it about vertical; same constraints and snapping |

## 5. Not holding anything: add with the trigger (half-pull preview, full pull commit)

- **Tool:** the trigger adds at the controller tip or ray, with the hand's **tool**: corner, door, window, box, entrance.
  - The thumbstick left/right changes the tool, shown as a small icon on the controller.
  - It starts on **entrance** before recording and **box** during the second pass.
- **Half-pull** shows a snapped ghost:
  - a door fitted to the wall, standard 0.9 × 2.0 m;
  - a box sitting on the floor under the ray, sized to the real object by its depth silhouette if it can be found;
  - a corner inserted on the nearest floor edge.
- **Full pull** commits it.
- **Labels:** a new box gets a label from its size (table, bed, storage…). The thumbstick left/right while it is highlighted relabels it.

## 6. The rest, few and fixed

| Input | Action |
| --- | --- |
| Left **menu ☰** | Start/stop recording (X no longer does) |
| **Upper face button** (B or Y), tap | **Undo** (Gravity Sketch's red button with nothing held) |
| Modifier-hand thumbstick, back/forward (not grabbing) | Scrub history: undo/redo several steps, with a ghost preview |
| **Y** hold 1 s | Open Space Setup. Hold 3 s: reset the room to Space Setup and discard edits. |
| **X** | Show all handles / proximity only |
| **A** (during recording, not grabbing) | "Detail here": 2 s of every frame, kept |
| Wrist **palm up** (left) | Small palette: snapping on/off per kind, grab-sphere size, handedness swap, room name |

There is no edit mode switch: editing is always one grip away, scanning continues meanwhile, and nothing appears
in front of the eyes unless a hand is near it.

## Data and persistence

- **The editable room:**
  - **floor polygon:** corners on the floor plane;
  - **ceiling height**, per wall later;
  - **openings:** on a wall edge, with offset, sill, width and height;
  - **boxes:** label, centre, yaw, size.
- Initialised from Space Setup. Saved on the headset per Space Setup room UUID, with undo history per session.
- Sent as `ROOM_MODEL` with `"edited": true`. The laptop export, spawn and floor logic need no changes.

## Build order

1. Editable room model, persistence, `ROOM_MODEL` from it.
2. Grab sphere + ray, handles with proximity reveal, natural constraints, grip drag, undo/redo, haptics. *(This answers the immediate request.)*
3. Modifier trigger: axis lock (half) / free (full), room axes, guide visuals.
4. Snapping: reality, alignment inference, angles, increments, live dimensions.
5. While-holding actions: duplicate, delete, nudge, precision; two-handed box.
6. Trigger tools: add corner, door, window, box, entrance, with half-pull ghosts; labels.
7. Mini model as an editing surface; wrist palette.

## Open questions for the user

1. Is "the other hand's trigger is the constraint modifier" right? The alternative is a modifier on the grabbing hand itself, e.g. a thumbstick press.
2. Should natural constraints be the default, with free movement on full pull? Or should free be the default, with constraints on demand?
3. Handles only near the hands (proximity reveal, X to show all), or always visible while not recording?

## Sources

- Gravity Sketch controllers layout: https://help.gravitysketch.com/hc/en-us/articles/9380640515485-controllers-layout
- Gravity Sketch edit objects: https://help.gravitysketch.com/hc/en-us/articles/5902620122269-Edit-Objects
- Gravity Sketch editing properties: https://help.gravitysketch.com/hc/en-us/articles/5902621323293-Editing-Properties
- Gravity Sketch grid: https://help.gravitysketch.com/hc/en-us/articles/5912795706397-Grid
