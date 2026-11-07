# Capture controls and in-app room editing (design)

Status: **design v3, 2026-11-07.** The user approved the direction of v2 ("looks good") and asked for a check for
contradictions and improvements. That review is at the end. Not implemented.

This document specifies room editing for the headset capture app (`QuestRealityCapture` fork). It follows
`FUTURE.md` §1, §4 and §5.

## The problem

User feedback, 2026-11-07:

- "I can't actually move the objects with the grip trigger."
- "I want to move the individual vertices while holding the left grip."
- "There are so many buttons and almost everything is unused."
- "Put some effort into thinking how we can better the editing experience… constrained movement vs free movement, while holding a specific button or trigger."

The task is not sculpting, as in Gravity Sketch. It is **making a model agree with the room the user can already see**
through passthrough. So the best constraint is reality itself, and the most precise tool is the user's own hand,
touching the real corner.

## Principles

1. **One gesture edits: grip grabs.** Grip does nothing else.
2. **Constrained by default.** Every element moves only in the ways a room allows. The other hand's trigger *changes* the constraint, the way Gravity Sketch's off-hand half-pull does.
3. **You always see what a grip will do before you squeeze.** The target highlights and a short label says how it will move.
4. **Each input has one meaning per state.** States are "idle" and "grabbing", and the wrist legend shows the current meanings. No tap-versus-hold tricks on the same button.
5. **Nothing destructive on a timer.** Every edit can be undone; reset needs a confirmation.
6. **Nothing in front of the eyes while scanning.** Handles show near the hands or the ray; text and the miniature sit on the wrist.

## Hands

- **Edit hand and tool hand.** The **edit hand** is a setting, by default the **left** (the user's request). The other hand is the **tool hand**.
  - Fixed idle roles avoid the v2 contradiction: "grab hand" and "modifier hand" only existed while grabbing.
  - Either grip can still grab. The roles only decide idle bindings and where the miniature lives.
- **While grabbing,** the grabbing hand is the *grab hand* and the other is the *modifier hand*, whichever physical hands those are.

## 1. Grabbing

- **Near:** the **grab sphere** sits at the controller tip, 3 cm by default.
  - On grab, the handle **jumps to the sphere centre**. Touch the real corner, squeeze, and the corner is where the hand is. This is the precise path.
- **Far:** if no handle is within 30 cm of the tip, a **ray** (up to 5 m) picks the handle nearest to it.
  - A ray grab moves the handle to where the ray meets the handle's constraint plane or axis (**ray-plane drag**). It does not use hand displacement times distance, so far edits stay steady.
  - Ceiling corners and far walls use this.
- **Handle visibility:** handles show within 1.5 m of a controller tip **and within 0.5 m of where a ray is pointing**, so far handles appear when aimed at. X shows all handles.
- **Hover:** before any grab, the target highlights with a label, e.g. "corner · on floor" or "wall · parallel".
- **What each handle does:**

  | Handle | Grab moves… | Natural constraint |
  | --- | --- | --- |
  | ● floor corner | that corner; both walls at it follow | on the floor plane |
  | ● ceiling corner | the ceiling height (whole room; per-wall heights later) | vertical |
  | ▬ wall midpoint | the whole wall, staying parallel; both corners follow | along the wall's normal |
  | ● opening corner | that corner of the door or window (resize) | in the wall plane |
  | ▬ opening body | the door or window | along the wall, sill fixed for doors |
  | ● box corner | that corner, opposite one fixed (resize) | in the box's own axes |
  | ■ box face | that face only (push/pull) | along the face normal |
  | ◆ box top | the whole box | on its support surface, plus yaw from the hand's twist |

- **Two grips on the same box:** two-handed. Distance between the hands resizes it along the box axis closest to the hand-to-hand direction; twisting turns it.
  - There is no modifier hand then. Snapping stays on.
  - A second grip on a *different* element is ignored in v1.

## 2. Constraint modifier: the other hand's trigger

| Modifier trigger (while the other hand grabs) | Movement |
| --- | --- |
| released | **Natural:** the element's constraint (table above), with snapping |
| **half** | **Axis lock:** after 2 cm of motion it locks to the dominant direction, shown as a coloured guide line |
| **full** | **Unsnapped:** still the natural constraint, but no snapping or inference, for the rare case reality snapping is wrong (glass, mirrors, clutter) |

- **Axes:** the **grabbed element's own frame** (along and across its wall, vertical), falling back to the room's main axes. Non-rectangular rooms then lock to the wall at hand, not to a global grid.
- **Reliable half and full pulls:** thresholds with hysteresis (half engages at 0.35 and releases at 0.25; full engages at 0.9 and releases at 0.8), and a **haptic detent** on each crossing. A half-pull then holds without fighting the trigger, and passing through half on the way to full doesn't flicker the lock.
- **Why full pull is not "free 6-DoF" (v2 said it was):** the data model (floor polygon, one ceiling height, yaw-only boxes) cannot represent a corner off the floor or a tilted box. Sloped ceilings come with per-wall heights later.

## 3. Snapping (natural and axis-locked modes)

Snapping happens **inside the active constraint**: a floor corner only snaps to targets it can reach on the floor plane.

Priority:

1. **Reality:** the real surface along the constrained direction (Quest depth or the Space Setup global mesh). Corners go where two real wall surfaces meet; box faces sit flush with the real object.
2. **Alignment inference,** as in SketchUp: in line with another corner along the element's axes, on an edge's extension, parallel or perpendicular to another wall. Shown as dashed lines.
3. **Angles:** 90° and 45° between walls; box yaw to the wall it faces.
4. **Increments:** 1 cm.

- **Sticky snaps:** a snap holds until the hand moves 2 cm past it, so a snapped point doesn't jitter. Each new snap gives a haptic tick and a label ("on wall", "90°", "aligned").
- **Live dimensions:**
  - wall lengths at a corner;
  - box W × D × H;
  - door width × height and its distance to the nearest corner;
  - the gap to the real surface when not snapped.

## 4. While grabbing: the grab hand's own controls act on what it holds

| Grab hand, while grabbing | Action |
| --- | --- |
| **Trigger** | **Duplicate** a box or opening, keeping the copy in the hand. On a floor corner: **split** (insert a corner beside it). On a ceiling corner: nothing. |
| **Lower face button** (A or X) | **Delete** what is held. A floor corner merges its two walls; no fewer than 3 corners remain. |
| **Thumbstick push** | **Nudge** 1 cm along the constraint in the pushed direction, repeating while held (v2 had both "flick = nudge" and "hold = precision" on the same stick) |
| **Thumbstick click** | **Precision** on/off for this grab: movement scaled 1:4, for steady far edits |

## 5. Idle (nothing held)

| Input | Edit hand (default left) | Tool hand (default right) |
| --- | --- | --- |
| Grip | Grab (§1) | Grab (§1) |
| Trigger, half / full | Add with the tool: half shows a snapped ghost, full places it | Same |
| Thumbstick ← → | **Undo / redo**, one step per push; the HUD shows the step | **Tool:** corner, door, window, box, entrance (icon on the controller); remembered between sessions |
| Thumbstick ↑ ↓ | Grab-sphere size (1–25 cm) | Highlighted box: change its label; else nothing |
| Lower face button | X: show all handles / proximity only | A: **"detail here"** (while recording) |
| Upper face button | Y: **open Space Setup** (not while recording) | B: show/hide the wrist miniature |
| Menu ☰ (left only) | Start/stop recording | — |
| Palm up (edit hand) | Palette: snapping kinds on/off, edit-hand swap, room name, **reset to Space Setup (with confirmation)** | — |

- **Ghosts on half-pull:**
  - a door fitted to the wall, 0.9 × 2.0 m;
  - a box sitting on the floor under the ray, sized to the real object from depth if possible;
  - a corner on the nearest floor edge;
  - entrance: the door under the ray highlights.
- **New boxes** are labelled from their size; the tool-hand stick relabels the highlighted one.

## 6. The miniature

- It lives **on the tool hand** (opposite the edit hand), level and room-aligned, so the edit hand is free to work in it. v2 put it on the left wrist while also wanting left-grip edits.
- **Grab its base** with either grip and release it to **park** it in mid-air, world-locked, for editing far corners in miniature. Grab the base again to bring it back to the wrist.
- Edits inside it use the same handles, constraints and snapping, scaled.

## 7. One source of truth

- **The edited room** feeds everything:
  - the outlines;
  - the miniature;
  - the per-object coverage % on the HUD (memberships recomputed from the edited boxes, not from Space Setup's);
  - `ROOM_MODEL`.

  Only the coverage *grid* stays on the Space Setup global mesh, because it measures what was captured, not the layout.
- **`ROOM_MODEL`** goes out on grab release and on undo/redo (debounced), with `"edited": true`, not continuously.
- **Persistence:**
  - Saved on the headset per Space Setup room UUID.
  - Re-running Space Setup can change the UUID. The app then says so and offers to re-apply the previous edits, matched by position.
  - Undo history is per session.

## Data model

| Element | Fields |
| --- | --- |
| Floor polygon | corners on the floor plane |
| Ceiling | height (per wall later) |
| Openings | wall edge index, offset, sill, width, height, kind (door/window), entrance flag |
| Boxes | label, centre, yaw, size |

Initialised from Space Setup.

## Build order

1. Editable room model, single source of truth (§7), persistence, `ROOM_MODEL` from it.
2. Grab sphere (handle jumps to the tip) + ray with ray-plane drag, hover labels, handle visibility, natural constraints, undo/redo (edit-hand stick), haptics. *(Answers "move vertices with the left grip".)*
3. Modifier trigger with hysteresis and detents: axis lock (element frame) and unsnapped; guide visuals.
4. Snapping: reality, inference, angles, increments, sticky snaps, live dimensions.
5. While grabbing: duplicate/split, delete/merge, nudge, precision; two-handed box.
6. Tools and ghosts: corner, door, window, box, entrance; labels.
7. Miniature on the tool hand, park/unpark, editing in it; palette with reset confirmation.

## Review of v2 (2026-11-07)

**Contradictions fixed:**

| # | v2 | Problem | v3 |
| --- | --- | --- | --- |
| 1 | Idle actions bound to the "grab hand" or "modifier hand" | Those roles only exist while grabbing, so idle bindings were undefined | Fixed edit hand / tool hand for idle (§5) |
| 2 | Grab-hand stick: flick = nudge, hold = precision, click-hold = bypass snapping; idle up/down = sphere size *and* history scrub | One stick, several meanings in the same state | While grabbing: push = nudge, click = precision. Idle: edit-hand stick = undo/redo and sphere size, tool-hand stick = tool and label. Bypass moved to the full modifier pull. |
| 3 | Full pull = free 6-DoF, "a corner may leave the floor, a box may tilt" | The data model cannot store either | Full pull = unsnapped within the natural constraint; slopes come with per-wall heights |
| 4 | Handles shown only within 1.5 m of the hands, yet the ray grabs ceiling corners | You can't aim at handles you can't see | Handles also show near the ray's target |
| 5 | Miniature on the left wrist; edits preferably with the left grip; editing inside the miniature | The hand holding it would be the one editing | Miniature on the tool hand, and it can be parked in mid-air |
| 6 | Undo on B/Y tap *and* on the stick; Y tap = undo, Y hold = Space Setup, Y 3 s = reset | Redundant (the user's complaint); tap and hold on one button delays undo; a destructive action on a timer | Undo only on the edit-hand stick; Y = Space Setup; reset in the palette with confirmation |
| 7 | HUD per-object % and the miniature computed from Space Setup anchors | They would disagree with the edited room | Everything reads the edited room (§7) |
| 8 | Default tool "box during the second pass" | The app has no notion of a second pass | The tool is remembered |
| 9 | Duplicate and delete for "what is held" | Undefined for corners | Per element: split/merge corners, nothing for ceiling corners, at least 3 corners |
| 10 | Two-handed grab | Modifier hand missing; second grip on another element undefined | Stated: snapping stays on, no modifier, other elements ignored in v1 |

**Improvements added:**

- the handle jumps to the grab sphere;
- ray-plane drag for far edits;
- hover label before grabbing;
- axis lock in the element's own frame;
- trigger hysteresis plus haptic detents;
- sticky snaps;
- `ROOM_MODEL` only on release;
- the Space Setup UUID-change case.

## Open questions for the user

1. Edit hand: **left** (default, from your request) or right?
2. Should new elements be added with the trigger of **either** hand, or only the tool hand?

## Sources

- Gravity Sketch controllers layout: <https://help.gravitysketch.com/hc/en-us/articles/9380640515485-controllers-layout>
- Gravity Sketch edit objects: <https://help.gravitysketch.com/hc/en-us/articles/5902620122269-Edit-Objects>
- Gravity Sketch editing properties: <https://help.gravitysketch.com/hc/en-us/articles/5902621323293-Editing-Properties>
- Gravity Sketch grid: <https://help.gravitysketch.com/hc/en-us/articles/5912795706397-Grid>
