# Capture controls and in-app room editing (design)

Status: **design v4, 2026-11-07.** Not implemented.

- **v2:** the user approved the editing direction.
- **v3:** fixed 10 contradictions.
- **v4:** replaces the single "floor polygon + one ceiling height" room with **regions**, after this user feedback:
  - the ceiling height can change within one room (false ceiling over part of it);
  - the floor can be a step lower in part of the room;
  - ceilings can have beams;
  - "I should be able to create edges and move individual planes in or out."

Scope: room editing in the headset capture app (`QuestRealityCapture` fork). It follows `FUTURE.md` §1, §4 and §5.

## The problem

User feedback, 2026-11-07:

- "Move the individual vertices while holding the left grip."
- "There are so many buttons and almost everything is unused."
- "Constrained movement vs free movement, while holding a specific button or trigger."
- "Create edges and move individual planes in or out", for false ceilings, sunken floors and beams.

The task is not sculpting, as in Gravity Sketch. It is **making a model agree with the room the user can already
see** through passthrough. So the best constraint is reality itself, and the most precise tool is the user's own
hand, touching the real corner.

## Principles

1. **One gesture edits: grip grabs.** Grip does nothing else.
2. **Constrained by default.** Every element moves only in the ways a room allows. The other hand's trigger *changes* the constraint, as Gravity Sketch's off-hand half-pull does.
3. **You always see what a grip will do before you squeeze.** The target highlights, with a short label saying how it will move.
4. **Each input has one meaning per state.** States: idle, hovering a box or region, grabbing. The wrist legend shows the current meanings. No tap-versus-hold tricks on one button.
5. **Nothing destructive on a timer.** Everything can be undone; reset needs a confirmation.
6. **Nothing in front of the eyes while scanning.** Handles show near the hands or the ray. Text and the miniature sit on a wrist.
7. **Architectural shapes from simple operations.**
   - Split a surface with an edge, then push or pull a part.
   - Steps, bulkheads, beams and pillars fall out of that, as in SketchUp.
   - Walls stay vertical; floors and ceilings are planes (flat or sloped).

## The room model: plan regions with their own floor and ceiling

The room is a **floor plan divided into regions**, and each region has a **floor plane** and a **ceiling plane**.

| Element | What it is | How it appears |
| --- | --- | --- |
| **Vertex** | A point in plan (x, z) | Corners, step ends, beam ends |
| **Edge** | A segment between two vertices | On the outer boundary it is a **wall**; between two regions it is a **step line** |
| **Region** | A polygon of edges (holes allowed) with a floor plane and a ceiling plane | Each plane is a height plus an optional slope |
| **Solid region** | A region that is *not* room space | A pillar, chimney breast or duct shaft. Its edges become walls. |
| **Wall face** (derived) | Vertical quad from the region's floor to its ceiling | Along each boundary edge and solid-region edge |
| **Riser** (derived) | Vertical face between two floor heights | Along an inner edge whose two regions have different floors: a step |
| **Bulkhead** (derived) | Vertical face between two ceiling heights | Along an inner edge whose two regions have different ceilings: a beam side or false-ceiling edge |
| **Opening** | Door or window on a wall face: offset, sill, width, height, entrance flag | |
| **Box** | Furniture or fixture: label, centre, yaw, size, **support** (floor, ceiling or another box) | A duct or light can hang from the ceiling |

Derived faces are never edited directly. They follow the regions. One model therefore covers:

| Real feature | In the model |
| --- | --- |
| False ceiling over part of the room | Region with a lower ceiling; a bulkhead appears at its edge |
| Ceiling beam | Narrow strip region, ceiling pushed down; two bulkheads (the beam's sides) |
| Sunken floor (one step down) | Region with a lower floor; a riser appears at the step |
| Raised platform | Region with a higher floor |
| Pillar or chimney breast | Solid region |
| Wall jog or niche | Extra vertices on the boundary |
| Sloped ceiling (attic) | Region with a sloped ceiling plane |

## Hands

- **Edit hand** is a setting, by default the **left** (the user's request). The other hand is the **tool hand**. Either grip can grab.
- **While grabbing,** the grabbing hand is the *grab hand* and the other is the *modifier hand*.

## 1. Grabbing

- **Near:** the **grab sphere** sits at the controller tip, 3 cm by default. On grab the handle **jumps to the sphere centre**: touch the real corner, squeeze, and the corner is where the hand is.
- **Far:** if no handle is within 30 cm of the tip, a **ray** (up to 5 m) picks the handle nearest to it.
  - A ray grab moves the handle to where the ray meets the handle's constraint plane or axis (**ray-plane drag**), so far edits stay steady.
- **Handle visibility:** within 1.5 m of a controller tip, and within 0.5 m of where a ray points. X shows all handles.
- **Hover:** the target highlights with a label, e.g. "corner · on floor", "ceiling · up/down", "beam edge · slide".
- **What each handle does:**

  | Handle | Grab moves… | Natural constraint |
  | --- | --- | --- |
  | ● vertex | that point; every wall, step and bulkhead through it follows | in plan (horizontal) |
  | ▬ wall (boundary or solid-region edge, mid-height bar) | the whole wall, staying parallel; neighbouring walls stretch ("move a plane in or out") | along the wall's normal |
  | ▬ step line (inner edge, at floor or ceiling level) | the step or bulkhead line, staying parallel (widen a beam, move a step) | along its normal, in plan |
  | ■ region floor (centre of the region's floor) | that floor's height (sink or raise this part of the room) | vertical |
  | ■ region ceiling (centre of the region's ceiling) | that ceiling's height (drop a false ceiling, deepen a beam) | vertical |
  | ▬ ceiling edge (where a ceiling meets a wall or bulkhead) | lifts or lowers that edge only, tilting the ceiling about the region's opposite edge (attic slope) | vertical |
  | ● opening corner / ▬ opening body | resize / slide the door or window | in or along the wall plane |
  | ● box corner / ■ box face / ◆ box top | resize / push-pull / move and turn the box | in its axes / along the face normal / on its support (floor, ceiling or box) |

- **Two grips on the same box:** two-handed. Distance between the hands resizes it along the nearest box axis; twisting turns it. There is no modifier hand, snapping stays on, and other elements are ignored in v1.

## 2. Constraint modifier: the other hand's trigger

| Modifier trigger (while the other hand grabs) | Movement |
| --- | --- |
| released | **Natural:** the element's constraint (table above), with snapping |
| **half** | **Axis lock:** after 2 cm of motion it locks to the dominant direction, shown as a coloured guide line. For a ceiling-edge lift: lock the slope to the nearest 5°. |
| **full** | **Unsnapped:** the natural constraint without any snapping or inference, for when reality snapping is wrong (glass, mirrors, clutter) |

- **Axes:** the grabbed element's own frame (along and across its wall, vertical), falling back to the room's main axes.
- **Thresholds with hysteresis:** half engages at 0.35 and releases at 0.25; full engages at 0.9 and releases at 0.8. Each crossing gives a haptic detent.

## 3. Snapping (inside the active constraint)

Priority:

1. **Reality:**
   - walls and vertices snap to the real surfaces and corners (Quest depth or the Space Setup global mesh);
   - **heights snap to the real surface measured inside the region**: the median of the global mesh at that level, e.g. the false ceiling's real height;
   - step and beam lines snap to where the real surface height jumps.
2. **Alignment:**
   - in line with other vertices along the element's axes, or on an edge's extension;
   - parallel or perpendicular to walls;
   - **a neighbour's height**: pulling a step flat to its neighbour removes the step, and a haptic tick says so.
3. **Angles:** 90° and 45° between walls; box yaw to the wall it faces; slopes to 5°.
4. **Increments:** 1 cm.

- **Sticky snaps:** a snap holds until the hand moves 2 cm past it. Each snap gives a haptic tick and a label ("on wall", "90°", "ceiling 2.41 m", "flush with neighbour").
- **Live dimensions:**
  - wall lengths at a vertex;
  - **floor and ceiling heights of the region**, and the step or beam depth (difference to the neighbour);
  - beam width;
  - box W × D × H;
  - door width × height and its distance to the nearest corner.

## 4. While grabbing: the grab hand's own controls act on what it holds

| Grab hand, while grabbing | Action |
| --- | --- |
| **Trigger** | **Duplicate** a box, opening or **beam** (a strip region, placed parallel at the same spacing as the hand moves). On a vertex: **split** (insert a vertex beside it). Elsewhere: nothing. |
| **Lower face button** (A or X) | **Delete** what is held. Vertex: its edges merge (each region keeps at least 3). Step line: the two regions merge, taking the larger one's floor and ceiling (the HUD says which). Box or opening: removed. |
| **Thumbstick push** | **Nudge** 1 cm along the constraint, repeating while held |
| **Thumbstick click** | **Precision** on/off for this grab: movement scaled 1:4 |

## 5. Idle (nothing held)

| Input | Edit hand (default left) | Tool hand (default right) |
| --- | --- | --- |
| Grip | Grab (§1) | Grab (§1) |
| Trigger | Use the tool (below) | Use the tool (below) |
| Thumbstick | ← → undo / redo; ↑ ↓ grab-sphere size | **Radial menu:** push toward an icon, release to choose a tool. While hovering a box or region it shows that element's **labels or kind** instead (table, bed, storage… / room, solid). |
| Lower face button | X: show all handles / proximity only | A: **"detail here"** (while recording) |
| Upper face button | Y: **open Space Setup** (not while recording) | B: show/hide the wrist miniature |
| Menu ☰ (left only) | Start/stop recording | — |
| Palm up (edit hand) | Palette: snapping kinds, edit-hand swap, room name, auto-suggest on/off, **reset to Space Setup (with confirmation)** | — |

**Tools** (trigger: press to start, half-pull shows the snapped ghost, release or full pull to commit):

| Tool | Use | Result |
| --- | --- | --- |
| **Rectangle** | Press on a floor or ceiling corner point, release at the opposite corner. Aligned to the nearest wall. | A new region (island, or against walls). Then grab its floor or ceiling handle and push/pull. **Beam:** long thin rectangle on the ceiling, pushed down. **False ceiling:** rectangle over part, pushed down. **Sunken area / platform:** rectangle on the floor, pushed down or up. **Pillar:** rectangle on the floor, kind set to *solid* in the radial. |
| **Edge** | Press on an existing edge or vertex, release on another. Snaps to where the real surface height jumps. | Splits a region along any line (L-shaped steps, angled false ceilings). A red ghost means the line does not split anything. |
| **Vertex** | Click on an edge | Inserts a vertex, e.g. for a wall jog: insert two, then pull the wall between them |
| **Door / window** | Click on a wall | Fitted ghost (door 0.9 × 2.0 m) |
| **Box** | Click on a floor or ceiling | Box on that support, sized from the real object's depth silhouette if found |
| **Entrance** | Click a door | Visitors start there |

The tool is remembered between sessions.

## 6. The miniature

- It lives **on the tool hand**, level and room-aligned, so the edit hand is free to work in it.
- **Grab its base** to **park** it in mid-air (world-locked); grab the base again to bring it back.
- **Editing in it** uses the same handles, constraints and snapping, scaled. This is how far ceiling regions and beam lines are easiest to reach.
- **Ceiling views:** while parked, it can be **flipped** to look at the ceiling from below. Grab with both hands and rotate it 180° about the room axis; it snaps back level when released near upright.

## 7. One source of truth

- The edited region model feeds the outlines, the miniature, the per-object and per-region coverage % on the HUD, and `ROOM_MODEL`. Only the coverage *grid* stays on the Space Setup global mesh, because it measures capture, not layout.
- **`ROOM_MODEL`** goes out on grab release and on undo/redo (debounced), with `"edited": true`.
- **Persistence:** saved on the headset per Space Setup room UUID. If a Space Setup re-run changes the UUID, the app offers to re-apply the edits, matched by position. Undo history is per session.

## Starting point and suggestions

- **Space Setup import:**
  - The app loads the room with MRUK's **high-fidelity scene model** (`SceneModel.V2`) when the device provides it: several floors and ceilings at different heights, slanted ceilings, and inner walls (pillars).
  - These become regions, sloped ceiling planes and solid regions.
  - With the simple scene (V1), the room starts as one region.
- **Auto-suggest (later step):**
  - The Space Setup global mesh already contains false ceilings, beams and steps as height changes. The app clusters its horizontal surfaces into levels and proposes the missing regions as dashed ghosts.
  - Trigger on a ghost accepts it; the edit-hand thumbstick ← (undo) rejects it.

## `ROOM_MODEL` schema v2 (laptop)

- **Coordinates:** Unity world, meters, the same frame as the camera poses. Plan points are (x, z).
- **Contents:**
  - `vertices`;
  - `regions`: loop, holes, `solid`, and floor and ceiling planes, each as `y` at `ref` plus `slope` (dy/dx, dy/dz);
  - `openings`: edge, offset, sill, width, height, kind, entrance;
  - `boxes`: label, centre, yaw, size, support.
- **Laptop build:**
  - region floors and ceilings: triangulated polygons, with holes;
  - wall faces along boundary and solid edges;
  - risers and bulkheads along inner edges, where the neighbours' planes differ, split where they cross;
  - door and window cut-outs; boxes.
- **SketchUp export:** each element is a named group. The entrance spawn uses the floor of the region it lies in.
- **Compatibility:** schema v1 (anchor list) stays readable; v1 becomes one region.

## Build order

1. Region model; Space Setup import (V1 and high-fidelity V2); schema v2 with laptop geometry (steps, bulkheads, beams, solids, slopes) and tests; persistence; single source of truth.
2. Grabbing with natural constraints for vertices, walls, step lines, region floors and ceilings, boxes and openings; hover labels; undo/redo; haptics. *(Answers "move vertices with the left grip" and "move planes in or out".)*
3. Modifier trigger: axis lock (element frame) and unsnapped; hysteresis and detents; guides.
4. Snapping: reality (including region heights and height-jump lines), alignment including flush-with-neighbour, angles, increments; sticky snaps; live dimensions.
5. While grabbing: duplicate (beams, boxes, openings), split, delete and merge; nudge; precision; two-handed box.
6. Tools: rectangle, edge, vertex, door, window, box, entrance; radial menu with labels and kinds.
7. Ceiling-edge lift (slopes); miniature park/flip and editing in it; palette; auto-suggest from the scan.

## Review of v2 (2026-11-07), kept for the record

v3 fixed these. The v2 → v3 changes still hold in v4.

**Contradictions fixed:**

1. Idle bindings named the grab/modifier hands, roles that only exist while grabbing. Now fixed edit and tool hands.
2. One thumbstick had several meanings per state. Now one meaning per input per state.
3. "Free 6-DoF" was unrepresentable in the model. Now "unsnapped"; slopes come from region planes.
4. Proximity-only handles hid the far handles the ray targets. Now also revealed near the ray target.
5. The miniature sat on the hand that edits. Now on the tool hand, and it can be parked.
6. Undo was on two inputs, with tap and hold on Y and a destructive reset on a timer. Now a single undo path; reset in the palette with confirmation.
7. The HUD and miniature read Space Setup anchors, not the edits. Now a single source of truth.
8. The default tool depended on an undetectable "second pass". The tool is now remembered.
9. Duplicate and delete were undefined per element. Now defined.
10. Two-handed grab rules were missing. Now stated.

**Improvements added:**

- the handle jumps to the grab sphere;
- ray-plane drag;
- hover labels;
- axis lock in the element's frame;
- trigger hysteresis and detents;
- sticky snaps;
- `ROOM_MODEL` on release;
- the Space Setup UUID-change case.

**v3 → v4:**

- One floor polygon with one ceiling height became the region model (steps, false ceilings, beams, pillars, slopes).
- New handles: region floor and ceiling, step lines, ceiling-edge lift.
- New tools: rectangle, edge (vertex kept); duplicate works on beams.
- Delete on a step line merges regions.
- Height snapping to the real surface and flush to neighbours.
- Radial tool menu, context-sensitive for labels and kinds.
- High-fidelity Space Setup import and auto-suggest from the scan.
- Laptop schema v2.

## Open questions for the user

1. Edit hand: **left** (default, from your request) or right?
2. Do tools work with either hand's trigger (as written), or only the tool hand?
3. Anything else structural to cover: stairs (a run of steps from the rectangle and edge tools, or a dedicated stair tool), a mezzanine, double-height spaces?

## Sources

- Gravity Sketch controllers layout: <https://help.gravitysketch.com/hc/en-us/articles/9380640515485-controllers-layout>
- Gravity Sketch edit objects: <https://help.gravitysketch.com/hc/en-us/articles/5902620122269-Edit-Objects>
- Gravity Sketch editing properties: <https://help.gravitysketch.com/hc/en-us/articles/5902621323293-Editing-Properties>
- Gravity Sketch grid: <https://help.gravitysketch.com/hc/en-us/articles/5912795706397-Grid>
- MRUK high-fidelity scene model (`SceneModel.V2`, MR Utility Kit 203 source): multiple floors and ceilings at different heights, slanted ceilings, inner wall faces.
