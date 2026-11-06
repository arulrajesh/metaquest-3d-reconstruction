# Web Location Export Specification

## Purpose

This document defines the walkable web location built from a recorded session,
and the local server and WebXR viewer that present it.

It owns requirements for:

* the location package layout and `location.json` schema
* mesh assets (GLB) and their coordinate frame
* floor and spawn estimation
* the local server and viewer behavior

---

# Package Layout

`build_location(session)` writes `<workspace>/locations/<id>/` and refreshes
`<workspace>/locations/index.json`.

| File | Content |
| --- | --- |
| `location.json` | Schema below |
| `visual.glb` | **Textured** mesh (default): decimated to `textured_triangles` (150k) with a baked `texture_size`² (4096²) JPEG atlas. Falls back to a vertex-colored mesh of `visual_triangles` (400k) when the session has no color frames or `textured` is off. |
| `collider.glb` | Uncolored mesh, decimated to `collider_triangles` (default 100k) |
| `thumbnail.jpg` | Middle left color keyframe of the session, longest side 512 px (when available) |

* The `<id>` is a slug of the location name.
* An existing non-empty location directory is never overwritten.
* `index.json` is `{"locations": [{id, name, kind, created_utc, thumbnail, triangles}]}`, newest first.

---

# Geometry

* **Fusion.** The session's left depth frames are fused with the Kalman TSDF (`LIVE_RECONSTRUCTION.md`) at 2 cm (indoor) or 4 cm (outdoor). Each frame uses the nearest left color frame within 300 ms. Only the first tracking segment is used.
* **Memory budget.** The volume covers the trajectory, extended by the depth range horizontally, 2.5 m below, and 1.5 m above. The voxel size grows in 10% steps until the dense volume fits 4 GB.
* **Mesh.** Extraction keeps points with posterior sigma ≤ 1 voxel.
* **Color fill.** Vertices without a color observation get colors diffused from colored neighbors for 3 rings. Others keep gray.
* **Cleanup.** Connected pieces with fewer than 1% of the triangles are removed.
* **Frame and units.** Coordinates are MQ3DRecon's Open3D world frame, which matches glTF (Y-up, right-handed), in meters.

---

# Texture Bake

Color keyframes from both eyes (at most 400, thinned evenly in time) are baked
into a UV atlas on the GPU:

1. **Unwrap.** Open3D UVAtlas with 16 parallel partitions and a 2-texel gutter.
2. **Rasterize.** Each texel gets its surface point, its face normal (oriented toward the cameras), and an interpolated vertex color used as fallback.
3. **Score views.** A view counts for a texel when it projects inside the image (6 px margin), faces it with `cos ≥ 0.2`, and is within 6 m. The score is `cos × exp(−((d − 1.4 m) / 1.2 m)²) × relative sharpness`.
4. **Occlusion.** A view is rejected when a ray cast against the mesh BVH from the texel to the camera hits geometry.
5. **Blend.** The top 3 views are blended with weights `score⁴`.
6. **Exposure.** Per-view gains are solved by regularized least squares in log space over texels seen by two views (clamped to [0.6, 1.6]), and the bake is run again with them.
7. **Finish.** Unseen texels use the fallback color. Charts are padded 8 texels and textured texels unsharp-masked (amount 0.35).

On the synthetic checker room, the median centroid color error is about 9×
lower than with vertex colors (0.006 vs 0.057).

---

# GLB Format

Each GLB is binary glTF 2.0 with a single mesh primitive in one `BIN` chunk:

* `POSITION`: float32;
* `COLOR_0` (vertex-colored meshes): **linear** RGBA as normalized uint16, alpha 1. glTF vertex colors are linear, so the camera's sRGB colors are converted.
* `TEXCOORD_0` plus one embedded JPEG `baseColorTexture` (textured meshes), with an unlit material (`KHR_materials_unlit`, double-sided). The captured lighting is already baked in.
* indices: uint32.

No normals are stored. Base64-embedded buffers must not be used.

---

# Floor and Spawn

Quest tracking uses a floor-level origin. When the median camera height is
1.2–2.1 m (a standing operator):

* the floor is the lowest substantial horizontal surface within ±0.25 m of y = 0, recorded as `floor_source: "captured_floor"`;
* if there is none, the floor is y = 0, recorded as `"tracking_origin"`.

Horizontal surfaces higher than that (beds, sofas, tables) are never taken as
the floor. Without a standing prior, the lowest substantial horizontal surface
is used (`"lowest_horizontal_surface"`).

The spawn point is the first depth camera's position projected to the floor.
`yaw_deg` is the rotation about +Y applied to −Z that gives that camera's
horizontal forward direction.

---

# location.json

```json
{
  "format": "mq3drecon-web-location", "version": 1,
  "id": "living-room", "name": "Living room", "kind": "indoor",
  "created_utc": "2026-11-06T13:07:07+00:00", "source_session": "…",
  "units": "meters", "frame": "glTF: Y-up, right-handed",
  "floor_y": 0.0, "floor_source": "tracking_origin",
  "spawn": {"position": [x, y, z], "yaw_deg": 44.0},
  "bounds": {"min": [x, y, z], "max": [x, y, z]},
  "assets": {"visual": "visual.glb", "collider": "collider.glb", "thumbnail": "thumbnail.jpg", "splat": null},
  "stats": {"visual_triangles": 0, "collider_triangles": 0, "voxel_size_m": 0.02, "frames": 0}
}
```

---

# Local Server

`mq3drecon serve-locations` serves:

* the packaged viewer at `/`;
* `<workspace>/locations` at `/locations/`, on `127.0.0.1:<port>` (default 8080).

Paths resolving outside those two roots must not be served. With adb, it runs
`adb reverse tcp:<port> tcp:<port>` for each headset, so the Quest browser can
open `http://localhost:<port>`. WebXR needs a secure context, and localhost is
one. `--open-headset` also opens that URL in the headset browser.

---

# Viewer

* **Rendering.** The visual mesh is unlit: textured when it has a base color texture, otherwise vertex colors.
* **Collision.** The collider is invisible and used for teleport ray casts, together with an invisible floor plane at `floor_y` covering the bounds plus 2 m.
* **In VR (`local-floor`)** the user starts at the spawn point:
  * **Teleport.** Hold the trigger to aim, release to teleport. Only surfaces with a normal `|y| ≥ 0.6` are targets. The head moves horizontally onto the target, and the floor height moves to the target's height.
  * **Snap turn.** The thumbstick turns 30° around the head.
* **On desktop:** orbit controls starting at the spawn point at eye height.

three.js is loaded from a pinned version (0.170.0).
