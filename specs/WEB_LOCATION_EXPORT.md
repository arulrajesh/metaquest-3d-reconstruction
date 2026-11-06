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
| `scene.splat` | Gaussian splats (optional; present when torch and gsplat are installed and `splats` is on) |
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

# Gaussian Splats

When enabled, splats are trained on the GPU (gsplat) from the same color
keyframes and headset poses, with no structure-from-motion:

* **Seed:** fused mesh vertices that a color camera observed, at most 300k. Never-seen geometry is not seeded, because it would stay as untrained gray splats.
* **Training:** 15000 iterations by default at half resolution. Loss is 0.8·L1 + 0.2·(1 − SSIM). The SH degree ramps up to 3. gsplat's `DefaultStrategy` densifies until 75% of training, but stops at iteration 5250 at the latest (`max_refine_iter`). Later iterations refine the existing splats without adding more.
  * Measured on the real recapture (holdout PSNR / splats):

    | Run | Holdout PSNR | Splats |
    | --- | --- | --- |
    | 7k | 29.7 | 878k |
    | 15k, densifying to 75% | 30.9 | 1.43M |
    | **15k, capped at 5250 (default)** | **30.95** | **893k** |
    | 30k | 31.4 | 2.09M |

  * About 5.5 min on an RTX 4080 Laptop GPU.
* **Held-out views:** every 8th view is held out, and `stats.splat_holdout_psnr` reports PSNR on them.
* **Export:** splats with opacity ≤ 0.005 are dropped. `scene.splat` is the 32-byte-per-splat antimatter15 layout:
  * position float32×3 and linear scale float32×3;
  * RGBA uint8, from the SH DC term and opacity;
  * rotation uint8×4 (w, x, y, z);
  * sorted by descending size × opacity.
* **Frame:** the same as the GLBs (Y-up). Viewers must not apply a COLMAP flip.

On Windows, the MSVC environment is loaded via `vswhere`/`vcvars64` before
gsplat's one-time CUDA compile. Without torch or gsplat the location is built
without splats.

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
* **Collision.** The collider is invisible and double-sided, and is used for every ray cast: teleport aim, walls, ground and ceiling. An invisible floor plane at `floor_y`, covering the bounds plus 2 m, backs it up.
  * Each collider mesh gets a BVH (`three-mesh-bvh` 0.9.16). A ray against 100k triangles then costs about 0.03 ms instead of about 20 ms.
  * A hit's normal is turned to face back along the ray, so scan winding doesn't matter.
* **In VR (`local-floor`)** the user starts at the spawn point. Movement follows Google Earth VR:
  * **Glide (left thumbstick).** Move where the left controller points, flattened onto the floor plane. Push forward or back; tilt sideways to strafe.
    * Speed: 1.5 m/s at full deflection after a 0.15 dead zone. It scales with altitude when flying (×(1 + 0.5·altitude), capped at ×10).
    * Walls: rays at knee height (floor + 0.45 m) and chest height (eyes − 0.2 m) stop the user 0.25 m from a wall and slide them along it. In a corner they stop.
    * Ground: the floor eases (rate 12/s) onto the first surface below, found by a ray starting `STEP_HEIGHT` = 0.4 m above the current floor. So ledges up to 40 cm are climbed and any drop is followed. Ground following runs only while gliding (plus 0.4 s after) or while landing, so standing still never bobs.
  * **Right thumbstick.**
    * Left/right snap-turns 30° around the head.
    * Up/down flies at 1.2 m/s, up to 30 m above the ground. A ceiling ray stops the user 0.15 m below a ceiling.
    * A click lands: the user falls to the ground below.
  * **Teleport (trigger).** Hold the trigger to aim, release to teleport. Only surfaces whose facing normal has `y ≥ 0.6` are targets, so ceilings no longer qualify. The head moves horizontally onto the target, the floor height moves to the target's height, and flying ends.
  * **Comfort vignette.** A head-locked ring closes in while gliding or flying: clear inside about 26°, dark beyond about 49°, strength 0.9 × stick deflection. `&vignette=0` in the URL turns it off.
* **On desktop:** orbit controls starting at the spawn point at eye height. WASD or the arrow keys move horizontally and Q/E move down/up (free flight, no collision). Shift moves 3× faster.

* **Splats.** When `assets.splat` is set, the viewer loads Spark (`@sparkjsdev/spark` 2.3.1) and shows the splat by default, with the visual mesh hidden. **M** on desktop, or a grip squeeze in VR, toggles splat and mesh. The collider and teleport are unchanged.

three.js is loaded from a pinned version (0.181.2); Spark needs ≥ 0.180. three-mesh-bvh 0.9.16 needs three ≥ 0.159.
