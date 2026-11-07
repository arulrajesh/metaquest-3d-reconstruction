"""GPU texture bake from color keyframes (PLAN D16, after QuestRoomScan's texture refinement).

For every atlas texel: find its surface point and normal, then over all color
keyframes pick the best views by (head-on angle) x (working-distance weight,
peak ~1.4 m) x (relative image sharpness), reject views where the point is
occluded (ray cast against the mesh BVH), and blend the top 3 with a strong
power so the hero view dominates. Exposure differences between views are
equalized with per-view gains solved by least squares over texels seen by two
views. Unseen texels fall back to the vertex colors; charts are padded and
the result mildly sharpened.

Conventions: Open3D world frame; cameras in the Open3D convention with top-down
images; UV (0, 0) is the texture's top-left pixel (glTF).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import warp as wp

wp.config.quiet = True


@dataclass(frozen=True)
class BakeView:
    rgb: np.ndarray  # (H, W, 3) uint8, top-down
    intrinsic: np.ndarray  # (3, 3)
    world_to_camera: np.ndarray  # (4, 4)
    sharpness: float = 1.0
    camera: int = 0  # physical camera (0 left, 1 right): views of one camera share its calibration refinement


@dataclass(frozen=True)
class BakeSettings:
    size: int = 4096
    best_distance_m: float = 1.4
    distance_sigma_m: float = 1.2
    min_cos: float = 0.2
    edge_margin_px: float = 6.0
    blend_power: float = 4.0
    occlusion_offset_m: float = 0.01
    max_distance_m: float = 6.0
    exposure_equalize: bool = True
    sharpen: float = 0.35
    dilate_px: int = 8


@dataclass(frozen=True)
class BakeResult:
    texture: np.ndarray  # (size, size, 3) uint8 RGB, top-down
    used_texels: int
    textured_fraction: float  # of used texels, the share colored from camera views (rest: vertex-color fallback)
    view_gains: np.ndarray  # (V,) exposure gains applied


@wp.func
def _cross2(a: wp.vec2, b: wp.vec2) -> float:
    return a[0] * b[1] - a[1] * b[0]


@wp.kernel
def _raster(
    uvs: wp.array(dtype=wp.vec2),
    tris: wp.array(dtype=int),
    positions: wp.array(dtype=wp.vec3),
    colors: wp.array(dtype=wp.vec3),
    size: int,
    normal_sign: float,
    out_pos: wp.array2d(dtype=wp.vec3),
    out_nrm: wp.array2d(dtype=wp.vec3),
    out_fallback: wp.array2d(dtype=wp.vec3),
    out_valid: wp.array2d(dtype=int),
):
    f = wp.tid()
    i0 = tris[3 * f]
    i1 = tris[3 * f + 1]
    i2 = tris[3 * f + 2]
    s = float(size)
    a = uvs[i0] * s
    b = uvs[i1] * s
    c = uvs[i2] * s
    area = _cross2(b - a, c - a)
    if wp.abs(area) < 1.0e-9:
        return
    x0 = wp.max(int(wp.floor(wp.min(a[0], wp.min(b[0], c[0])))), 0)
    x1 = wp.min(int(wp.ceil(wp.max(a[0], wp.max(b[0], c[0])))), size - 1)
    y0 = wp.max(int(wp.floor(wp.min(a[1], wp.min(b[1], c[1])))), 0)
    y1 = wp.min(int(wp.ceil(wp.max(a[1], wp.max(b[1], c[1])))), size - 1)
    p0 = positions[i0]
    p1 = positions[i1]
    p2 = positions[i2]
    normal = wp.normalize(wp.cross(p1 - p0, p2 - p0)) * normal_sign
    eps = 0.02
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            p = wp.vec2(float(x) + 0.5, float(y) + 0.5)
            w0 = _cross2(b - p, c - p) / area
            w1 = _cross2(c - p, a - p) / area
            w2 = 1.0 - w0 - w1
            if w0 >= -eps and w1 >= -eps and w2 >= -eps:
                out_pos[y, x] = p0 * w0 + p1 * w1 + p2 * w2
                out_nrm[y, x] = normal
                out_fallback[y, x] = colors[i0] * w0 + colors[i1] * w1 + colors[i2] * w2
                out_valid[y, x] = 1


@wp.func
def _bilinear(img: wp.array4d(dtype=wp.uint8), v: int, u: float, w: float) -> wp.vec3:
    h = img.shape[1]
    wd = img.shape[2]
    fx = wp.clamp(u - 0.5, 0.0, float(wd - 1))
    fy = wp.clamp(w - 0.5, 0.0, float(h - 1))
    x0 = int(wp.floor(fx))
    y0 = int(wp.floor(fy))
    x1 = wp.min(x0 + 1, wd - 1)
    y1 = wp.min(y0 + 1, h - 1)
    tx = fx - float(x0)
    ty = fy - float(y0)
    c00 = wp.vec3(float(img[v, y0, x0, 0]), float(img[v, y0, x0, 1]), float(img[v, y0, x0, 2]))
    c10 = wp.vec3(float(img[v, y0, x1, 0]), float(img[v, y0, x1, 1]), float(img[v, y0, x1, 2]))
    c01 = wp.vec3(float(img[v, y1, x0, 0]), float(img[v, y1, x0, 1]), float(img[v, y1, x0, 2]))
    c11 = wp.vec3(float(img[v, y1, x1, 0]), float(img[v, y1, x1, 1]), float(img[v, y1, x1, 2]))
    top = c00 * (1.0 - tx) + c10 * tx
    bottom = c01 * (1.0 - tx) + c11 * tx
    return (top * (1.0 - ty) + bottom * ty) / 255.0


@wp.kernel
def _bake(
    pos: wp.array2d(dtype=wp.vec3),
    nrm: wp.array2d(dtype=wp.vec3),
    valid: wp.array2d(dtype=int),
    images: wp.array4d(dtype=wp.uint8),
    view_k: wp.array(dtype=wp.vec4),
    view_wc: wp.array(dtype=wp.mat44),
    view_center: wp.array(dtype=wp.vec3),
    view_weight: wp.array(dtype=float),
    view_gain: wp.array(dtype=float),
    mesh: wp.uint64,
    params: wp.vec4,  # best distance, distance sigma, min cos, edge margin px
    params2: wp.vec4,  # blend power, occlusion offset, max distance, unused
    out_color: wp.array2d(dtype=wp.vec3),
    out_count: wp.array2d(dtype=int),
    out_v0: wp.array2d(dtype=int),
    out_v1: wp.array2d(dtype=int),
    out_l0: wp.array2d(dtype=float),
    out_l1: wp.array2d(dtype=float),
):
    y, x = wp.tid()
    if valid[y, x] == 0:
        return
    p = pos[y, x]
    n = nrm[y, x]
    h = float(images.shape[1])
    wd = float(images.shape[2])
    s0 = float(0.0)
    s1 = float(0.0)
    s2 = float(0.0)
    c0 = wp.vec3(0.0)
    c1 = wp.vec3(0.0)
    c2 = wp.vec3(0.0)
    v0 = int(-1)
    v1 = int(-1)
    count = int(0)
    for v in range(view_k.shape[0]):
        q = wp.transform_point(view_wc[v], p)
        if q[2] <= 0.05:
            continue
        k = view_k[v]
        u = k[0] * q[0] / q[2] + k[2]
        w = k[1] * q[1] / q[2] + k[3]
        if u < params[3] or w < params[3] or u > wd - params[3] or w > h - params[3]:
            continue
        to_cam = view_center[v] - p
        d = wp.length(to_cam)
        if d > params2[2] or d < 0.05:
            continue
        direction = to_cam / d
        cosang = wp.dot(n, direction)
        if cosang < params[2]:
            continue
        dd = (d - params[0]) / params[1]
        score = cosang * wp.exp(-dd * dd) * view_weight[v]
        if score <= s2:
            continue
        start = p + n * params2[1] + direction * params2[1]
        hit = wp.mesh_query_ray(mesh, start, direction, d - 2.0 * params2[1])
        if hit.result:
            continue
        color = _bilinear(images, v, u, w) * view_gain[v]
        count += 1
        if score > s0:
            s2 = s1
            c2 = c1
            s1 = s0
            c1 = c0
            v1 = v0
            s0 = score
            c0 = color
            v0 = v
        elif score > s1:
            s2 = s1
            c2 = c1
            s1 = score
            c1 = color
            v1 = v
        else:
            s2 = score
            c2 = color
    out_count[y, x] = count
    if v0 < 0:
        return
    w0 = wp.pow(s0, params2[0])
    w1 = wp.pow(s1, params2[0])
    w2 = wp.pow(s2, params2[0])
    out_color[y, x] = (c0 * w0 + c1 * w1 + c2 * w2) / (w0 + w1 + w2)
    out_v0[y, x] = v0
    out_v1[y, x] = v1
    out_l0[y, x] = wp.dot(c0, wp.vec3(0.299, 0.587, 0.114))
    out_l1[y, x] = wp.dot(c1, wp.vec3(0.299, 0.587, 0.114))


def _normal_sign(vertices: np.ndarray, triangles: np.ndarray, views: list[BakeView]) -> float:
    """Orient face normals toward the cameras (marching-cubes winding is consistent but its sign is not fixed)."""
    rng = np.random.default_rng(0)
    faces = triangles[rng.choice(len(triangles), size=min(4000, len(triangles)), replace=False)]
    p0, p1, p2 = (vertices[faces[:, i]].astype(np.float64) for i in range(3))
    normals = np.cross(p1 - p0, p2 - p0)
    centers = (p0 + p1 + p2) / 3.0
    cams = np.array([np.linalg.inv(v.world_to_camera)[:3, 3] for v in views])
    nearest = cams[np.argmin(((centers[:, None, :] - cams[None, :, :]) ** 2).sum(-1), axis=1)]
    return 1.0 if float(np.sum(np.einsum("ij,ij->i", normals, nearest - centers) > 0)) >= len(faces) / 2 else -1.0


def _solve_gains(v0: np.ndarray, v1: np.ndarray, l0: np.ndarray, l1: np.ndarray, n_views: int) -> np.ndarray:
    """Per-view gains g minimizing sum (log g_a + log l_a - log g_b - log l_b)^2, regularized toward 1."""
    from scipy.sparse import coo_matrix, vstack
    from scipy.sparse.linalg import lsqr

    v0, v1, l0, l1 = (np.ravel(a) for a in (v0, v1, l0, l1))
    ok = (v0 >= 0) & (v1 >= 0) & (l0 > 0.03) & (l1 > 0.03)
    if ok.sum() < 100:
        return np.ones(n_views)
    idx = np.flatnonzero(ok)
    if len(idx) > 400_000:
        idx = np.random.default_rng(0).choice(idx, 400_000, replace=False)
    m = len(idx)
    rows = np.repeat(np.arange(m), 2)
    cols = np.stack([v0[idx], v1[idx]], axis=1).reshape(-1)
    vals = np.tile([1.0, -1.0], m)
    a = coo_matrix((vals, (rows, cols)), shape=(m, n_views))
    b = np.log(l1[idx]) - np.log(l0[idx])
    reg = coo_matrix((np.full(n_views, 0.3 * np.sqrt(m / n_views)), (np.arange(n_views), np.arange(n_views))), shape=(n_views, n_views))
    solution = lsqr(vstack([a, reg]).tocsr(), np.concatenate([b, np.zeros(n_views)]))[0]
    return np.clip(np.exp(solution), 0.6, 1.6)


def _dilate(texture: np.ndarray, mask: np.ndarray, iterations: int) -> np.ndarray:
    """Pad charts outward so bilinear filtering and mipmaps never sample empty texels."""
    tex, filled = texture.copy(), mask.copy()
    kernel = np.ones((3, 3), np.float32)
    for _ in range(iterations):
        weight = cv2.filter2D(filled.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        grow = (~filled) & (weight > 0)
        if not grow.any():
            break
        summed = cv2.filter2D(tex * filled[..., None], -1, kernel, borderType=cv2.BORDER_CONSTANT)
        tex[grow] = summed[grow] / weight[grow, None]
        filled |= grow
    return tex


def bake_texture(
    vertices: np.ndarray,
    triangles: np.ndarray,
    uvs: np.ndarray,
    vertex_colors: np.ndarray,
    views: list[BakeView],
    settings: BakeSettings | None = None,
    device: str = "cuda:0",
) -> BakeResult:
    settings = settings or BakeSettings()
    wp.init()
    if device.startswith("cuda") and not wp.is_cuda_available():
        device = "cpu"
    size = settings.size
    vertices = np.ascontiguousarray(vertices, np.float32)
    triangles = np.ascontiguousarray(triangles, np.int32)
    sign = _normal_sign(vertices, triangles, views)

    pos = wp.zeros((size, size), dtype=wp.vec3, device=device)
    nrm = wp.zeros((size, size), dtype=wp.vec3, device=device)
    fallback = wp.zeros((size, size), dtype=wp.vec3, device=device)
    valid = wp.zeros((size, size), dtype=int, device=device)
    wp.launch(_raster, dim=len(triangles), device=device, inputs=[
        wp.array(np.ascontiguousarray(uvs, np.float32), dtype=wp.vec2, device=device),
        wp.array(triangles.reshape(-1), dtype=int, device=device),
        wp.array(vertices, dtype=wp.vec3, device=device),
        wp.array(np.ascontiguousarray(vertex_colors, np.float32), dtype=wp.vec3, device=device),
        size, sign, pos, nrm, fallback, valid,
    ])

    images = wp.array(np.stack([v.rgb[:, :, :3] for v in views]).astype(np.uint8), dtype=wp.uint8, device=device)
    k = np.array([[v.intrinsic[0, 0], v.intrinsic[1, 1], v.intrinsic[0, 2], v.intrinsic[1, 2]] for v in views], np.float32)
    wc = np.stack([v.world_to_camera for v in views]).astype(np.float32)
    centers = np.array([np.linalg.inv(v.world_to_camera)[:3, 3] for v in views], np.float32)
    sharp = np.array([v.sharpness for v in views], np.float64)
    weights = np.clip(sharp / max(np.median(sharp), 1e-9), 0.3, 1.5) / 1.5
    mesh = wp.Mesh(points=wp.array(vertices, dtype=wp.vec3, device=device), indices=wp.array(triangles.reshape(-1), dtype=int, device=device))

    outputs = [wp.zeros((size, size), dtype=wp.vec3, device=device), wp.zeros((size, size), dtype=int, device=device)]
    outputs += [wp.full((size, size), -1, dtype=int, device=device) for _ in range(2)]
    outputs += [wp.zeros((size, size), dtype=float, device=device) for _ in range(2)]

    def run(gains: np.ndarray) -> None:
        wp.launch(_bake, dim=(size, size), device=device, inputs=[
            pos, nrm, valid, images,
            wp.array(k, dtype=wp.vec4, device=device),
            wp.array(wc, dtype=wp.mat44, device=device),
            wp.array(centers, dtype=wp.vec3, device=device),
            wp.array(weights.astype(np.float32), dtype=float, device=device),
            wp.array(gains.astype(np.float32), dtype=float, device=device),
            mesh.id,
            wp.vec4(settings.best_distance_m, settings.distance_sigma_m, settings.min_cos, settings.edge_margin_px),
            wp.vec4(settings.blend_power, settings.occlusion_offset_m, settings.max_distance_m, 0.0),
            *outputs,
        ])

    gains = np.ones(len(views))
    run(gains)
    if settings.exposure_equalize and len(views) > 1:
        gains = _solve_gains(outputs[2].numpy(), outputs[3].numpy(), outputs[4].numpy(), outputs[5].numpy(), len(views))
        run(gains)

    used = valid.numpy().astype(bool)
    textured = used & (outputs[1].numpy() > 0)
    color = np.where(textured[..., None], outputs[0].numpy(), fallback.numpy())
    if settings.sharpen > 0:
        blurred = cv2.GaussianBlur(color, (0, 0), 1.2)
        color = np.where(textured[..., None], color + settings.sharpen * (color - blurred), color)
    color = _dilate(np.clip(color, 0.0, 1.0).astype(np.float32), used, settings.dilate_px)
    texture = np.clip(np.round(color * 255.0), 0, 255).astype(np.uint8)
    return BakeResult(
        texture=texture,
        used_texels=int(used.sum()),
        textured_fraction=float(textured.sum() / max(used.sum(), 1)),
        view_gains=gains,
    )
