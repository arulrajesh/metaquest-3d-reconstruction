"""Dense colored Kalman TSDF on the GPU (NVIDIA Warp, CPU fallback), for live preview and builds.

Each voxel holds a 1-D Kalman filter over its truncated signed distance
(normalized to [-1, 1] by the truncation distance):

* state mean ``tsdf`` and information ``info`` (inverse variance; 0 = unobserved);
* measurement variance from a Quest depth noise model ``sigma(z) = sigma0 + k * z^2``,
  inflated at depth discontinuities (flying pixels);
* small process noise so estimates can adapt (moved furniture);
* an innovation gate on confident voxels: near-surface readings further than
  ``gate_sigma`` standard deviations from the estimate are rejected as outliers;
  a running outlier rate is kept and voxels that disagree persistently are re-initialized.

Points "worth keeping" are those whose posterior sigma is small and whose
outlier rate is low; only those reach the extracted mesh. ``integrate`` returns
the frame's information gain (total variance reduction near surfaces), which
drives keyframe selection.

World coordinates are MQ3DRecon's Open3D world frame (Y-up, right-handed);
cameras use the Open3D convention (X right, Y down, Z forward). Voxel
``(i, j, k)`` is centered at ``origin + (i, j, k) * voxel_size``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

wp.config.quiet = True

UNCOLORED = wp.constant(0.6)
UNCOLORED_GRAY = 0.6  # must match UNCOLORED


@dataclass(frozen=True)
class NoiseModel:
    sigma0_m: float = 0.004
    k_per_m: float = 0.003
    edge_ratio: float = 0.05
    edge_inflation: float = 10.0
    process_noise: float = 1.0e-5
    gate_sigma: float = 3.0
    confident_sigma_m: float = 0.02
    reinit_outlier_rate: float = 0.6


@dataclass(frozen=True)
class VolumeSpec:
    origin: tuple[float, float, float]
    dims: tuple[int, int, int]
    voxel_size: float
    trunc_m: float
    max_info: float = 1.0e5

    @property
    def bytes_required(self) -> int:
        return int(np.prod(self.dims)) * 28

    @classmethod
    def around(cls, center, extent_m, voxel_size: float, trunc_m: float, max_info: float = 1.0e5) -> "VolumeSpec":
        extent = np.asarray(extent_m, dtype=np.float64)
        dims = tuple(int(np.ceil(e / voxel_size)) + 1 for e in extent)
        origin = tuple(float(v) for v in np.asarray(center, dtype=np.float64) - extent / 2.0)
        return cls(origin=origin, dims=dims, voxel_size=voxel_size, trunc_m=trunc_m, max_info=max_info)  # type: ignore[arg-type]


@dataclass(frozen=True)
class MeshArrays:
    vertices: np.ndarray  # (N, 3) float32 world
    triangles: np.ndarray  # (M, 3) int32
    colors: np.ndarray  # (N, 3) float32 in [0, 1]; UNCOLORED_GRAY where no color was observed
    color_valid: np.ndarray  # (N,) bool
    sigma_m: np.ndarray | None = None  # (N,) float32 posterior std-dev in meters


@dataclass(frozen=True)
class FrameScore:
    info_gain: float  # summed variance reduction near surfaces (normalized units, per voxel)
    surface_voxels: int  # near-surface voxels this frame updated
    new_voxels: int  # near-surface voxels observed for the first time
    rejected: int  # readings rejected by the innovation gate

    @property
    def novelty(self) -> float:
        """Mean variance reduction per updated surface voxel, in [0, 1]: high when the view shows new surface."""
        return self.info_gain / self.surface_voxels if self.surface_voxels else 0.0


@wp.kernel
def _integrate_kernel(
    tsdf: wp.array3d(dtype=float),
    info: wp.array3d(dtype=float),
    outlier: wp.array3d(dtype=float),
    color: wp.array3d(dtype=wp.vec3),
    color_info: wp.array3d(dtype=float),
    depth: wp.array2d(dtype=float),
    depth_k: wp.vec4,
    depth_wc: wp.mat44,
    rgb: wp.array3d(dtype=wp.uint8),
    color_k: wp.vec4,
    color_wc: wp.mat44,
    has_color: int,
    origin: wp.vec3,
    voxel: float,
    trunc: float,
    max_info: float,
    noise: wp.vec4,  # sigma0, k, edge_ratio, edge_inflation
    robust: wp.vec4,  # process_noise, gate_sigma^2, confident_info, reinit_outlier_rate
    stats: wp.array(dtype=float),  # [info_gain, surface_voxels, new_voxels, rejected]
):
    i, j, k = wp.tid()
    p = origin + wp.vec3(float(i), float(j), float(k)) * voxel
    pc = wp.transform_point(depth_wc, p)
    if pc[2] <= 0.01:
        return
    u = int(wp.round(depth_k[0] * pc[0] / pc[2] + depth_k[2]))
    v = int(wp.round(depth_k[1] * pc[1] / pc[2] + depth_k[3]))
    w_img = depth.shape[1]
    h_img = depth.shape[0]
    if u < 0 or v < 0 or u >= w_img or v >= h_img:
        return
    d = depth[v, u]
    if d <= 0.0:
        return
    sdf = d - pc[2]
    if sdf < -trunc:
        return

    # Measurement noise (meters), inflated at depth discontinuities (flying pixels).
    sigma_m = noise[0] + noise[1] * d * d
    dl = depth[v, wp.max(u - 1, 0)]
    dr = depth[v, wp.min(u + 1, w_img - 1)]
    du = depth[wp.max(v - 1, 0), u]
    dd = depth[wp.min(v + 1, h_img - 1), u]
    grad = wp.max(wp.abs(dr - dl), wp.abs(dd - du)) * 0.5
    if grad > noise[2] * d or dl <= 0.0 or dr <= 0.0 or du <= 0.0 or dd <= 0.0:
        sigma_m = sigma_m * noise[3]
    r = (sigma_m / trunc) * (sigma_m / trunc)

    obs = wp.min(1.0, sdf / trunc)
    near_surface = obs < 1.0
    x = tsdf[i, j, k]
    w = info[i, j, k]
    if w > 0.0:
        w = 1.0 / (1.0 / w + robust[0])  # process noise

    rate = outlier[i, j, k]
    if near_surface and w >= robust[2]:
        nu = obs - x
        s = 1.0 / w + r
        if nu * nu > robust[1] * s:
            rate = 0.9 * rate + 0.1
            if rate < robust[3]:
                outlier[i, j, k] = rate
                wp.atomic_add(stats, 3, 1.0)
                return
            # Persistently inconsistent: the scene changed here; restart this voxel's filter.
            w = 0.0
            rate = 0.0
        else:
            rate = 0.9 * rate
    outlier[i, j, k] = rate

    prior_var = 1.0
    if w > 0.0:
        prior_var = 1.0 / w
    new_w = wp.min(w + 1.0 / r, max_info)
    tsdf[i, j, k] = (x * w + obs / r) / (w + 1.0 / r)
    info[i, j, k] = new_w
    if near_surface:
        wp.atomic_add(stats, 0, prior_var - 1.0 / new_w)
        wp.atomic_add(stats, 1, 1.0)
        if w <= 0.0:
            wp.atomic_add(stats, 2, 1.0)

    if has_color == 0 or sdf > trunc:
        return
    qc = wp.transform_point(color_wc, p)
    if qc[2] <= 0.01:
        return
    cu = int(wp.round(color_k[0] * qc[0] / qc[2] + color_k[2]))
    cv = int(wp.round(color_k[1] * qc[1] / qc[2] + color_k[3]))
    if cu < 0 or cv < 0 or cu >= rgb.shape[1] or cv >= rgb.shape[0]:
        return
    c = wp.vec3(float(rgb[cv, cu, 0]), float(rgb[cv, cu, 1]), float(rgb[cv, cu, 2])) / 255.0
    cw = color_info[i, j, k]
    wc = 1.0 / r  # closer, cleaner readings weigh more
    color[i, j, k] = (color[i, j, k] * cw + c * wc) / (cw + wc)
    color_info[i, j, k] = wp.min(cw + wc, max_info)


@wp.kernel
def _vertex_kernel(
    verts: wp.array(dtype=wp.vec3),
    info: wp.array3d(dtype=float),
    outlier: wp.array3d(dtype=float),
    color: wp.array3d(dtype=wp.vec3),
    color_info: wp.array3d(dtype=float),
    min_info: float,
    max_outlier_rate: float,
    trunc: float,
    valid: wp.array(dtype=int),
    out_color: wp.array(dtype=wp.vec3),
    out_color_ok: wp.array(dtype=int),
    out_sigma: wp.array(dtype=float),
):
    # A marching-cubes vertex lies on one grid edge (two integer axes, one fractional).
    # Keep it only if both edge endpoints are confident and consistent: this drops
    # surfaces against never-observed space, uncertain surfaces, and outlier-ridden ones.
    t = wp.tid()
    g = verts[t]
    eps = 1.0e-3
    ai = wp.clamp(int(wp.floor(g[0] + eps)), 0, info.shape[0] - 1)
    aj = wp.clamp(int(wp.floor(g[1] + eps)), 0, info.shape[1] - 1)
    ak = wp.clamp(int(wp.floor(g[2] + eps)), 0, info.shape[2] - 1)
    fi = g[0] - float(ai)
    fj = g[1] - float(aj)
    fk = g[2] - float(ak)
    bi = wp.min(ai + wp.where(fi > eps, 1, 0), info.shape[0] - 1)
    bj = wp.min(aj + wp.where(fj > eps, 1, 0), info.shape[1] - 1)
    bk = wp.min(ak + wp.where(fk > eps, 1, 0), info.shape[2] - 1)
    w = wp.min(info[ai, aj, ak], info[bi, bj, bk])
    rate = wp.max(outlier[ai, aj, ak], outlier[bi, bj, bk])
    ok = 0
    if w >= min_info and rate <= max_outlier_rate:
        ok = 1
    valid[t] = ok
    out_sigma[t] = wp.where(w > 0.0, trunc / wp.sqrt(w), 1.0e9)
    f = wp.clamp(wp.max(fi, wp.max(fj, fk)), 0.0, 1.0)
    ca = color_info[ai, aj, ak]
    cb = color_info[bi, bj, bk]
    if ca > 0.0 and cb > 0.0:
        out_color[t] = color[ai, aj, ak] * (1.0 - f) + color[bi, bj, bk] * f
        out_color_ok[t] = 1
    elif ca > 0.0 or cb > 0.0:
        out_color[t] = wp.where(ca > 0.0, color[ai, aj, ak], color[bi, bj, bk])
        out_color_ok[t] = 1
    else:
        out_color[t] = wp.vec3(UNCOLORED, UNCOLORED, UNCOLORED)
        out_color_ok[t] = 0


@wp.kernel
def _count_observed(info: wp.array3d(dtype=float), counter: wp.array(dtype=int)):
    i, j, k = wp.tid()
    if info[i, j, k] > 0.0:
        wp.atomic_add(counter, 0, 1)


def _mat44(m: np.ndarray) -> wp.mat44:
    return wp.mat44(*np.asarray(m, dtype=np.float32).reshape(16))


def _k4(k: np.ndarray) -> wp.vec4:
    return wp.vec4(float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]))


class GpuTsdfVolume:
    def __init__(self, spec: VolumeSpec, device: str = "cuda:0", noise: NoiseModel | None = None):
        wp.init()
        if device.startswith("cuda") and not wp.is_cuda_available():
            device = "cpu"
        self.spec = spec
        self.device = device
        self.noise = noise or NoiseModel()
        self.tsdf = wp.full(spec.dims, 1.0, dtype=float, device=device)
        self.info = wp.zeros(spec.dims, dtype=float, device=device)
        self.outlier = wp.zeros(spec.dims, dtype=float, device=device)
        self.color = wp.zeros(spec.dims, dtype=wp.vec3, device=device)
        self.color_info = wp.zeros(spec.dims, dtype=float, device=device)
        self._stats = wp.zeros(4, dtype=float, device=device)
        self._empty_rgb = wp.zeros((1, 1, 3), dtype=wp.uint8, device=device)
        self.frames_integrated = 0

    def sigma_to_info(self, sigma_m: float) -> float:
        return float((self.spec.trunc_m / max(sigma_m, 1e-9)) ** 2)

    def integrate(
        self,
        depth_m: np.ndarray,
        depth_intrinsic: np.ndarray,
        depth_extrinsic_wc: np.ndarray,
        rgb: np.ndarray | None = None,
        color_intrinsic: np.ndarray | None = None,
        color_extrinsic_wc: np.ndarray | None = None,
    ) -> FrameScore:
        depth = wp.array(np.ascontiguousarray(depth_m, dtype=np.float32), dtype=float, device=self.device)
        has_color = rgb is not None and color_intrinsic is not None and color_extrinsic_wc is not None
        rgb_array = (
            wp.array(np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8), dtype=wp.uint8, device=self.device)
            if has_color
            else self._empty_rgb
        )
        n = self.noise
        self._stats.zero_()
        wp.launch(
            _integrate_kernel,
            dim=self.spec.dims,
            inputs=[
                self.tsdf, self.info, self.outlier, self.color, self.color_info,
                depth, _k4(depth_intrinsic), _mat44(depth_extrinsic_wc),
                rgb_array, _k4(color_intrinsic if has_color else np.eye(3)),
                _mat44(color_extrinsic_wc if has_color else np.eye(4)), int(has_color),
                wp.vec3(*self.spec.origin), float(self.spec.voxel_size), float(self.spec.trunc_m), float(self.spec.max_info),
                wp.vec4(n.sigma0_m, n.k_per_m, n.edge_ratio, n.edge_inflation),
                wp.vec4(n.process_noise, n.gate_sigma ** 2, self.sigma_to_info(n.confident_sigma_m), n.reinit_outlier_rate),
                self._stats,
            ],
            device=self.device,
        )
        self.frames_integrated += 1
        gain, surface, new, rejected = (float(v) for v in self._stats.numpy())
        return FrameScore(info_gain=gain, surface_voxels=int(surface), new_voxels=int(new), rejected=int(rejected))

    def observed_voxel_count(self) -> int:
        counter = wp.zeros(1, dtype=int, device=self.device)
        wp.launch(_count_observed, dim=self.spec.dims, inputs=[self.info, counter], device=self.device)
        return int(counter.numpy()[0])

    def extract_mesh(self, max_sigma_m: float | None = None, max_outlier_rate: float = 0.3) -> MeshArrays:
        """Mesh of the points worth keeping: posterior sigma <= ``max_sigma_m`` (default half a voxel)."""
        import warp.geometry as wg

        max_sigma = 0.5 * self.spec.voxel_size if max_sigma_m is None else max_sigma_m
        mc = wg.IsoSurfaceMarchingCubes(*self.spec.dims)
        mc.surface(self.tsdf, 0.0)
        n_verts = len(mc.verts)
        if n_verts == 0:
            empty = np.zeros((0, 3), np.float32)
            return MeshArrays(empty, np.zeros((0, 3), np.int32), empty, np.zeros(0, bool), np.zeros(0, np.float32))
        valid = wp.zeros(n_verts, dtype=int, device=self.device)
        colors = wp.zeros(n_verts, dtype=wp.vec3, device=self.device)
        color_ok = wp.zeros(n_verts, dtype=int, device=self.device)
        sigma = wp.zeros(n_verts, dtype=float, device=self.device)
        wp.launch(
            _vertex_kernel,
            dim=n_verts,
            inputs=[
                mc.verts, self.info, self.outlier, self.color, self.color_info,
                self.sigma_to_info(max_sigma), float(max_outlier_rate), float(self.spec.trunc_m),
                valid, colors, color_ok, sigma,
            ],
            device=self.device,
        )
        grid_verts = mc.verts.numpy()
        triangles = mc.indices.numpy().reshape(-1, 3)
        keep = valid.numpy().astype(bool)[triangles].all(axis=1)
        triangles = triangles[keep]
        used = np.unique(triangles)
        remap = np.full(n_verts, -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        vertices = np.asarray(self.spec.origin, np.float32) + grid_verts[used] * np.float32(self.spec.voxel_size)
        return MeshArrays(
            vertices=vertices.astype(np.float32),
            triangles=remap[triangles].astype(np.int32),
            colors=np.clip(colors.numpy()[used], 0.0, 1.0).astype(np.float32),
            color_valid=color_ok.numpy()[used].astype(bool),
            sigma_m=sigma.numpy()[used].astype(np.float32),
        )
