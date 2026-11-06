"""Dense colored TSDF volume on the GPU (NVIDIA Warp), used for live preview and builds.

World coordinates are MQ3DRecon's Open3D world frame (Y-up, right-handed).
Cameras follow the Open3D convention (X right, Y down, Z forward); intrinsics
must already use the mirrored principal point (see ``native_depth``).

Voxel ``(i, j, k)`` has its center at ``origin + (i, j, k) * voxel_size``.
Unobserved voxels have weight 0 and are excluded from extracted meshes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

wp.config.quiet = True

UNCOLORED = wp.constant(0.6)


@dataclass(frozen=True)
class VolumeSpec:
    origin: tuple[float, float, float]
    dims: tuple[int, int, int]
    voxel_size: float
    trunc_m: float
    max_weight: float = 64.0

    @property
    def bytes_required(self) -> int:
        return int(np.prod(self.dims)) * 24

    @classmethod
    def around(cls, center, extent_m, voxel_size: float, trunc_m: float, max_weight: float = 64.0) -> "VolumeSpec":
        extent = np.asarray(extent_m, dtype=np.float64)
        dims = tuple(int(np.ceil(e / voxel_size)) + 1 for e in extent)
        origin = tuple(float(v) for v in np.asarray(center, dtype=np.float64) - extent / 2.0)
        return cls(origin=origin, dims=dims, voxel_size=voxel_size, trunc_m=trunc_m, max_weight=max_weight)  # type: ignore[arg-type]


UNCOLORED_GRAY = 0.6  # must match UNCOLORED


@dataclass(frozen=True)
class MeshArrays:
    vertices: np.ndarray  # (N, 3) float32 world
    triangles: np.ndarray  # (M, 3) int32
    colors: np.ndarray  # (N, 3) float32 in [0, 1]; UNCOLORED_GRAY where no color was observed
    color_valid: np.ndarray  # (N,) bool: True where a color camera observed the vertex


@wp.kernel
def _integrate_kernel(
    tsdf: wp.array3d(dtype=float),
    weight: wp.array3d(dtype=float),
    color: wp.array3d(dtype=wp.vec3),
    color_weight: wp.array3d(dtype=float),
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
    max_weight: float,
):
    i, j, k = wp.tid()
    p = origin + wp.vec3(float(i), float(j), float(k)) * voxel
    pc = wp.transform_point(depth_wc, p)
    if pc[2] <= 0.01:
        return
    u = int(wp.round(depth_k[0] * pc[0] / pc[2] + depth_k[2]))
    v = int(wp.round(depth_k[1] * pc[1] / pc[2] + depth_k[3]))
    if u < 0 or v < 0 or u >= depth.shape[1] or v >= depth.shape[0]:
        return
    d = depth[v, u]
    if d <= 0.0:
        return
    sdf = d - pc[2]
    if sdf < -trunc:
        return
    obs = wp.min(1.0, sdf / trunc)
    w = weight[i, j, k]
    tsdf[i, j, k] = (tsdf[i, j, k] * w + obs) / (w + 1.0)
    weight[i, j, k] = wp.min(w + 1.0, max_weight)

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
    cw = color_weight[i, j, k]
    color[i, j, k] = (color[i, j, k] * cw + c) / (cw + 1.0)
    color_weight[i, j, k] = wp.min(cw + 1.0, max_weight)


@wp.kernel
def _vertex_kernel(
    verts: wp.array(dtype=wp.vec3),
    weight: wp.array3d(dtype=float),
    color: wp.array3d(dtype=wp.vec3),
    color_weight: wp.array3d(dtype=float),
    min_weight: float,
    valid: wp.array(dtype=int),
    out_color: wp.array(dtype=wp.vec3),
    out_color_ok: wp.array(dtype=int),
):
    # A marching-cubes vertex lies on one grid edge: integer coordinates on two
    # axes, fractional on the third. Keep it only if both edge endpoints were
    # observed; this removes the spurious surface where observed voxels behind
    # a surface meet never-observed voxels.
    t = wp.tid()
    g = verts[t]
    eps = 1.0e-3
    ai = wp.clamp(int(wp.floor(g[0] + eps)), 0, weight.shape[0] - 1)
    aj = wp.clamp(int(wp.floor(g[1] + eps)), 0, weight.shape[1] - 1)
    ak = wp.clamp(int(wp.floor(g[2] + eps)), 0, weight.shape[2] - 1)
    fi = g[0] - float(ai)
    fj = g[1] - float(aj)
    fk = g[2] - float(ak)
    bi = wp.min(ai + wp.where(fi > eps, 1, 0), weight.shape[0] - 1)
    bj = wp.min(aj + wp.where(fj > eps, 1, 0), weight.shape[1] - 1)
    bk = wp.min(ak + wp.where(fk > eps, 1, 0), weight.shape[2] - 1)
    w = wp.min(weight[ai, aj, ak], weight[bi, bj, bk])
    valid[t] = wp.where(w >= min_weight, 1, 0)
    f = wp.clamp(wp.max(fi, wp.max(fj, fk)), 0.0, 1.0)
    ca = color_weight[ai, aj, ak]
    cb = color_weight[bi, bj, bk]
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
def _count_observed(weight: wp.array3d(dtype=float), counter: wp.array(dtype=int)):
    i, j, k = wp.tid()
    if weight[i, j, k] > 0.0:
        wp.atomic_add(counter, 0, 1)


def _mat44(m: np.ndarray) -> wp.mat44:
    return wp.mat44(*np.asarray(m, dtype=np.float32).reshape(16))


def _k4(k: np.ndarray) -> wp.vec4:
    return wp.vec4(float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]))


class GpuTsdfVolume:
    def __init__(self, spec: VolumeSpec, device: str = "cuda:0"):
        wp.init()
        if device.startswith("cuda") and not wp.is_cuda_available():
            device = "cpu"
        self.spec = spec
        self.device = device
        self.tsdf = wp.full(spec.dims, 1.0, dtype=float, device=device)
        self.weight = wp.zeros(spec.dims, dtype=float, device=device)
        self.color = wp.zeros(spec.dims, dtype=wp.vec3, device=device)
        self.color_weight = wp.zeros(spec.dims, dtype=float, device=device)
        self._empty_rgb = wp.zeros((1, 1, 3), dtype=wp.uint8, device=device)
        self.frames_integrated = 0

    def integrate(
        self,
        depth_m: np.ndarray,
        depth_intrinsic: np.ndarray,
        depth_extrinsic_wc: np.ndarray,
        rgb: np.ndarray | None = None,
        color_intrinsic: np.ndarray | None = None,
        color_extrinsic_wc: np.ndarray | None = None,
    ) -> None:
        depth = wp.array(np.ascontiguousarray(depth_m, dtype=np.float32), dtype=float, device=self.device)
        has_color = rgb is not None and color_intrinsic is not None and color_extrinsic_wc is not None
        rgb_array = (
            wp.array(np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8), dtype=wp.uint8, device=self.device)
            if has_color
            else self._empty_rgb
        )
        identity = np.eye(4)
        wp.launch(
            _integrate_kernel,
            dim=self.spec.dims,
            inputs=[
                self.tsdf, self.weight, self.color, self.color_weight,
                depth, _k4(depth_intrinsic), _mat44(depth_extrinsic_wc),
                rgb_array, _k4(color_intrinsic if has_color else np.eye(3)),
                _mat44(color_extrinsic_wc if has_color else identity), int(has_color),
                wp.vec3(*self.spec.origin), float(self.spec.voxel_size), float(self.spec.trunc_m),
                float(self.spec.max_weight),
            ],
            device=self.device,
        )
        self.frames_integrated += 1

    def observed_voxel_count(self) -> int:
        counter = wp.zeros(1, dtype=int, device=self.device)
        wp.launch(_count_observed, dim=self.spec.dims, inputs=[self.weight, counter], device=self.device)
        return int(counter.numpy()[0])

    def extract_mesh(self, min_weight: float = 1.0) -> MeshArrays:
        import warp.geometry as wg

        mc = wg.IsoSurfaceMarchingCubes(*self.spec.dims)
        mc.surface(self.tsdf, 0.0)
        n_verts = len(mc.verts)
        if n_verts == 0:
            empty = np.zeros((0, 3), np.float32)
            return MeshArrays(empty, np.zeros((0, 3), np.int32), empty, np.zeros(0, bool))
        valid = wp.zeros(n_verts, dtype=int, device=self.device)
        colors = wp.zeros(n_verts, dtype=wp.vec3, device=self.device)
        color_ok = wp.zeros(n_verts, dtype=int, device=self.device)
        wp.launch(
            _vertex_kernel,
            dim=n_verts,
            inputs=[mc.verts, self.weight, self.color, self.color_weight, float(min_weight), valid, colors, color_ok],
            device=self.device,
        )
        grid_verts = mc.verts.numpy()
        triangles = mc.indices.numpy().reshape(-1, 3)
        keep_vertex = valid.numpy().astype(bool)
        keep = keep_vertex[triangles].all(axis=1)
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
        )
