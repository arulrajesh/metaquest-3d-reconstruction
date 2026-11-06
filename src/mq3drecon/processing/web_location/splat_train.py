"""Gaussian splat training on the laptop GPU (gsplat + PyTorch), from color keyframes with known poses.

Camera poses come straight from the headset (no SfM): views use MQ3DRecon's
Open3D camera convention (X right, Y down, Z forward), which is the OpenCV
convention gsplat expects. Gaussians are seeded from the fused TSDF mesh's
colored vertices. Loss is 0.8·L1 + 0.2·(1 − SSIM); densification, pruning and
opacity resets use gsplat's ``DefaultStrategy``. Every ``holdout_every``-th view
is held out to report an honest PSNR.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from mq3drecon.processing.web_location.splat_io import SH_C0
from mq3drecon.processing.web_location.texture_bake import BakeView


@dataclass(frozen=True)
class SplatSettings:
    iterations: int = 7000
    downscale: int = 2
    max_init_points: int = 300_000
    sh_degree: int = 3
    holdout_every: int = 8
    seed: int = 0
    # Real captures: headset poses are a little off for the passthrough cameras, and auto-exposure
    # changes between frames. Both make views disagree, which shows up as blur and floaters.
    pose_opt: bool = False  # learn a small SE(3) correction per training view
    pose_opt_lr: float = 5e-5
    exposure_opt: bool = False  # learn a per-view RGB gain and bias applied to the render
    exposure_opt_lr: float = 1e-3


@dataclass(frozen=True)
class SplatResult:
    means: np.ndarray  # (N, 3) Open3D world
    scales: np.ndarray  # (N, 3) linear
    quats_wxyz: np.ndarray  # (N, 4)
    rgb: np.ndarray  # (N, 3) in [0, 1] (SH DC)
    opacity: np.ndarray  # (N,)
    train_psnr: float
    holdout_psnr: float
    seconds: float


def _ssim(a, b):
    import torch
    import torch.nn.functional as F

    window = torch.exp(-((torch.arange(11, device=a.device, dtype=a.dtype) - 5) ** 2) / (2 * 1.5 ** 2))
    window = (window / window.sum())
    kernel = (window[:, None] * window[None, :]).expand(3, 1, 11, 11).contiguous()
    x, y = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    mu_x, mu_y = F.conv2d(x, kernel, padding=5, groups=3), F.conv2d(y, kernel, padding=5, groups=3)
    sxx = F.conv2d(x * x, kernel, padding=5, groups=3) - mu_x ** 2
    syy = F.conv2d(y * y, kernel, padding=5, groups=3) - mu_y ** 2
    sxy = F.conv2d(x * y, kernel, padding=5, groups=3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))).mean()


def _psnr(a, b) -> float:
    import torch

    mse = torch.mean((a - b) ** 2).item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def train_splats(
    views: list[BakeView],
    init_points: np.ndarray,
    init_colors: np.ndarray,
    settings: SplatSettings | None = None,
    device: str = "cuda",
) -> SplatResult:
    import cv2
    import torch

    from mq3drecon.processing.web_location.msvc_env import prepare_gsplat_build

    prepare_gsplat_build()  # before gsplat is imported: its CUDA kernels JIT-compile on first use
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy
    from scipy.spatial import cKDTree

    settings = settings or SplatSettings()
    start = time.time()
    rng = np.random.default_rng(settings.seed)
    torch.manual_seed(settings.seed)

    # Views (downscaled), split into train / holdout.
    images, viewmats, ks = [], [], []
    for view in views:
        h, w = view.rgb.shape[:2]
        sw, sh = w // settings.downscale, h // settings.downscale
        images.append(cv2.resize(view.rgb, (sw, sh), interpolation=cv2.INTER_AREA))
        k = view.intrinsic.copy().astype(np.float64)
        k[0, :] *= sw / w
        k[1, :] *= sh / h
        ks.append(k)
        viewmats.append(view.world_to_camera)
    height, width = images[0].shape[:2]
    images_t = torch.tensor(np.stack(images), dtype=torch.float32, device=device) / 255.0
    viewmats_t = torch.tensor(np.stack(viewmats), dtype=torch.float32, device=device)
    ks_t = torch.tensor(np.stack(ks), dtype=torch.float32, device=device)
    all_idx = np.arange(len(views))
    holdout = all_idx[settings.holdout_every - 1 :: settings.holdout_every] if len(views) >= 2 * settings.holdout_every else np.array([], int)
    train = np.setdiff1d(all_idx, holdout)

    # Seed Gaussians from the fused mesh's colored vertices.
    points = np.asarray(init_points, np.float64)
    colors = np.asarray(init_colors, np.float64)
    if len(points) > settings.max_init_points:
        keep = rng.choice(len(points), settings.max_init_points, replace=False)
        points, colors = points[keep], colors[keep]
    distances, _ = cKDTree(points).query(points, k=4)
    log_scale = np.log(np.clip(distances[:, 1:].mean(axis=1), 1e-4, 0.5))
    n = len(points)
    centers = np.array([np.linalg.inv(v.world_to_camera)[:3, 3] for v in views])
    scene_scale = float(np.linalg.norm(centers - centers.mean(axis=0), axis=1).max() * 1.1) or 1.0

    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.tensor(points, dtype=torch.float32, device=device)),
        "scales": torch.nn.Parameter(torch.tensor(np.repeat(log_scale[:, None], 3, axis=1), dtype=torch.float32, device=device)),
        "quats": torch.nn.Parameter(torch.tensor(np.tile([1.0, 0, 0, 0], (n, 1)), dtype=torch.float32, device=device)),
        "opacities": torch.nn.Parameter(torch.full((n,), math.log(0.1 / 0.9), dtype=torch.float32, device=device)),
        "sh0": torch.nn.Parameter(torch.tensor((colors - 0.5) / SH_C0, dtype=torch.float32, device=device)[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros((n, (settings.sh_degree + 1) ** 2 - 1, 3), dtype=torch.float32, device=device)),
    })
    rates = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {name: torch.optim.Adam([{"params": params[name], "lr": lr, "name": name}], eps=1e-15) for name, lr in rates.items()}
    means_decay = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / settings.iterations))
    strategy = DefaultStrategy(verbose=False, refine_stop_iter=int(settings.iterations * 0.75))
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    # Per-view corrections, outside the Gaussian ParameterDict (the densification strategy must not see them).
    n_views = len(views)
    pose_delta = torch.zeros((n_views, 6), dtype=torch.float32, device=device, requires_grad=True)  # axis-angle, translation
    exposure = torch.zeros((n_views, 6), dtype=torch.float32, device=device, requires_grad=True)  # RGB gain-1, RGB bias
    extra_optimizers = []
    if settings.pose_opt:
        extra_optimizers.append(torch.optim.Adam([pose_delta], lr=settings.pose_opt_lr))
    if settings.exposure_opt:
        extra_optimizers.append(torch.optim.Adam([exposure], lr=settings.exposure_opt_lr))

    def corrected_viewmat(index: int):
        viewmat = viewmats_t[index]
        if not settings.pose_opt:
            return viewmat
        omega, translation = pose_delta[index, :3], pose_delta[index, 3:]
        skew = torch.zeros((3, 3), dtype=viewmat.dtype, device=viewmat.device)
        skew[0, 1], skew[0, 2], skew[1, 2] = -omega[2], omega[1], -omega[0]
        skew[1, 0], skew[2, 0], skew[2, 1] = omega[2], -omega[1], omega[0]
        rotation = torch.linalg.matrix_exp(skew)
        # camera_to_world' = camera_to_world @ D, so world_to_camera' = D^-1 @ world_to_camera.
        inverse_delta = torch.eye(4, dtype=viewmat.dtype, device=viewmat.device)
        inverse_delta[:3, :3] = rotation.T
        inverse_delta[:3, 3] = -rotation.T @ translation
        return inverse_delta @ viewmat

    def render(index: int, sh_degree: int, *, corrected: bool = True):
        viewmat = corrected_viewmat(index) if corrected else viewmats_t[index]
        out, _alpha, info = rasterization(
            means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]), colors=torch.cat([params["sh0"], params["shN"]], dim=1),
            viewmats=viewmat[None], Ks=ks_t[index : index + 1], width=width, height=height,
            sh_degree=sh_degree, packed=False,
        )
        image = out[0]
        if corrected and settings.exposure_opt:
            image = image * (1.0 + exposure[index, :3]) + exposure[index, 3:]
        return image.clamp(0.0, 1.0), info

    for step in range(settings.iterations):
        index = int(rng.choice(train))
        sh_degree = min(step // 1000, settings.sh_degree)
        image, info = render(index, sh_degree)
        target = images_t[index]
        loss = 0.8 * torch.abs(image - target).mean() + 0.2 * (1.0 - _ssim(image, target))
        strategy.step_pre_backward(params, optimizers, state, step, info)
        loss.backward()
        for optimizer in [*optimizers.values(), *extra_optimizers]:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        means_decay.step()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)

    with torch.no_grad():
        def mean_psnr(indices) -> float:
            if len(indices) == 0:
                return float("nan")
            return float(np.mean([_psnr(render(int(i), settings.sh_degree)[0], images_t[int(i)]) for i in indices]))

        train_psnr, holdout_psnr = mean_psnr(train[:: max(1, len(train) // 16)]), mean_psnr(holdout)
        opacity = torch.sigmoid(params["opacities"]).cpu().numpy()
        keep = opacity > 0.005
        sh0 = params["sh0"][:, 0, :].cpu().numpy()
        return SplatResult(
            means=params["means"].cpu().numpy()[keep],
            scales=torch.exp(params["scales"]).cpu().numpy()[keep],
            quats_wxyz=params["quats"].cpu().numpy()[keep],
            rgb=np.clip(0.5 + SH_C0 * sh0[keep], 0.0, 1.0),
            opacity=opacity[keep],
            train_psnr=train_psnr,
            holdout_psnr=holdout_psnr,
            seconds=time.time() - start,
        )
