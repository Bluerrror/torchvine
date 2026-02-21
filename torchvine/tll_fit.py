"""Transformation local likelihood (TLL) nonparametric copula estimator."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F

from . import stats
from .interpolation import InterpolationGrid, make_normal_grid


def to_pseudo_obs(x: torch.Tensor, *, ties_method: str = "average", seeds: Sequence[int] = (5,),
                  weights: torch.Tensor | None = None) -> torch.Tensor:
    """Rank transform to (0,1), matching pyvinecopulib API.

    ties_method: 'average' or 'random' (random uses jitter for tie-breaking).
    """
    x = torch.as_tensor(x)
    if x.ndim == 1:
        x = x.unsqueeze(1)
    n, d = x.shape
    g = torch.Generator(device=x.device)
    if seeds:
        g.manual_seed(int(seeds[0]))

    if ties_method == "random":
        jitter = torch.rand((n, d), generator=g, device=x.device, dtype=x.dtype) * 1e-12
        order = torch.argsort(x + jitter, dim=0)
    else:
        order = torch.argsort(x, dim=0, stable=True)

    # Vectorized rank computation for all columns at once
    ranks = torch.empty_like(x)
    col_idx = torch.arange(d, device=x.device).unsqueeze(0).expand(n, d)
    row_ranks = torch.arange(1, n + 1, device=x.device, dtype=x.dtype).unsqueeze(1).expand(n, d)
    ranks[order, col_idx] = row_ranks
    out = ranks / (n + 1.0)
    return stats.clamp_unit(out)


def _pearson_corr(x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    x = torch.as_tensor(x)
    y = torch.as_tensor(y)
    if weights is None:
        x0 = x - x.mean()
        y0 = y - y.mean()
        den = torch.sqrt((x0 * x0).sum() * (y0 * y0).sum()).clamp_min(torch.finfo(x.dtype).tiny)
        return (x0 * y0).sum() / den
    w = torch.as_tensor(weights, device=x.device, dtype=x.dtype)
    w = w / w.sum().clamp_min(torch.finfo(x.dtype).tiny)
    mx = (w * x).sum()
    my = (w * y).sum()
    x0 = x - mx
    y0 = y - my
    cov = (w * x0 * y0).sum()
    vx = (w * x0 * x0).sum().clamp_min(torch.finfo(x.dtype).tiny)
    vy = (w * y0 * y0).sum().clamp_min(torch.finfo(x.dtype).tiny)
    return cov / torch.sqrt(vx * vy)


def _win_smoother(x: torch.Tensor, wl: int) -> torch.Tensor:
    # Port of vinecopulib::tools_stats::win (moving average with edge flattening).
    if wl <= 0:
        return x
    x = torch.as_tensor(x)
    n = x.numel()
    k = 2 * wl + 1
    kernel = torch.ones((1, 1, k), dtype=x.dtype, device=x.device) / float(k)
    y = F.conv1d(x.reshape(1, 1, n), kernel, padding=wl).reshape(n)
    if wl < n:
        y[:wl] = y[wl]
        y[-wl:] = y[n - wl - 1]
    return y


def _cef(x: torch.Tensor, order: torch.Tensor, inv_order: torch.Tensor, wl: int) -> torch.Tensor:
    # win(x[order])[inv_order]
    xs = x[order]
    ys = _win_smoother(xs, wl)
    return ys[inv_order]


def ace(data: torch.Tensor, *, weights: torch.Tensor | None = None) -> torch.Tensor:
    # Port of vinecopulib::tools_stats::ace for 2D data.
    data = torch.as_tensor(data)
    n = int(data.shape[0])
    n_dbl = float(n)
    dt = data.dtype
    dev = data.device

    if weights is None or weights.numel() == 0:
        w = torch.ones((n,), dtype=dt, device=dev)
        nw = 0
    else:
        w = torch.as_tensor(weights, dtype=dt, device=dev)
        if int(w.numel()) != n:
            raise ValueError("weights length must match number of rows")
        nw = n

    wl = int(math.ceil(n_dbl / 5.0))

    order0 = torch.argsort(data[:, 0])
    inv0 = torch.empty((n,), dtype=torch.long, device=data.device)
    inv0[order0] = torch.arange(n, device=data.device)

    order1 = torch.argsort(data[:, 1])
    inv1 = torch.empty((n,), dtype=torch.long, device=data.device)
    inv1[order1] = torch.arange(n, device=data.device)

    ranks = torch.stack([inv0.to(dt), inv1.to(dt)], dim=1)
    phi = ranks.clone()
    phi = phi - ((n_dbl - 1.0) / 2.0 - 1.0)
    phi = phi / math.sqrt(n_dbl * (n_dbl - 1.0) / 12.0)
    if nw > 0:
        phi[:, 0] = phi[:, 0] * w
        phi[:, 1] = phi[:, 1] * w

    outer_iter = 1
    outer_eps = 1.0
    outer_abs_err = 1.0

    while outer_iter <= 100 and outer_abs_err > 2e-15:
        inner_iter = 1
        inner_eps = 1.0
        inner_abs_err = 1.0
        while inner_iter <= 10 and inner_abs_err > 1e-4:
            phi[:, 1] = _cef(phi[:, 0] * w, order1, inv1, wl)
            m1 = phi[:, 1].sum() / n_dbl
            phi[:, 1] = phi[:, 1] - m1
            s1 = torch.sqrt((phi[:, 1] * phi[:, 1]).sum() / (n_dbl - 1.0)).clamp_min(torch.finfo(dt).tiny)
            phi[:, 1] = phi[:, 1] / s1

            inner_abs_err = inner_eps
            tmp = phi[:, 1] - phi[:, 0]
            inner_eps = float((tmp * tmp).sum().item() / n_dbl)
            inner_abs_err = abs(inner_abs_err - inner_eps)
            inner_iter += 1

        phi[:, 0] = _cef(phi[:, 1] * w, order0, inv0, wl)
        m0 = phi[:, 0].sum() / n_dbl
        phi[:, 0] = phi[:, 0] - m0
        s0 = torch.sqrt((phi[:, 0] * phi[:, 0]).sum() / (n_dbl - 1.0)).clamp_min(torch.finfo(dt).tiny)
        phi[:, 0] = phi[:, 0] / s0

        outer_abs_err = outer_eps
        tmp = phi[:, 1] - phi[:, 0]
        outer_eps = float((tmp * tmp).sum().item() / n_dbl)
        outer_abs_err = abs(outer_abs_err - outer_eps)
        outer_iter += 1

    return phi


def pairwise_mcor(x: torch.Tensor, *, weights: torch.Tensor | None = None) -> torch.Tensor:
    phi = ace(x, weights=weights)
    return _pearson_corr(phi[:, 0], phi[:, 1], weights=weights)


def fit_tll(
    data: torch.Tensor,
    *,
    method: str = "constant",
    mult: float = 1.0,
    weights: torch.Tensor | None = None,
    grid_size: int = 30,
) -> tuple[torch.Tensor, InterpolationGrid, float, float]:
    """Fit the TLL nonparametric copula on a fixed grid (Torch port, continuous-only)."""
    if method not in ("constant", "linear", "quadratic"):
        raise ValueError("method must be 'constant', 'linear', or 'quadratic'")
    if mult <= 0:
        raise ValueError("mult must be positive")

    u = stats.clamp_unit(torch.as_tensor(data)[:, :2])
    dt = u.dtype
    dev = u.device
    ps = to_pseudo_obs(u, seeds=(5,))
    z_data = stats.qnorm(ps)

    m = int(grid_size)
    grid_points = make_normal_grid(m, boundary_to_01=False, device=dev, dtype=dt)
    g0 = grid_points.repeat_interleave(m)
    g1 = grid_points.repeat(m)
    grid_2d = torch.stack([g0, g1], dim=1)
    z = stats.qnorm(grid_2d)

    # Bandwidth selection (ported; includes ACE-based mcor scaling).
    cor = _pearson_corr(z_data[:, 0], z_data[:, 1], weights=weights).clamp(-0.95, 0.95)
    cov = torch.tensor([[1.0, float(cor.item())], [float(cor.item()), 1.0]], dtype=dt, device=dev)
    n = float(z_data.shape[0])
    if method == "constant":
        mult0 = n ** (-1.0 / 3.0)
    else:
        degree = 1.0 if method == "linear" else 2.0
        mult0 = 1.5 * (n ** (-1.0 / (2.0 * degree + 1.0)))
    mcor = pairwise_mcor(z_data, weights=weights).clamp(-0.99, 0.99)
    base = (cor.abs() / mcor.abs().clamp_min(1e-6)).clamp_min(1e-12)
    scale = torch.pow(base, 0.5 * mcor)
    B = cov * (mult0 * float(mult)) * scale

    rB = torch.linalg.cholesky(B)
    irB = torch.inverse(rB)
    det_irB = torch.det(irB)

    z_eval = (irB @ z.t()).t()
    z_data2 = (irB @ z_data.t()).t()

    w = None
    if weights is not None and weights.numel() > 0:
        w = torch.as_tensor(weights, device=u.device, dtype=u.dtype)

    n_eval = z_eval.shape[0]
    n_data = z_data2.shape[0]
    kernel0 = stats.dnorm(torch.zeros((1, 2), device=u.device, dtype=u.dtype)).prod(dim=1)[0]

    if method == "constant":
        # Fully vectorized: compute all kernel values at once
        # z_eval: (n_eval, 2), z_data2: (n_data, 2)
        # diff: (n_eval, n_data, 2)
        diff = z_eval.unsqueeze(1) - z_data2.unsqueeze(0)
        # kernels: (n_eval, n_data)
        kernels = stats.dnorm(diff).prod(dim=2) * det_irB
        if w is not None:
            kernels = kernels * w.unsqueeze(0)
        f0 = kernels.mean(dim=1).clamp_min(torch.finfo(u.dtype).tiny)  # (n_eval,)
        ll_fit = torch.empty((n_eval, 2), device=u.device, dtype=u.dtype)
        
        ll_fit[:, 0] = f0
        ll_fit[:, 1] = kernel0 * det_irB * (1.0 / f0) / float(n_data)
    else:
        # linear/quadratic still per-grid-point (complex per-point operations)
        ll_fit = torch.empty((n_eval, 2), device=u.device, dtype=u.dtype)
        for k in range(n_eval):
            zz = z_data2 - z_eval[k, :]
            kernels = stats.dnorm(zz).prod(dim=1) * det_irB
            if w is not None:
                kernels = kernels * w
            f0 = kernels.mean().clamp_min(torch.finfo(u.dtype).tiny)

            resk = 1.0
            b = torch.zeros((2,), device=u.device, dtype=u.dtype)
            S = B
            zz2 = (irB @ zz.t()).t()
            f1 = (zz2 * kernels[:, None]).mean(dim=0)
            b = f1 / f0
            if method == "quadratic":
                zz3 = (zz2 * kernels[:, None]) / (f0 * float(n_data))
                bB = B @ b
                M = B @ (zz2.t() @ zz3) @ B - torch.outer(bB, bB)
                S = torch.inverse(M)
                resk = resk * math.sqrt(float(torch.det(S).clamp_min(0.0).item())) / float(det_irB.item())
            resk = resk * math.exp(-0.5 * float((b @ (S @ b)).item()))

            ll_fit[k, 0] = float(f0.item()) * resk
            ll_fit[k, 1] = kernel0 * det_irB * (1.0 / f0) / float(n_data)

    c = ll_fit[:, 0] / stats.dnorm(z).prod(dim=1)
    values = c.reshape(m, m)

    grid2 = grid_points.clone()
    grid2[0] = 0.0
    grid2[-1] = 1.0
    interp = InterpolationGrid(grid2, values)

    npars = float(max(1.0, ll_fit[:, 1].sum().item()))
    loglik = float(torch.log(interp.interpolate(u).clamp_min(torch.finfo(u.dtype).tiny)).sum().item())

    return values, interp, loglik, npars
