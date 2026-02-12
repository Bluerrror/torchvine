"""Statistical helper functions — Kendall's tau, normal CDF/quantile, etc."""

from __future__ import annotations

import math
import torch


def _as_tensor(x, *, device=None, dtype=None):
    if torch.is_tensor(x):
        t = x
        if device is not None:
            t = t.to(device=device)
        if dtype is not None:
            t = t.to(dtype=dtype)
        return t
    return torch.as_tensor(x, device=device, dtype=dtype)


def dnorm(x: torch.Tensor) -> torch.Tensor:
    x = _as_tensor(x)
    inv_sqrt_2pi = 0.39894228040143270286
    return inv_sqrt_2pi * torch.exp(-0.5 * x * x)


def pnorm(x: torch.Tensor) -> torch.Tensor:
    x = _as_tensor(x)
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def qnorm(u: torch.Tensor) -> torch.Tensor:
    u = _as_tensor(u)
    # torch.special.ndtri is the inverse of the standard normal CDF
    return torch.special.ndtri(u)


def clamp_unit(u: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    # Avoid infs in qnorm and log(0) etc.
    return u.clamp(min=eps, max=1.0 - eps)


def pbvnorm_drezner(z1: torch.Tensor, z2: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """Approximate bivariate normal CDF Phi_2(z1, z2; rho).

    This is a CPU/GPU-friendly approximation (no SciPy). Accuracy is good for
    most modeling needs but not bit-identical to boost/Fortran implementations.

    Reference: Drezner & Wesolowsky (1990) style approximations.
    """
    z1 = _as_tensor(z1)
    z2 = _as_tensor(z2)
    rho = _as_tensor(rho, device=z1.device, dtype=z1.dtype)

    # Handle extremes: if correlation ~ 0, CDF factorizes.
    near0 = rho.abs() < 1e-6
    out = torch.empty(torch.broadcast_shapes(z1.shape, z2.shape, rho.shape), device=z1.device, dtype=z1.dtype)
    if near0.any():
        out = torch.where(near0, pnorm(z1) * pnorm(z2), out)

    if (~near0).any():
        r = rho
        a = torch.as_tensor([0.3253030, 0.4211071, 0.1334425, 0.006374323], device=z1.device, dtype=z1.dtype)
        b = torch.as_tensor([0.1337764, 0.6243247, 1.3425378, 2.2626645], device=z1.device, dtype=z1.dtype)

        # Implementation based on Genz-style quadrature with fixed nodes.
        # Vectorized over batch; sum over 4 nodes.
        hs = (z1 * z1 + z2 * z2) * 0.5
        asr = torch.asin(r.clamp(-0.999999, 0.999999))

        # Expand for node summation
        t = (asr[..., None] * (b / 2.0))[...]
        sn = torch.sin(t)
        term = torch.exp((sn * z1[..., None] * z2[..., None] - hs[..., None]) / (1.0 - sn * sn))
        s = torch.sum(a * term, dim=-1)
        approx = pnorm(z1) * pnorm(z2) + s * asr / (2.0 * math.pi)
        out = torch.where(near0, out, approx)

    return out


def _log_beta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = _as_tensor(a)
    b = _as_tensor(b, device=a.device, dtype=a.dtype)
    return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)


def _betacf(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor, *, max_iter: int = 200, eps: float = 3e-14) -> torch.Tensor:
    # Continued fraction for incomplete beta (Numerical Recipes / Cephes style).
    a = _as_tensor(a)
    b = _as_tensor(b, device=a.device, dtype=a.dtype)
    x = _as_tensor(x, device=a.device, dtype=a.dtype)

    tiny = torch.finfo(a.dtype).tiny
    one = torch.ones((), device=a.device, dtype=a.dtype)

    qab = a + b
    qap = a + one
    qam = a - one

    c = one
    d = one - qab * x / qap
    d = one / d.clamp_min(tiny)
    h = d

    for m in range(1, max_iter + 1):
        m_t = torch.as_tensor(float(m), device=a.device, dtype=a.dtype)
        m2 = 2.0 * m_t

        aa = m_t * (b - m_t) * x / ((qam + m2) * (a + m2))
        d = one + aa * d
        d = one / d.clamp_min(tiny)
        c = one + aa / c.clamp_min(tiny)
        h = h * d * c

        aa = -(a + m_t) * (qab + m_t) * x / ((a + m2) * (qap + m2))
        d = one + aa * d
        d = one / d.clamp_min(tiny)
        c = one + aa / c.clamp_min(tiny)
        delta = d * c
        h = h * delta

        if torch.max(torch.abs(delta - one)).item() < eps:
            break

    return h


def betainc_reg(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Regularized incomplete beta I_x(a,b) in pure torch (no SciPy)."""
    a = _as_tensor(a)
    b = _as_tensor(b, device=a.device, dtype=a.dtype)
    x = _as_tensor(x, device=a.device, dtype=a.dtype).clamp(0.0, 1.0)

    tiny = torch.finfo(a.dtype).tiny
    one = torch.ones((), device=a.device, dtype=a.dtype)

    # Avoid log(0) / nan; endpoints are exact.
    out = torch.zeros(torch.broadcast_shapes(a.shape, b.shape, x.shape), device=a.device, dtype=a.dtype)
    out = torch.where(x <= 0.0, torch.zeros_like(out), out)
    out = torch.where(x >= 1.0, torch.ones_like(out), out)

    mask = (x > 0.0) & (x < 1.0)
    if not mask.any():
        return out

    xx = x[mask]
    aa = a.expand(out.shape)[mask]
    bb = b.expand(out.shape)[mask]

    # bt = exp(log(Beta(a+b)) - log(Beta(a)) - log(Beta(b)) + a*log(x) + b*log(1-x))
    bt = torch.exp(aa * torch.log(xx.clamp_min(tiny)) + bb * torch.log((one - xx).clamp_min(tiny)) - _log_beta(aa, bb))

    thresh = (aa + one) / (aa + bb + 2.0 * one)
    use_direct = xx < thresh

    cf1 = _betacf(aa, bb, xx)
    val1 = bt * cf1 / aa

    cf2 = _betacf(bb, aa, one - xx)
    val2 = one - bt * cf2 / bb

    val = torch.where(use_direct, val1, val2)
    out[mask] = val.clamp(0.0, 1.0)
    return out


def pearson_cor(x: torch.Tensor, y: torch.Tensor, *, weights: torch.Tensor | None = None) -> float:
    """Weighted Pearson correlation (weights optional)."""
    x = _as_tensor(x, dtype=torch.float64).reshape(-1)
    y = _as_tensor(y, device=x.device, dtype=x.dtype).reshape(-1)
    if x.numel() != y.numel():
        raise ValueError("x and y must have the same length")
    if x.numel() == 0:
        return float("nan")

    if weights is None or weights.numel() == 0:
        xm = x.mean()
        ym = y.mean()
        xc = x - xm
        yc = y - ym
        num = (xc * yc).sum()
        den = torch.sqrt((xc * xc).sum() * (yc * yc).sum()).clamp_min(torch.finfo(x.dtype).tiny)
        return float((num / den).clamp(-1.0, 1.0).item())

    w = _as_tensor(weights, device=x.device, dtype=x.dtype).reshape(-1)
    if w.numel() != x.numel():
        raise ValueError("weights must have same length as x")
    ws = w.sum().clamp_min(torch.finfo(x.dtype).tiny)
    xm = (w * x).sum() / ws
    ym = (w * y).sum() / ws
    xc = x - xm
    yc = y - ym
    num = (w * xc * yc).sum()
    den = torch.sqrt((w * xc * xc).sum() * (w * yc * yc).sum()).clamp_min(torch.finfo(x.dtype).tiny)
    return float((num / den).clamp(-1.0, 1.0).item())


def _merge_count_inversions(arr: torch.Tensor) -> int:
    """Count inversions in *arr* using merge-sort. GPU-friendly (stays on device)."""
    n = arr.numel()
    if n <= 1:
        return 0
    mid = n // 2
    left = arr[:mid].clone()
    right = arr[mid:].clone()
    inv = _merge_count_inversions(left) + _merge_count_inversions(right)
    i = j = k = 0
    nl, nr = left.numel(), right.numel()
    while i < nl and j < nr:
        if left[i].item() <= right[j].item():
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            inv += nl - i
            j += 1
        k += 1
    while i < nl:
        arr[k] = left[i]
        i += 1
        k += 1
    while j < nr:
        arr[k] = right[j]
        j += 1
        k += 1
    return inv


def kendall_tau(x: torch.Tensor, y: torch.Tensor, *, weights: torch.Tensor | None = None) -> float:
    """Kendall's tau for (mostly) continuous data.

    Unweighted path: O(n log n) via merge-sort inversion counting (GPU-friendly).
    Weighted path: vectorized O(n^2) using torch (GPU-friendly, moderate n).
    """
    x = _as_tensor(x, dtype=torch.float64).reshape(-1)
    y = _as_tensor(y, device=x.device, dtype=x.dtype).reshape(-1)
    n = int(x.numel())
    if n != int(y.numel()):
        raise ValueError("x and y must have the same length")
    if n < 2:
        return float("nan")

    if weights is not None and weights.numel() > 0:
        w = _as_tensor(weights, device=x.device, dtype=x.dtype).reshape(-1)
        if int(w.numel()) != n:
            raise ValueError("weights must have same length as x")
        if n > 5000:
            raise ValueError("weighted kendall_tau is O(n^2); omit weights or subsample for large n")

        # Vectorized pairwise computation (stays on device)
        dx = x.unsqueeze(0) - x.unsqueeze(1)  # (n, n)
        dy = y.unsqueeze(0) - y.unsqueeze(1)
        s = dx * dy
        ww = w.unsqueeze(0) * w.unsqueeze(1)
        # Upper triangle only (i < j)
        mask = torch.triu(torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=1)
        conc = (ww * (s > 0).to(ww.dtype))[mask].sum()
        disc = (ww * (s < 0).to(ww.dtype))[mask].sum()
        denom = conc + disc
        return 0.0 if float(denom.item()) == 0.0 else float(((conc - disc) / denom).clamp(-1.0, 1.0).item())

    # Sort by x and count inversions in y (stays on device).
    idx = torch.argsort(x)
    y_sorted = y[idx]

    # Rank-compress y to 1..n (stable, on-device).
    order = torch.argsort(y_sorted, stable=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, n + 1, device=x.device, dtype=order.dtype)

    inv = _merge_count_inversions(ranks.to(torch.int64).clone())
    tau = 1.0 - 4.0 * float(inv) / (float(n) * float(n - 1))
    return float(max(-1.0, min(1.0, tau)))
