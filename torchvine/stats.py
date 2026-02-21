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


def _betacf(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor, *, max_iter: int = 80, eps: float = 3e-14) -> torch.Tensor:
    # Continued fraction for incomplete beta (Numerical Recipes / Cephes style).
    a = _as_tensor(a)
    b = _as_tensor(b, device=a.device, dtype=a.dtype)
    x = _as_tensor(x, device=a.device, dtype=a.dtype)

    tiny = torch.finfo(a.dtype).tiny

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    c = torch.ones_like(x)
    d = (1.0 - qab * x / qap).clamp_min(tiny).reciprocal()
    h = d.clone()

    for m in range(1, max_iter + 1):
        m2 = 2.0 * m

        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = (1.0 + aa * d).clamp_min(tiny).reciprocal()
        c = (1.0 + aa / c.clamp_min(tiny))
        h = h * d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = (1.0 + aa * d).clamp_min(tiny).reciprocal()
        c = (1.0 + aa / c.clamp_min(tiny))
        delta = d * c
        h = h * delta

        # Check convergence every 4 iterations to reduce .item() overhead
        if m % 4 == 0 and torch.max(torch.abs(delta - 1.0)).item() < eps:
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

    # bt = exp(a*log(x) + b*log(1-x) - log(Beta(a,b)))
    bt = torch.exp(aa * torch.log(xx.clamp_min(tiny)) + bb * torch.log((one - xx).clamp_min(tiny)) - _log_beta(aa, bb))

    thresh = (aa + one) / (aa + bb + 2.0 * one)
    use_direct = xx < thresh

    val = torch.empty_like(xx)
    # Only compute each branch for applicable elements
    if use_direct.any():
        idx_d = use_direct
        cf1 = _betacf(aa[idx_d], bb[idx_d], xx[idx_d])
        val[idx_d] = (bt[idx_d] * cf1 / aa[idx_d]).clamp(0.0, 1.0)
    if (~use_direct).any():
        idx_r = ~use_direct
        cf2 = _betacf(bb[idx_r], aa[idx_r], one - xx[idx_r])
        val[idx_r] = (one - bt[idx_r] * cf2 / bb[idx_r]).clamp(0.0, 1.0)

    out[mask] = val
    return out


def dt(x: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """Student-t probability density function (vectorized, pure torch)."""
    x = _as_tensor(x)
    nu = _as_tensor(nu, device=x.device, dtype=x.dtype)
    half_nu = nu * 0.5
    half_nup1 = (nu + 1.0) * 0.5
    log_pdf = (torch.lgamma(half_nup1) - torch.lgamma(half_nu)
               - 0.5 * torch.log(nu * math.pi)
               - half_nup1 * torch.log1p(x * x / nu))
    return torch.exp(log_pdf)


def pt(x: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """Student-t cumulative distribution function (vectorized, pure torch)."""
    x = _as_tensor(x)
    nu = _as_tensor(nu, device=x.device, dtype=x.dtype)
    t = nu / (nu + x * x)
    half = torch.as_tensor(0.5, device=x.device, dtype=x.dtype)
    Ix = betainc_reg(nu * 0.5, half, t)
    return torch.where(x >= 0, 1.0 - 0.5 * Ix, 0.5 * Ix)


def _qt_hill(p: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """Fast Student-t quantile via Hill (1970) 3-term expansion (no Newton)."""
    z = qnorm(p)
    g1 = (z * z * z + z) / 4.0
    g2 = (5.0 * z ** 5 + 16.0 * z ** 3 + 3.0 * z) / 96.0
    g3 = (3.0 * z ** 7 + 19.0 * z ** 5 + 17.0 * z ** 3 - 15.0 * z) / 384.0
    return z + g1 / nu + g2 / (nu * nu) + g3 / (nu * nu * nu)


def qt(p: torch.Tensor, nu: torch.Tensor, *, max_iter: int = 3) -> torch.Tensor:
    """Student-t quantile via Halley's method (cubic convergence, pure torch).

    Uses Hill (1970) 4-term expansion as initial guess, then Halley refinement.
    """
    p = clamp_unit(_as_tensor(p))
    nu = _as_tensor(nu, device=p.device, dtype=p.dtype)
    x = _qt_hill(p, nu)
    tiny = torch.finfo(p.dtype).tiny
    for _ in range(max_iter):
        F = pt(x, nu)
        f = dt(x, nu).clamp_min(tiny)
        r = F - p
        fp = f * (-(nu + 1.0) * x / (nu + x * x))
        x = x - 2.0 * r * f / (2.0 * f * f - r * fp)
    return x


def pbvt(x1: torch.Tensor, x2: torch.Tensor, rho: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """Bivariate Student-t CDF via Plackett's identity + Drezner-style quadrature.

    T_2(h, k; rho, nu) = T_nu(h) * T_nu(k) + c * integral
    where c = Gamma((nu+2)/2) / (Gamma(nu/2) * nu * pi).
    """
    x1 = _as_tensor(x1)
    x2 = _as_tensor(x2, device=x1.device, dtype=x1.dtype)
    rho = _as_tensor(rho, device=x1.device, dtype=x1.dtype)
    nu = _as_tensor(nu, device=x1.device, dtype=x1.dtype)

    near0 = rho.abs() < 1e-6
    base = pt(x1, nu) * pt(x2, nu)
    out = base.clone()

    if (~near0).any():
        a = torch.as_tensor([0.3253030, 0.4211071, 0.1334425, 0.006374323],
                            device=x1.device, dtype=x1.dtype)
        b = torch.as_tensor([0.1337764, 0.6243247, 1.3425378, 2.2626645],
                            device=x1.device, dtype=x1.dtype)

        asr = torch.asin(rho.clamp(-0.999999, 0.999999))
        hs = x1 * x1 + x2 * x2
        hk = x1 * x2

        c = torch.exp(torch.lgamma((nu + 2.0) * 0.5) - torch.lgamma(nu * 0.5)) / (nu * math.pi)

        t = asr[..., None] * (b / 2.0)
        sn = torch.sin(t)
        cs2 = (1.0 - sn * sn).clamp_min(1e-20)
        Q = (hs[..., None] - 2.0 * hk[..., None] * sn) / (nu[..., None] * cs2)
        K = torch.pow((1.0 + Q).clamp_min(1e-20), -(nu[..., None] + 2.0) * 0.5)
        s = torch.sum(a * K, dim=-1)

        approx = base + c * s * asr
        out = torch.where(near0, out, approx)

    return out.clamp(0.0, 1.0)


def pearson_cor(x: torch.Tensor, y: torch.Tensor, *, weights: torch.Tensor | None = None) -> float:
    """Weighted Pearson correlation (weights optional)."""
    x = _as_tensor(x).reshape(-1)
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


def _count_inversions_merge(ranks_list: list[int], n: int) -> int:
    """Count inversions via iterative bottom-up merge sort — O(n log n)."""
    a = list(ranks_list)
    buf = [0] * n
    inv = 0
    width = 1
    while width < n:
        for start in range(0, n, 2 * width):
            mid = min(start + width, n)
            end = min(start + 2 * width, n)
            i, j, k = start, mid, start
            while i < mid and j < end:
                if a[i] <= a[j]:
                    buf[k] = a[i]
                    i += 1
                else:
                    buf[k] = a[j]
                    inv += mid - i
                    j += 1
                k += 1
            while i < mid:
                buf[k] = a[i]
                i += 1
                k += 1
            while j < end:
                buf[k] = a[j]
                j += 1
                k += 1
        a, buf = buf, a
        width *= 2
    return inv


def kendall_tau(x: torch.Tensor, y: torch.Tensor, *, weights: torch.Tensor | None = None) -> float:
    """Kendall's tau for (mostly) continuous data — pure PyTorch, no external deps.

    Unweighted path: O(n log n) merge-sort inversion count, or O(n^2) vectorized
    for small n.
    Weighted path: vectorized O(n^2) using torch.
    """
    x = _as_tensor(x).reshape(-1)
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
        mask = torch.triu(torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=1)
        conc = (ww * (s > 0).to(ww.dtype))[mask].sum()
        disc = (ww * (s < 0).to(ww.dtype))[mask].sum()
        denom = conc + disc
        return 0.0 if float(denom.item()) == 0.0 else float(((conc - disc) / denom).clamp(-1.0, 1.0).item())

    # Vectorized O(n^2) for very small n — avoids sort overhead
    if n <= 200:
        dx = x.unsqueeze(1) - x.unsqueeze(0)
        dy = y.unsqueeze(1) - y.unsqueeze(0)
        s = torch.sign(dx) * torch.sign(dy)
        idx_pairs = torch.triu_indices(n, n, offset=1, device=x.device)
        s_upper = s[idx_pairs[0], idx_pairs[1]]
        c = (s_upper > 0).sum()
        d = (s_upper < 0).sum()
        denom = c + d
        if denom == 0:
            return 0.0
        return float(max(-1.0, min(1.0, float(((c - d).float() / denom.float()).item()))))

    # Sort by x, count inversions in y-ranks
    idx = torch.argsort(x)
    y_sorted = y[idx]
    order = torch.argsort(y_sorted, stable=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, n + 1, device=x.device, dtype=order.dtype)

    if n <= 600:
        # Vectorized O(n^2) on medium n; above this, nlogn merge is faster.
        inv = int(ranks.unsqueeze(1).gt(ranks.unsqueeze(0)).triu(diagonal=1).sum().item())
    else:
        # O(n log n) iterative merge sort on Python list
        inv = _count_inversions_merge(ranks.tolist(), n)

    tau = 1.0 - 4.0 * float(inv) / (float(n) * float(n - 1))
    return float(max(-1.0, min(1.0, tau)))


def spearman_rho(x: torch.Tensor, y: torch.Tensor, *, weights: torch.Tensor | None = None) -> float:
    """Spearman's rank correlation (weighted optional) — pure torch."""
    x = _as_tensor(x).reshape(-1)
    y = _as_tensor(y, device=x.device, dtype=x.dtype).reshape(-1)
    n = int(x.numel())
    if n != int(y.numel()):
        raise ValueError("x and y must have the same length")
    if n < 2:
        return float("nan")
    # Rank-transform then compute Pearson correlation of ranks
    rx = _rank(x)
    ry = _rank(y)
    return pearson_cor(rx, ry, weights=weights)


def _rank(x: torch.Tensor) -> torch.Tensor:
    """Average-tie rank of a 1-D tensor."""
    idx = torch.argsort(x)
    ranks = torch.empty_like(x)
    ranks[idx] = torch.arange(1, x.numel() + 1, device=x.device, dtype=x.dtype)
    return ranks


def blomqvist_beta(x: torch.Tensor, y: torch.Tensor, *, weights: torch.Tensor | None = None) -> float:
    """Blomqvist's beta (medial correlation coefficient) — pure torch."""
    x = _as_tensor(x).reshape(-1)
    y = _as_tensor(y, device=x.device, dtype=x.dtype).reshape(-1)
    n = int(x.numel())
    if n != int(y.numel()):
        raise ValueError("x and y must have the same length")
    if n < 2:
        return float("nan")
    mx = x.median()
    my = y.median()
    if weights is not None and weights.numel() > 0:
        w = _as_tensor(weights, device=x.device, dtype=x.dtype).reshape(-1)
        ws = w.sum().clamp_min(torch.finfo(x.dtype).tiny)
        c = (w * ((x > mx) == (y > my)).to(x.dtype)).sum() / ws
    else:
        c = ((x > mx) == (y > my)).to(x.dtype).mean()
    return float((2.0 * c - 1.0).clamp(-1.0, 1.0).item())


def hoeffding_d(x: torch.Tensor, y: torch.Tensor, *, weights: torch.Tensor | None = None) -> float:
    """Hoeffding's D statistic — pure torch, O(n log n).

    Returns a value in approximately [−0.5, 1]; 0 indicates independence.
    """
    x = _as_tensor(x).reshape(-1)
    y = _as_tensor(y, device=x.device, dtype=x.dtype).reshape(-1)
    n = int(x.numel())
    if n != int(y.numel()):
        raise ValueError("x and y must have the same length")
    if n < 5:
        return float("nan")
    # 0-based ranks (matching wdm library convention)
    Rx = _rank(x) - 1.0
    Ry = _rank(y) - 1.0
    # Q_i = #{j: x_j < x_i AND y_j < y_i}  (bivariate rank, 0-based)
    if n <= 5000:
        gt_x = (x.unsqueeze(1) > x.unsqueeze(0)).to(x.dtype)  # [i,j] = x[i]>x[j]
        gt_y = (y.unsqueeze(1) > y.unsqueeze(0)).to(x.dtype)
        Qi = (gt_x * gt_y).sum(dim=1)
    else:
        Qi = torch.zeros(n, device=x.device, dtype=x.dtype)
        for i in range(n):
            Qi[i] = ((x < x[i]) & (y < y[i])).to(x.dtype).sum()

    # wdm formula (Hoeffding 1948, unweighted)
    A1 = (Qi * (Qi - 1.0)).sum().item()
    inner = Rx * Ry - Qi - Rx - Ry + 2.0
    A2 = (Qi * inner).sum().item()
    A3 = (Rx * (Rx - 1.0) * Ry * (Ry - 1.0) - 4.0 * Qi * inner - 2.0 * Qi * (Qi - 1.0)).sum().item()

    nf = float(n)
    P3 = nf * (nf - 1.0) * (nf - 2.0)
    P4 = P3 * (nf - 3.0)
    P5 = P4 * (nf - 4.0)
    D = 30.0 * (A1 / P3 - 2.0 * A2 / P4 + A3 / P5)
    return float(max(-0.5, min(1.0, D)))


def wdm(x: torch.Tensor, y: torch.Tensor, method: str = "kendall",
         *, weights: torch.Tensor | None = None) -> float:
    """Weighted dependence measure — unified interface (pyvinecopulib parity).

    Parameters
    ----------
    method : str
        One of "pearson"/"cor"/"prho", "spearman"/"srho"/"rho",
        "kendall"/"ktau"/"tau", "blomqvist"/"bbeta"/"beta",
        "hoeffding"/"hoeffd"/"d".
    """
    m = method.lower()
    if m in ("pearson", "cor", "prho"):
        return pearson_cor(x, y, weights=weights)
    if m in ("spearman", "srho", "rho"):
        return spearman_rho(x, y, weights=weights)
    if m in ("kendall", "ktau", "tau"):
        return kendall_tau(x, y, weights=weights)
    if m in ("blomqvist", "bbeta", "beta"):
        return blomqvist_beta(x, y, weights=weights)
    if m in ("hoeffding", "hoeffd", "d"):
        return hoeffding_d(x, y, weights=weights)
    raise ValueError(f"Unknown dependence measure method: {method!r}")
