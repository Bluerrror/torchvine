"""Parameter optimization (MLE) for bivariate copulas.

This module is torch-native. If ``pytorch-minimize`` is installed it is used
as the primary backend; otherwise robust pure-PyTorch fallbacks are used.
"""

from __future__ import annotations

import math
from contextlib import suppress
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class OptimizeResult:
    x: torch.Tensor
    fun: float  # objective value at x (maximized)
    n_eval: int


def _as_float(x: torch.Tensor | float) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def _to_box(z: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    # Smoothly maps unconstrained z to [lb, ub].
    return lb + (ub - lb) * torch.sigmoid(z)


def _from_box(x: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    s = ((x - lb) / (ub - lb).clamp_min(eps)).clamp(eps, 1.0 - eps)
    return torch.log(s) - torch.log1p(-s)


def _try_pytorch_minimize(
    f_neg: Callable[[torch.Tensor], torch.Tensor],
    *,
    z0: torch.Tensor,
    max_iter: int,
    tol: float,
) -> tuple[torch.Tensor, int] | None:
    """Try using pytorch-minimize if available.

    The package API differs by version; we support known entry points
    opportunistically and return None when unavailable.
    """
    with suppress(Exception):
        try:
            import torchmin as _torchmin  # type: ignore[import-not-found]
            fn = getattr(_torchmin, "minimize", None)
        except Exception:
            import pytorch_minimize as _pytorch_minimize  # type: ignore[import-not-found]
            fn = getattr(_pytorch_minimize, "minimize", None)
        if callable(fn):
            # torchmin API: minimize(fun, x0, method, ...)
            try:
                res = fn(
                    f_neg,
                    z0.detach().clone(),
                    method="l-bfgs",
                    max_iter=max_iter,
                    tol=tol,
                )
            except TypeError:
                res = fn(f_neg, z0.detach().clone(), "l-bfgs")
            x = getattr(res, "x", None)
            nit = int(getattr(res, "nit", max_iter))
            if x is not None and torch.is_tensor(x):
                return x.detach().clone(), nit
    return None


def _lbfgs_minimize(
    f_neg: Callable[[torch.Tensor], torch.Tensor],
    *,
    z0: torch.Tensor,
    max_iter: int,
    tol: float,
) -> tuple[torch.Tensor, int]:
    z = torch.nn.Parameter(z0.detach().clone())
    opt = torch.optim.LBFGS(
        [z],
        max_iter=max(1, int(max_iter)),
        tolerance_grad=float(tol),
        tolerance_change=float(tol),
        line_search_fn="strong_wolfe",
    )
    n_eval = 0

    def closure() -> torch.Tensor:
        nonlocal n_eval
        opt.zero_grad(set_to_none=True)
        loss = f_neg(z)
        loss.backward()
        n_eval += 1
        return loss

    opt.step(closure)
    return z.detach().clone(), n_eval


def _torch_minimize_bounded_with_grad(
    f_tensor: Callable[[torch.Tensor], torch.Tensor],
    *,
    x0: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> OptimizeResult:
    x0 = torch.as_tensor(x0, dtype=torch.float64).reshape(-1)
    lb = torch.as_tensor(lb, dtype=torch.float64).reshape(-1)
    ub = torch.as_tensor(ub, dtype=torch.float64).reshape(-1)
    x0 = torch.max(torch.min(x0, ub), lb)
    z0 = _from_box(x0, lb, ub)
    n_eval = 0

    def f_neg(z: torch.Tensor) -> torch.Tensor:
        nonlocal n_eval
        n_eval += 1
        x = _to_box(z, lb, ub)
        return -f_tensor(x)

    z_opt, n_backend = _try_pytorch_minimize(f_neg, z0=z0, max_iter=max_iter, tol=tol) or _lbfgs_minimize(
        f_neg, z0=z0, max_iter=max_iter, tol=tol
    )
    x_opt = _to_box(z_opt, lb, ub)
    with torch.no_grad():
        best = float(f_tensor(x_opt).detach().cpu().item())
    return OptimizeResult(x=x_opt.detach().clone(), fun=best, n_eval=max(n_eval, n_backend))


def golden_section_maximize(
    f: Callable[[float], float],
    *,
    a: float,
    b: float,
    x0: float | None = None,
    max_iter: int = 30,
    tol: float = 1e-10,
) -> OptimizeResult:
    """Bounded 1D maximization using golden section search.

    Assumes f is unimodal-ish on [a,b]. This mirrors vinecopulib's use of
    Brent-like bounded search for 1-parameter families.
    """
    if not (a < b):
        raise ValueError("require a < b")

    phi = (1.0 + math.sqrt(5.0)) / 2.0
    invphi = 1.0 / phi

    # Initial interior points.
    if x0 is not None:
        # Bias the initial bracket around x0 if provided.
        x0 = float(min(max(x0, a), b))
        span = b - a
        c = max(a + 0.25 * span, min(b - 0.25 * span, x0 - 0.1 * span))
        d = min(b - 0.25 * span, max(a + 0.25 * span, x0 + 0.1 * span))
    else:
        c = b - (b - a) * invphi
        d = a + (b - a) * invphi

    fc = f(float(c))
    fd = f(float(d))
    n_eval = 2

    for _ in range(max_iter):
        if abs(b - a) <= tol * (1.0 + abs(a) + abs(b)):
            break
        if fc > fd:
            b = d
            d = c
            fd = fc
            c = b - (b - a) * invphi
            fc = f(float(c))
            n_eval += 1
        else:
            a = c
            c = d
            fc = fd
            d = a + (b - a) * invphi
            fd = f(float(d))
            n_eval += 1

    if fc > fd:
        x = c
        fun = fc
    else:
        x = d
        fun = fd
    return OptimizeResult(x=torch.tensor(float(x), dtype=torch.float64), fun=float(fun), n_eval=n_eval)


def coordinate_descent_maximize(
    f: Callable[[torch.Tensor], float],
    *,
    x0: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    max_outer: int = 7,
    max_inner: int = 60,
    tol: float = 1e-8,
) -> OptimizeResult:
    """Derivative-free coordinate descent with 1D golden-section substeps."""
    x = torch.as_tensor(x0, dtype=torch.float64).reshape(-1).clone()
    lb = torch.as_tensor(lb, dtype=torch.float64).reshape(-1)
    ub = torch.as_tensor(ub, dtype=torch.float64).reshape(-1)
    if x.numel() != lb.numel() or x.numel() != ub.numel():
        raise ValueError("x0, lb, ub must have the same length")
    x = torch.max(torch.min(x, ub), lb)

    best = float(f(x))
    n_eval = 1

    for _ in range(max_outer):
        improved = False
        for k in range(int(x.numel())):
            a = float(lb[k].item())
            b = float(ub[k].item())
            if not (a < b):
                continue

            saved_val = float(x[k].item())

            def fk(v: float, _k=k) -> float:
                x[_k] = float(v)
                return float(f(x))

            res = golden_section_maximize(fk, a=a, b=b, x0=saved_val, max_iter=max_inner, tol=tol)
            n_eval += res.n_eval
            x[k] = float(res.x.item())
            if res.fun > best + 1e-12:
                best = float(res.fun)
                improved = True

        if not improved:
            break

    return OptimizeResult(x=x, fun=best, n_eval=n_eval)


def torch_maximize_1d(
    f: Callable[[float], float],
    *,
    a: float,
    b: float,
    x0: float | None = None,
    tol: float = 1e-10,
) -> OptimizeResult:
    # Generic black-box fallback (non-differentiable objective).
    return golden_section_maximize(f, a=a, b=b, x0=x0, tol=tol)


def torch_maximize_1d_with_grad(
    f_tensor: Callable[[torch.Tensor], torch.Tensor],
    *,
    a: float,
    b: float,
    x0: float | None = None,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> OptimizeResult:
    lb = torch.tensor([float(a)], dtype=torch.float64)
    ub = torch.tensor([float(b)], dtype=torch.float64)
    if x0 is None:
        x0 = 0.5 * (float(a) + float(b))
    x0_t = torch.tensor([float(x0)], dtype=torch.float64)

    def f_vec(x: torch.Tensor) -> torch.Tensor:
        return f_tensor(x.reshape(-1)[0])

    res = _torch_minimize_bounded_with_grad(f_vec, x0=x0_t, lb=lb, ub=ub, tol=tol, max_iter=max_iter)
    return OptimizeResult(x=res.x.reshape(-1)[0].detach().clone(), fun=res.fun, n_eval=res.n_eval)


def torch_maximize_bounded(
    f: Callable[[torch.Tensor], float],
    *,
    x0: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    tol: float = 1e-8,
) -> OptimizeResult:
    # Generic black-box fallback (non-differentiable objective).
    return coordinate_descent_maximize(f, x0=x0, lb=lb, ub=ub, tol=tol)


def torch_maximize_bounded_with_grad(
    f_tensor: Callable[[torch.Tensor], torch.Tensor],
    *,
    x0: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> OptimizeResult:
    return _torch_minimize_bounded_with_grad(f_tensor, x0=x0, lb=lb, ub=ub, tol=tol, max_iter=max_iter)


def scipy_maximize_1d(
    f: Callable[[float], float],
    *,
    a: float,
    b: float,
    x0: float | None = None,
    tol: float = 1e-10,
) -> OptimizeResult:
    """Bounded 1D maximization.

    Keeps the existing function name for compatibility; implementation is
    torch-native (golden-section), with no SciPy dependency.
    """
    return torch_maximize_1d(f, a=a, b=b, x0=x0, tol=tol)


def scipy_maximize_bounded(
    f: Callable[[torch.Tensor], float],
    *,
    x0: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    tol: float = 1e-8,
) -> OptimizeResult:
    """Multi-dimensional box-constrained maximization.

    For scalar-returning black-box objectives we keep a derivative-free fallback.
    """
    return torch_maximize_bounded(f, x0=x0, lb=lb, ub=ub, tol=tol)


def scipy_maximize_bounded_with_grad(
    f_tensor: Callable[[torch.Tensor], torch.Tensor],
    *,
    x0: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    tol: float = 1e-8,
) -> OptimizeResult:
    """Multi-dimensional box-constrained maximization with autograd.

    Keeps compatibility name; implementation prefers pytorch-minimize and falls
    back to torch.optim.LBFGS.
    """
    try:
        return torch_maximize_bounded_with_grad(f_tensor, x0=x0, lb=lb, ub=ub, tol=tol)
    except Exception:
        def f_float(x: torch.Tensor) -> float:
            with torch.no_grad():
                return float(f_tensor(x).item())

        return scipy_maximize_bounded(f_float, x0=x0, lb=lb, ub=ub, tol=tol)
