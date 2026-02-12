"""Parameter optimization (MLE) for bivariate copulas."""

from __future__ import annotations

import math
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


def golden_section_maximize(
    f: Callable[[float], float],
    *,
    a: float,
    b: float,
    x0: float | None = None,
    max_iter: int = 60,
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

            def fk(v: float) -> float:
                xx = x.clone()
                xx[k] = float(v)
                return float(f(xx))

            res = golden_section_maximize(fk, a=a, b=b, x0=float(x[k].item()), max_iter=max_inner, tol=tol)
            n_eval += res.n_eval
            x[k] = float(res.x.item())
            if res.fun > best + 1e-12:
                best = float(res.fun)
                improved = True

        if not improved:
            break

    return OptimizeResult(x=x, fun=best, n_eval=n_eval)
