"""Pair-copula data visualization — pure-torch + matplotlib.

Mirrors :func:`pyvinecopulib.pairs_copula_data`.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


def pairs_copula_data(
    data,
    main: str = "",
    cols: Optional[list[str]] = None,
    grid_size: int = 50,
    bins: int = 20,
    scatter_size: float = 6.0,
):
    """Pair plot for copula data *U* in (0, 1)^d.

    * **Lower**: bivariate copula density contours (TLL) in *z*-space.
    * **Upper**: scatter with Kendall's τ annotation (copula space).
    * **Diagonal**: histograms (copula space).

    Parameters
    ----------
    data : (n, d) array-like or Tensor
        Copula-scale data with entries strictly in (0, 1).
    main : str
        Figure title.
    cols : list[str] or None
        Column labels (length *d*).
    grid_size : int
        Contour grid resolution per dimension.
    bins : int
        Number of histogram bins on the diagonal.
    scatter_size : float
        Marker size for upper-panel scatter.

    Returns
    -------
    fig, axes : matplotlib Figure and Axes array of shape (d, d)
    """
    import matplotlib.pyplot as plt

    from .bicop import Bicop
    from .families import BicopFamily
    from .fit_controls import FitControlsBicop
    from .stats import kendall_tau

    # ---- input validation ------------------------------------------------
    if data is None:
        raise ValueError("`data` cannot be None.")
    U = np.asarray(data if not isinstance(data, torch.Tensor) else data.numpy(),
                   dtype=float)
    if U.ndim != 2:
        raise ValueError("`data` must be 2-D (n, d).")
    if U.size == 0:
        raise ValueError("`data` cannot be empty.")
    if not (np.all(U > 0.0) and np.all(U < 1.0)):
        raise ValueError("All values must lie strictly in (0, 1).")
    if grid_size <= 0 or bins <= 0 or scatter_size <= 0:
        raise ValueError("grid_size, bins, and scatter_size must be positive.")

    n, d = U.shape
    if n < 2:
        raise ValueError(f"Need ≥ 2 observations, got {n}.")
    if cols is not None and (len(cols) != d or not all(isinstance(c, str) for c in cols)):
        raise ValueError(f"`cols` must be a list of {d} strings or None.")
    if cols is None:
        cols = [f"var{i + 1}" for i in range(d)]

    # ---- z-space grid ----------------------------------------------------
    z = np.linspace(-3.0, 3.0, grid_size)
    Z1, Z2 = np.meshgrid(z, z, indexing="xy")
    _sqrt2 = math.sqrt(2.0)
    _isqrt2pi = 1.0 / math.sqrt(2.0 * math.pi)
    U1 = 0.5 * (1.0 + _erf_np(Z1 / _sqrt2))
    U2 = 0.5 * (1.0 + _erf_np(Z2 / _sqrt2))
    grid_u = np.column_stack([U1.ravel(), U2.ravel()])
    norm_Z1 = _isqrt2pi * np.exp(-0.5 * Z1 * Z1)
    norm_Z2 = _isqrt2pi * np.exp(-0.5 * Z2 * Z2)

    fig, axes = plt.subplots(d, d, figsize=(2.8 * d, 2.8 * d))
    if d == 1:
        axes = np.array([[axes]])

    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i == j:
                # --- diagonal: histogram ---
                ax.hist(U[:, j], bins=bins, range=(0.0, 1.0),
                        density=True, edgecolor="white")
                ax.hlines(1.0, 0.0, 1.0, linestyles="dashed", linewidth=0.8)
                _set_01(ax)
            elif i < j:
                # --- upper: scatter + tau ---
                ax.scatter(U[:, j], U[:, i], s=scatter_size, alpha=0.6)
                try:
                    tau = kendall_tau(
                        torch.as_tensor(U[:, j], dtype=torch.float64),
                        torch.as_tensor(U[:, i], dtype=torch.float64),
                    )
                    txt = f"\u03c4 = {tau:.2f}"
                    fs = 10 + 8 * abs(tau)
                except Exception:
                    txt, fs = "\u03c4 = N/A", 10
                ax.text(0.5, 0.5, txt, transform=ax.transAxes,
                        ha="center", va="center", fontsize=fs, weight="bold")
                _set_01(ax)
            else:
                # --- lower: TLL contours in z-space ---
                uv = np.column_stack([U[:, j], U[:, i]])
                try:
                    ctrl = FitControlsBicop(family_set=[BicopFamily.tll])
                    cop = Bicop(data=torch.as_tensor(uv, dtype=torch.float64),
                                controls=ctrl)
                    cvals = cop.pdf(torch.as_tensor(grid_u, dtype=torch.float64))
                    cvals = cvals.numpy().reshape(grid_size, grid_size)
                    dens = cvals * norm_Z1 * norm_Z2
                    if np.allclose(dens.min(), dens.max()):
                        dens = dens.copy()
                        dens.flat[0] *= 1.000001
                    ax.contour(Z1, Z2, dens,
                               levels=[0.01, 0.025, 0.05, 0.1, 0.15,
                                       0.2, 0.3, 0.4, 0.5],
                               linewidths=0.8)
                except Exception as exc:
                    ax.text(0.5, 0.5, f"Fit failed:\n{type(exc).__name__}",
                            transform=ax.transAxes, ha="center", va="center",
                            fontsize=10, weight="bold")
                _set_z(ax)

            ax.tick_params(labelbottom=True, labelleft=True)
            if i == d - 1:
                ax.set_xlabel(cols[j])
            if j == 0:
                ax.set_ylabel(cols[i])

    if main:
        fig.suptitle(main, y=1.02)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    return fig, axes


# ---- small helpers (avoid scipy import) --------------------------------

def _erf_np(x):
    """Vectorised erf using numpy (stdlib math.erf is scalar-only)."""
    return np.vectorize(math.erf)(x)


def _set_01(ax):
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])


def _set_z(ax):
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_xticks([-3.0, 0.0, 3.0])
    ax.set_yticks([-3.0, 0.0, 3.0])
