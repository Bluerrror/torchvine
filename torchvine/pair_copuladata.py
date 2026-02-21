"""Pair-copula data visualization — torch-native + matplotlib."""

from __future__ import annotations

import math
from typing import Optional

import torch


def pairs_copula_data(
    data,
    main: str = "",
    cols: Optional[list[str]] = None,
    grid_size: int = 50,
    bins: int = 20,
    scatter_size: float = 6.0,
):
    """Pair plot for copula data U in (0, 1)^d."""
    import matplotlib.pyplot as plt

    from .bicop import Bicop
    from .families import BicopFamily
    from .fit_controls import FitControlsBicop
    from .stats import kendall_tau

    if data is None:
        raise ValueError("`data` cannot be None.")
    U = torch.as_tensor(data)
    if U.ndim != 2:
        raise ValueError("`data` must be 2-D (n, d).")
    if U.numel() == 0:
        raise ValueError("`data` cannot be empty.")
    if not bool(((U > 0.0) & (U < 1.0)).all()):
        raise ValueError("All values must lie strictly in (0, 1).")
    if grid_size <= 0 or bins <= 0 or scatter_size <= 0:
        raise ValueError("grid_size, bins, and scatter_size must be positive.")

    n, d = int(U.shape[0]), int(U.shape[1])
    if n < 2:
        raise ValueError(f"Need >= 2 observations, got {n}.")
    if cols is not None and (len(cols) != d or not all(isinstance(c, str) for c in cols)):
        raise ValueError(f"`cols` must be a list of {d} strings or None.")
    if cols is None:
        cols = [f"var{i + 1}" for i in range(d)]

    z = torch.linspace(-3.0, 3.0, int(grid_size), dtype=U.dtype, device=U.device)
    Z1, Z2 = torch.meshgrid(z, z, indexing="xy")
    U1 = 0.5 * (1.0 + torch.erf(Z1 / math.sqrt(2.0)))
    U2 = 0.5 * (1.0 + torch.erf(Z2 / math.sqrt(2.0)))
    grid_u = torch.stack([U1.reshape(-1), U2.reshape(-1)], dim=1)
    isqrt2pi = 1.0 / math.sqrt(2.0 * math.pi)
    norm_Z1 = isqrt2pi * torch.exp(-0.5 * Z1 * Z1)
    norm_Z2 = isqrt2pi * torch.exp(-0.5 * Z2 * Z2)

    fig, axes = plt.subplots(d, d, figsize=(2.8 * d, 2.8 * d))
    if d == 1:
        axes = [[axes]]

    for i in range(d):
        for j in range(d):
            ax = axes[i][j] if d == 1 else axes[i, j]
            if i == j:
                ax.hist(U[:, j].detach().cpu().tolist(), bins=bins, range=(0.0, 1.0), density=True, edgecolor="white")
                ax.hlines(1.0, 0.0, 1.0, linestyles="dashed", linewidth=0.8)
                _set_01(ax)
            elif i < j:
                xj = U[:, j].detach().cpu().tolist()
                yi = U[:, i].detach().cpu().tolist()
                ax.scatter(xj, yi, s=scatter_size, alpha=0.6)
                try:
                    tau = kendall_tau(U[:, j], U[:, i])
                    txt = f"tau = {float(tau):.2f}"
                    fs = 10 + 8 * abs(float(tau))
                except Exception:
                    txt, fs = "tau = N/A", 10
                ax.text(0.5, 0.5, txt, transform=ax.transAxes, ha="center", va="center", fontsize=fs, weight="bold")
                _set_01(ax)
            else:
                uv = torch.stack([U[:, j], U[:, i]], dim=1)
                try:
                    ctrl = FitControlsBicop(family_set=[BicopFamily.tll])
                    cop = Bicop.from_data(uv, controls=ctrl)
                    cvals = cop.pdf(grid_u).reshape(int(grid_size), int(grid_size))
                    dens = cvals * norm_Z1 * norm_Z2
                    if float((dens.max() - dens.min()).abs().item()) < 1e-15:
                        dens = dens.clone()
                        dens[0, 0] = dens[0, 0] * 1.000001
                    ax.contour(
                        Z1.detach().cpu().tolist(),
                        Z2.detach().cpu().tolist(),
                        dens.detach().cpu().tolist(),
                        levels=[0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
                        linewidths=0.8,
                    )
                except Exception as exc:
                    ax.text(0.5, 0.5, f"Fit failed:\n{type(exc).__name__}", transform=ax.transAxes, ha="center", va="center", fontsize=10, weight="bold")
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
