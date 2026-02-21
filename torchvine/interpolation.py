"""Grid interpolation utilities for the TLL copula estimator."""

from __future__ import annotations

import torch

from . import stats


def make_normal_grid(m: int = 30, *, boundary_to_01: bool = True, device=None, dtype=None) -> torch.Tensor:
    # Matches KernelBicop::make_normal_grid + boundary shift used by KernelBicop/TllBicop.
    if m < 2:
        raise ValueError("m must be >= 2")
    z = torch.linspace(-3.25, 3.25, steps=int(m), device=device, dtype=dtype)
    grid = stats.pnorm(z)
    if boundary_to_01:
        grid = grid.clone()
        grid[0] = 0.0
        grid[-1] = 1.0
    return grid


def _int_on_grid(upr: torch.Tensor, vals: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    # Vectorized version of InterpolationGrid::int_on_grid (trapezoid rule on a piecewise linear function).
    # upr: (n,), vals: (n,m), grid: (m,)
    upr = torch.as_tensor(upr, device=vals.device, dtype=vals.dtype)
    if upr.ndim != 1:
        upr = upr.reshape(-1)
    n, m = vals.shape
    if grid.ndim != 1 or grid.numel() != m:
        raise ValueError("grid must have shape (m,) matching vals second dimension")

    x0 = grid[:-1]  # (m-1,)
    x1 = grid[1:]
    v0 = vals[:, :-1]  # (n,m-1)
    v1 = vals[:, 1:]

    upr2 = upr[:, None]  # (n,1)

    # Full cells where upr >= x1
    full = upr2 >= x1[None, :]
    seg_full = (v0 + v1) * (x1 - x0)[None, :] * 0.5

    # Partial cell where x0 <= upr < x1
    part = (upr2 >= x0[None, :]) & (upr2 < x1[None, :])
    dx = (upr2 - x0[None, :]).clamp_min(0.0)
    # (2*v0 + (v1-v0)*dx/(x1-x0)) * dx / 2
    slope = (v1 - v0) / (x1 - x0)[None, :]
    seg_part = (2.0 * v0 + slope * dx) * dx * 0.5

    return (seg_full * full + seg_part * part).sum(dim=1)


class InterpolationGrid:
    # Torch port of vinecopulib::tools_interpolation::InterpolationGrid.
    def __init__(self, grid_points: torch.Tensor, values: torch.Tensor, norm_times: int = 3):
        self.grid_points = torch.as_tensor(grid_points)
        self.values = torch.as_tensor(values)
        if self.values.ndim != 2 or self.values.shape[0] != self.values.shape[1]:
            raise ValueError("values must be a square (m,m) tensor")
        if self.grid_points.ndim != 1 or self.grid_points.numel() != self.values.shape[0]:
            raise ValueError("grid_points must have shape (m,) matching values")
        self.normalize_margins(int(norm_times))

    def to(self, *args, **kwargs) -> "InterpolationGrid":
        self.grid_points = self.grid_points.to(*args, **kwargs)
        self.values = self.values.to(*args, **kwargs)
        return self

    def get_values(self) -> torch.Tensor:
        return self.values

    def set_values(self, values: torch.Tensor, norm_times: int = 3) -> None:
        values = torch.as_tensor(values, device=self.values.device, dtype=self.values.dtype)
        if values.shape != self.values.shape:
            raise ValueError(f"values must have shape {tuple(self.values.shape)}, got {tuple(values.shape)}")
        self.values = values
        self.normalize_margins(int(norm_times))

    def flip(self) -> None:
        self.values = self.values.t().contiguous()

    def normalize_margins(self, times: int) -> None:
        m = int(self.grid_points.numel())
        if times <= 0:
            return
        grid = self.grid_points
        ones_upr = torch.ones((1,), device=grid.device, dtype=grid.dtype)
        for _ in range(int(times)):
            # Rows (vectorized)
            row_integrals = _int_on_grid(
                ones_upr.expand(m), self.values, grid
            ).clamp_min(1e-20)
            self.values = self.values / row_integrals[:, None]
            # Cols (vectorized)
            col_integrals = _int_on_grid(
                ones_upr.expand(m), self.values.t().contiguous(), grid
            ).clamp_min(1e-20)
            self.values = self.values / col_integrals[None, :]

    def _indices(self, x0: torch.Tensor, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grid = self.grid_points.contiguous()
        m = grid.numel()
        i = torch.searchsorted(grid, x0.contiguous(), right=True) - 1
        j = torch.searchsorted(grid, x1.contiguous(), right=True) - 1
        i = i.clamp(0, m - 2)
        j = j.clamp(0, m - 2)
        return i, j

    def interpolate(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x, device=self.values.device, dtype=self.values.dtype)
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("x must have shape (n,2)")
        x0 = x[:, 0]
        x1 = x[:, 1]
        i, j = self._indices(x0, x1)
        grid = self.grid_points

        x_lo = grid[i]
        x_hi = grid[i + 1]
        y_lo = grid[j]
        y_hi = grid[j + 1]

        z11 = self.values[i, j]
        z12 = self.values[i, j + 1]
        z21 = self.values[i + 1, j]
        z22 = self.values[i + 1, j + 1]

        # bilinear interpolation identical to C++ code
        x2x1 = (x_hi - x_lo)
        y2y1 = (y_hi - y_lo)
        x2x = (x_hi - x0)
        y2y = (y_hi - x1)
        yy1 = (x1 - y_lo)
        xx1 = (x0 - x_lo)
        return (z11 * x2x * y2y + z21 * xx1 * y2y + z12 * x2x * yy1 + z22 * xx1 * yy1) / (x2x1 * y2y1)

    def integrate_1d(self, u: torch.Tensor, cond_var: int) -> torch.Tensor:
        # cond_var: 1 => fix u1, integrate in u2; 2 => fix u2, integrate in u1
        u = torch.as_tensor(u, device=self.values.device, dtype=self.values.dtype)
        if u.ndim != 2 or u.shape[1] != 2:
            raise ValueError("u must have shape (n,2)")
        if cond_var not in (1, 2):
            raise ValueError("cond_var must be 1 or 2")

        n = u.shape[0]
        m = int(self.grid_points.numel())
        grid = self.grid_points

        if cond_var == 1:
            upr = u[:, 1]
            tmpgrid = torch.stack([u[:, 0].repeat_interleave(m), grid.repeat(n)], dim=1)
        else:
            upr = u[:, 0]
            tmpgrid = torch.stack([grid.repeat(n), u[:, 1].repeat_interleave(m)], dim=1)

        tmpvals = self.interpolate(tmpgrid).reshape(n, m).clamp_min(1e-4)
        tmpint = _int_on_grid(upr, tmpvals, grid)
        int1 = _int_on_grid(torch.ones_like(upr), tmpvals, grid).clamp_min(1e-20)
        return (tmpint / int1).clamp(1e-10, 1.0 - 1e-10)

    def integrate_2d(self, u: torch.Tensor) -> torch.Tensor:
        u = torch.as_tensor(u, device=self.values.device, dtype=self.values.dtype)
        if u.ndim != 2 or u.shape[1] != 2:
            raise ValueError("u must have shape (n,2)")

        n = u.shape[0]
        m = int(self.grid_points.numel())
        grid = self.grid_points

        # For each x-grid point, integrate over y up to u2 (vectorized).
        # Build all (n*m, 2) evaluation points at once.
        gk = grid.unsqueeze(0).expand(n, m).reshape(n * m)  # (n*m,)
        g_rep = grid.unsqueeze(0).expand(n, m).reshape(-1).repeat_interleave(1)  # not needed
        # For each of m grid points, evaluate at all m y-grid points for all n samples
        gk_col = grid.repeat(n * m)  # (n*m*m,)
        gk_row = gk.repeat_interleave(m)  # (n*m*m,)
        tmpgrid_all = torch.stack([gk_row, gk_col], dim=1)  # (n*m*m, 2)
        tmpvals_all = self.interpolate(tmpgrid_all).reshape(n * m, m)  # (n*m, m)
        u2_rep = u[:, 1].repeat_interleave(m)  # (n*m,)
        tmpvals2 = _int_on_grid(u2_rep, tmpvals_all, grid).reshape(n, m)  # (n, m)

        tmpint = _int_on_grid(u[:, 0], tmpvals2, grid)
        tmpint1 = _int_on_grid(torch.ones((n,), device=u.device, dtype=u.dtype), tmpvals2, grid).clamp_min(1e-20)
        return (tmpint / tmpint1).clamp(1e-10, 1.0 - 1e-10)

