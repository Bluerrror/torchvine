"""One-dimensional kernel density estimation — pure torch.

Local polynomial density estimation following the approach of
Fan & Gijbels (1996): fit local polynomial regressions to the
empirical CDF, then read off the density from the linear coefficient.
"""

import math
from typing import Optional

import torch


class Kde1d:
    """Local polynomial kernel density estimation in 1D.

    Pure PyTorch implementation matching the pyvinecopulib Kde1d API.

    Parameters
    ----------
    xmin, xmax : float
        Bounds of the support (NaN = unbounded).
    type : str
        ``"continuous"`` or ``"discrete"``.
    multiplier : float
        Bandwidth multiplier.
    bandwidth : float or None
        Fixed bandwidth (``None`` = automatic selection).
    degree : int
        Degree of the local polynomial (0, 1, or 2).
    grid_size : int
        Number of *intervals* for the interpolation grid (grid has
        ``grid_size + 1`` points).
    """

    def __init__(
        self,
        *,
        xmin: float = float("nan"),
        xmax: float = float("nan"),
        type: str = "continuous",
        multiplier: float = 1.0,
        bandwidth: Optional[float] = None,
        degree: int = 2,
        grid_size: int = 400,
    ):
        self._xmin = float(xmin)
        self._xmax = float(xmax)
        self._type = type
        self._multiplier = float(multiplier)
        self._bw_init = bandwidth
        self._degree = min(max(int(degree), 0), 2)
        self._grid_size = max(int(grid_size), 4)

        self._fitted = False
        self._bw: float = float("nan")
        self._loglik_val: float = float("nan")
        self._edf_val: float = float("nan")
        self._prob0_val: float = 0.0
        self._nobs: int = 0
        self._grid_t: Optional[torch.Tensor] = None
        self._pdf_t: Optional[torch.Tensor] = None
        self._cdf_t: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_params(
        cls,
        *,
        xmin: float = float("nan"),
        xmax: float = float("nan"),
        type: str = "continuous",
        multiplier: float = 1.0,
        bandwidth: Optional[float] = None,
        degree: int = 2,
        grid_size: int = 400,
    ) -> "Kde1d":
        """Create a ``Kde1d`` object from parameters (needs ``fit()``)."""
        return cls(
            xmin=xmin, xmax=xmax, type=type, multiplier=multiplier,
            bandwidth=bandwidth, degree=degree, grid_size=grid_size,
        )

    @classmethod
    def from_grid(
        cls,
        grid_points,
        values,
        *,
        xmin: float = float("nan"),
        xmax: float = float("nan"),
        type: str = "continuous",
    ) -> "Kde1d":
        """Construct directly from pre-computed grid and PDF values."""
        grid_t = torch.as_tensor(grid_points, dtype=torch.float64).reshape(-1)
        pdf_t = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
        obj = cls(xmin=xmin, xmax=xmax, type=type, grid_size=grid_t.numel() - 1)
        obj._grid_t = grid_t
        obj._pdf_t = pdf_t
        obj._nobs = 0
        # Build CDF
        dx = grid_t[1] - grid_t[0]
        cdf = torch.zeros_like(pdf_t)
        cdf[1:] = torch.cumsum(0.5 * (pdf_t[:-1] + pdf_t[1:]) * dx, dim=0)
        total = cdf[-1].clamp(min=1e-300)
        obj._cdf_t = cdf / total
        obj._fitted = True
        return obj

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, x) -> "Kde1d":
        """Fit the density to data *x* (1-D array or tensor)."""
        x = torch.as_tensor(x, dtype=torch.float64).reshape(-1)
        if self._type == "discrete":
            return self._fit_discrete(x)
        return self._fit_continuous(x)

    # -- bandwidth selection -------------------------------------------
    @staticmethod
    def _silverman(x: torch.Tensor, degree: int, multiplier: float) -> float:
        n = x.numel()
        std = x.std().item()
        q75 = torch.quantile(x, 0.75).item()
        q25 = torch.quantile(x, 0.25).item()
        iqr = q75 - q25
        s = min(std, iqr / 1.349) if iqr > 0 else std
        if s <= 0:
            s = std if std > 0 else 1.0
        # Standard Silverman bandwidth, adjusted for local polynomial degree
        return max(1.06 * s * n ** (-0.2) * multiplier, 1e-10)

    # -- continuous fit ------------------------------------------------
    def _fit_continuous(self, x: torch.Tensor) -> "Kde1d":
        n = x.numel()
        self._nobs = n
        xs = x.sort().values

        # Bandwidth
        if self._bw_init is not None and not math.isnan(self._bw_init):
            self._bw = self._bw_init * self._multiplier
        else:
            self._bw = self._silverman(xs, self._degree, self._multiplier)
        h = self._bw

        has_lb = not math.isnan(self._xmin)
        has_ub = not math.isnan(self._xmax)

        pad = 4.0 * h
        lb = self._xmin if has_lb else xs[0].item() - pad
        ub = self._xmax if has_ub else xs[-1].item() + pad

        G = self._grid_size + 1
        grid = torch.linspace(lb, ub, G, dtype=xs.dtype, device=xs.device)

        # Reflection for boundary correction
        data = xs
        if has_lb:
            data = torch.cat([2.0 * self._xmin - xs.flip(0), data])
        if has_ub:
            data = torch.cat([data, 2.0 * self._xmax - xs.flip(0)])

        # Standard Gaussian KDE (always non-negative)
        N = data.numel()
        chunk = max(1, min(G, 50_000_000 // max(N, 1)))
        pdf_vals = torch.zeros(G, dtype=xs.dtype, device=xs.device)
        for s in range(0, G, chunk):
            e = min(s + chunk, G)
            g = grid[s:e]
            u = (g.unsqueeze(1) - data.unsqueeze(0)) / h
            w = torch.exp(-0.5 * u * u) / (math.sqrt(2.0 * math.pi) * h)
            pdf_vals[s:e] = w.sum(dim=1) / n

        # Normalise on the grid
        area = torch.trapezoid(pdf_vals, grid)
        if area > 0:
            pdf_vals = pdf_vals / area

        self._grid_t = grid
        self._pdf_t = pdf_vals

        # CDF via cumulative trapezoid
        dx = (ub - lb) / (G - 1)
        cdf = torch.zeros(G, dtype=xs.dtype, device=xs.device)
        cdf[1:] = torch.cumsum(0.5 * (pdf_vals[:-1] + pdf_vals[1:]) * dx, dim=0)
        cdf = cdf / cdf[-1].clamp(min=1e-300)
        if has_lb:
            cdf[0] = 0.0
        if has_ub:
            cdf[-1] = 1.0
        self._cdf_t = cdf

        # Log-likelihood
        pdf_data = self._interp(xs, grid, pdf_vals)
        self._loglik_val = torch.log(pdf_data.clamp(min=1e-300)).sum().item()

        # Effective degrees of freedom (rough)
        self._edf_val = float("nan")

        self._fitted = True
        return self

    # -- discrete fit --------------------------------------------------
    def _fit_discrete(self, x: torch.Tensor) -> "Kde1d":
        n = x.numel()
        self._nobs = n
        vals, counts = x.unique(sorted=True, return_counts=True)
        probs = counts.to(torch.float64) / n

        lb = vals[0].item()
        ub = vals[-1].item()
        G = int(ub - lb) + 1
        grid = torch.linspace(lb, ub, G, dtype=torch.float64, device=x.device)
        pdf_vals = torch.zeros(G, dtype=torch.float64, device=x.device)
        for v, p in zip(vals, probs):
            idx = int(v.item() - lb)
            if 0 <= idx < G:
                pdf_vals[idx] = p

        self._grid_t = grid
        self._pdf_t = pdf_vals

        cdf = torch.cumsum(pdf_vals, dim=0)
        self._cdf_t = cdf / cdf[-1].clamp(min=1e-300)

        self._bw = 1.0
        pdf_data = self._interp(x, grid, pdf_vals)
        self._loglik_val = torch.log(pdf_data.clamp(min=1e-300)).sum().item()
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @staticmethod
    def _interp(x: torch.Tensor, grid: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        lb = grid[0].item()
        ub = grid[-1].item()
        G = grid.numel()
        idx = (x - lb) / (ub - lb) * (G - 1)
        idx = idx.clamp(0.0, G - 2.0)
        lo = idx.long()
        hi = lo + 1
        frac = idx - lo.to(idx.dtype)
        return values[lo] * (1.0 - frac) + values[hi] * frac

    def pdf(self, x):
        """Evaluate the density at *x*."""
        self._check_fitted()
        xt = torch.as_tensor(x, dtype=torch.float64).reshape(-1)
        return self._interp(xt, self._grid_t, self._pdf_t).clamp(min=0.0)

    def cdf(self, x):
        """Evaluate the CDF at *x*."""
        self._check_fitted()
        xt = torch.as_tensor(x, dtype=torch.float64).reshape(-1)
        return self._interp(xt, self._grid_t, self._cdf_t).clamp(0.0, 1.0)

    def quantile(self, p):
        """Evaluate the quantile function at probability *p*."""
        self._check_fitted()
        pt = torch.as_tensor(p, dtype=torch.float64).reshape(-1).clamp(0.0, 1.0)
        idx = torch.searchsorted(self._cdf_t.contiguous(), pt.contiguous()).clamp(1, self._grid_t.numel() - 1)
        c0 = self._cdf_t[idx - 1]
        c1 = self._cdf_t[idx]
        dc = (c1 - c0).clamp(min=1e-300)
        frac = (pt - c0) / dc
        g0 = self._grid_t[idx - 1]
        g1 = self._grid_t[idx]
        return g0 + frac * (g1 - g0)

    def simulate(self, n: int, seeds=None):
        """Draw *n* random samples from the fitted density."""
        self._check_fitted()
        if seeds is not None:
            torch.manual_seed(seeds[0] if isinstance(seeds, (list, tuple)) else int(seeds))
        u = torch.rand(n, dtype=torch.float64)
        return self.quantile(u)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def set_xmin_xmax(self, xmin: float, xmax: float):
        self._xmin = float(xmin)
        self._xmax = float(xmax)

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Must call fit() first")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def bandwidth(self) -> float:
        return self._bw

    @property
    def degree(self) -> int:
        return self._degree

    @property
    def xmin(self) -> float:
        return self._xmin

    @property
    def xmax(self) -> float:
        return self._xmax

    @property
    def type(self) -> str:
        return self._type

    @property
    def multiplier(self) -> float:
        return self._multiplier

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @property
    def loglik(self) -> float:
        return self._loglik_val

    @property
    def edf(self) -> float:
        return self._edf_val

    @property
    def prob0(self) -> float:
        return self._prob0_val

    @property
    def nobs(self) -> int:
        return self._nobs

    @property
    def values(self):
        return self._pdf_t.clone() if self._pdf_t is not None else None

    @property
    def grid_points(self):
        return self._grid_t.clone() if self._grid_t is not None else None

    def __repr__(self) -> str:
        if not self._fitted:
            return f"Kde1d(type={self._type!r}, degree={self._degree}, fitted=False)"
        return (
            f"Kde1d(type={self._type!r}, degree={self._degree}, "
            f"bw={self._bw:.4g}, nobs={self._nobs})"
        )
