"""Bivariate copula implementation — all families, fitting, and evaluation."""

from __future__ import annotations

import math
import bisect
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from . import stats
from .families import BicopFamily, family_can_rotate, normalize_family
from .fit_controls import FitControlsBicop
from .interpolation import InterpolationGrid, make_normal_grid
from .optimize import (
    torch_maximize_1d,
    torch_maximize_1d_with_grad,
    torch_maximize_bounded,
    torch_maximize_bounded_with_grad,
    torchmin_strict_enabled,
)


_ROTATIONS = (0, 90, 180, 270)

# Mirrors vinecopulib::bicop_families::itau.
_ITAU_FAMILIES = (
    BicopFamily.indep,
    BicopFamily.gaussian,
    BicopFamily.student,
    BicopFamily.clayton,
    BicopFamily.gumbel,
    BicopFamily.frank,
    BicopFamily.joe,
)

_FAST_MLE_1P_FAMILIES = (
    BicopFamily.gaussian,
    BicopFamily.clayton,
    BicopFamily.gumbel,
    BicopFamily.frank,
    BicopFamily.joe,
)

_FRANK_TAU_GRID: list[float] | None = None
_FRANK_THETA_GRID: list[float] | None = None
_JOE_TAU_GRID: list[float] | None = None
_JOE_THETA_GRID: list[float] | None = None


def _interp_monotone(x: float, xs: list[float], ys: list[float]) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    j = bisect.bisect_left(xs, x)
    x0, x1 = xs[j - 1], xs[j]
    y0, y1 = ys[j - 1], ys[j]
    if x1 <= x0:
        return y0
    w = (x - x0) / (x1 - x0)
    return y0 + w * (y1 - y0)


def _init_frank_tau_inverse_grid() -> tuple[list[float], list[float]]:
    global _FRANK_TAU_GRID, _FRANK_THETA_GRID
    if _FRANK_TAU_GRID is not None and _FRANK_THETA_GRID is not None:
        return _FRANK_TAU_GRID, _FRANK_THETA_GRID
    thetas = [1e-4 + (40.0 - 1e-4) * i / 4095.0 for i in range(4096)]
    taus: list[float] = []
    for th in thetas:
        t = 1.0 - 4.0 / th + 4.0 * _debye1(th) / th
        taus.append(max(0.0, min(0.999999, float(t))))
    _FRANK_TAU_GRID = taus
    _FRANK_THETA_GRID = thetas
    return taus, thetas


def _init_joe_tau_inverse_grid() -> tuple[list[float], list[float]]:
    global _JOE_TAU_GRID, _JOE_THETA_GRID
    if _JOE_TAU_GRID is not None and _JOE_THETA_GRID is not None:
        return _JOE_TAU_GRID, _JOE_THETA_GRID
    thetas = [1.0 + 1e-6 + (30.0 - (1.0 + 1e-6)) * i / 4095.0 for i in range(4096)]
    dg2 = float(torch.digamma(torch.tensor(2.0)).item())
    taus: list[float] = []
    for th in thetas:
        tmp = 2.0 / th + 1.0
        dig = dg2 - float(torch.digamma(torch.tensor(tmp)).item())
        t = 1.0 + 2.0 * dig / (2.0 - th)
        taus.append(max(0.0, min(0.999999, float(t))))
    _JOE_TAU_GRID = taus
    _JOE_THETA_GRID = thetas
    return taus, thetas


def _check_rotation(rotation: int):
    if rotation not in _ROTATIONS:
        raise ValueError(f"rotation must be one of {_ROTATIONS}, got {rotation}")


def _rotate_data_like_vinecopulib(u: torch.Tensor, rotation: int) -> torch.Tensor:
    # Mirrors C++ Bicop::rotate_data (counter-clockwise rotations).
    # Supports 2 columns (continuous) or 4 columns (discrete format).
    if rotation == 0:
        return u
    if u.ndim != 2 or u.shape[1] not in (2, 4):
        raise ValueError("u must have shape (n,2) or (n,4)")
    if rotation == 90:
        if u.shape[1] == 2:
            return torch.stack([u[:, 1], 1.0 - u[:, 0]], dim=1)
        return torch.stack([u[:, 1], 1.0 - u[:, 0], u[:, 3], 1.0 - u[:, 2]], dim=1)
    if rotation == 180:
        return 1.0 - u
    if rotation == 270:
        if u.shape[1] == 2:
            return torch.stack([1.0 - u[:, 1], u[:, 0]], dim=1)
        return torch.stack([1.0 - u[:, 1], u[:, 0], 1.0 - u[:, 3], u[:, 2]], dim=1)
    _check_rotation(rotation)
    raise AssertionError("unreachable")


def _bisect_in_01_for_target_phi_prime(
    phi_prime_fn: Callable[[torch.Tensor], torch.Tensor],
    target: torch.Tensor,
    *,
    max_iter: int = 50,
    eps: float = 1e-10,
) -> torch.Tensor:
    # Solve phi'(x) = target for x in (0,1) by bisection.
    # Assumes phi' is monotone increasing in x (true for the implemented Archimedean families).
    lo = torch.full_like(target, eps)
    hi = torch.full_like(target, 1.0 - eps)
    for _ in range(max_iter):
        mid = (lo + hi) * 0.5
        val = phi_prime_fn(mid)
        lo = torch.where(val < target, mid, lo)
        hi = torch.where(val >= target, mid, hi)
    return (lo + hi) * 0.5


def _newton_raphson_phi_prime_gumbel(
    theta: torch.Tensor,
    target: torch.Tensor,
    *,
    max_iter: int = 12,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Newton-Raphson for gumbel phi'(x) = target. Quadratic convergence (8-12 iters).

    phi'(x)  = -theta * (-log x)^(theta-1) / x
    phi''(x) =  theta * (-log x)^(theta-2) * ((-log x) + theta - 1) / x^2
    Newton:  x_new = x - (phi'(x) - target) / phi''(x)
    """
    tiny = torch.finfo(target.dtype).tiny
    # Initial guess via bisection (4 steps) for safe warm start
    lo = torch.full_like(target, eps)
    hi = torch.full_like(target, 1.0 - eps)
    for _ in range(6):
        mid = (lo + hi) * 0.5
        neg_log_mid = (-torch.log(mid)).clamp_min(tiny)
        val = -theta * torch.pow(neg_log_mid, theta - 1.0) / mid
        lo = torch.where(val < target, mid, lo)
        hi = torch.where(val >= target, mid, hi)
    x = (lo + hi) * 0.5

    for _ in range(max_iter):
        neg_log_x = (-torch.log(x.clamp_min(tiny))).clamp_min(tiny)
        phi_p = -theta * torch.pow(neg_log_x, theta - 1.0) / x
        phi_pp = theta * torch.pow(neg_log_x, theta - 2.0) * (neg_log_x + theta - 1.0) / (x * x)
        residual = phi_p - target
        step = residual / phi_pp.clamp(tiny)
        x = (x - step).clamp(eps, 1.0 - eps)
    return x.clamp(eps, 1.0 - eps)


def _newton_raphson_phi_prime_joe(
    theta: torch.Tensor,
    target: torch.Tensor,
    *,
    max_iter: int = 12,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Newton-Raphson for joe phi'(x) = target. Quadratic convergence.

    phi'(x)  = -theta * (1-x)^(theta-1) / (1 - (1-x)^theta)
    phi''(x) = theta*(theta-1)*(1-x)^(theta-2)/D + theta^2*(1-x)^(2*theta-2)/D^2
    Newton:  x_new = x - (phi'(x) - target) / phi''(x)
    """
    tiny = torch.finfo(target.dtype).tiny
    # Initial guess via bisection (4 steps)
    lo = torch.full_like(target, eps)
    hi = torch.full_like(target, 1.0 - eps)
    for _ in range(6):
        mid = (lo + hi) * 0.5
        b = (1.0 - mid).clamp_min(tiny)
        bt = torch.pow(b, theta)
        D = (1.0 - bt).clamp_min(tiny)
        val = -theta * torch.pow(b, theta - 1.0) / D
        lo = torch.where(val < target, mid, lo)
        hi = torch.where(val >= target, mid, hi)
    x = (lo + hi) * 0.5

    for _ in range(max_iter):
        b = (1.0 - x).clamp_min(tiny)
        bt = torch.pow(b, theta)
        D = (1.0 - bt).clamp_min(tiny)
        phi_p = -theta * torch.pow(b, theta - 1.0) / D
        phi_pp = (theta * (theta - 1.0) * torch.pow(b, theta - 2.0) / D
                  + theta * theta * torch.pow(b, 2.0 * theta - 2.0) / (D * D))
        residual = phi_p - target
        step = residual / phi_pp.clamp(tiny)
        x = (x - step).clamp(eps, 1.0 - eps)
    return x.clamp(eps, 1.0 - eps)


# Keep bisection variants as fallbacks
def _bisect_in_01_for_target_phi_prime_gumbel(
    theta: torch.Tensor,
    target: torch.Tensor,
    *,
    max_iter: int = 40,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Specialized bisection for gumbel phi'(x) = target, inlined formula."""
    tiny = torch.finfo(target.dtype).tiny
    lo = torch.full_like(target, eps)
    hi = torch.full_like(target, 1.0 - eps)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        neg_log_mid = (-torch.log(mid)).clamp_min(tiny)
        val = -theta * torch.pow(neg_log_mid, theta - 1.0) / mid
        lo = torch.where(val < target, mid, lo)
        hi = torch.where(val >= target, mid, hi)
    return (lo + hi) * 0.5


def _bisect_in_01_for_target_phi_prime_joe(
    theta: torch.Tensor,
    target: torch.Tensor,
    *,
    max_iter: int = 40,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Specialized bisection for joe phi'(x) = target, inlined formula."""
    tiny = torch.finfo(target.dtype).tiny
    lo = torch.full_like(target, eps)
    hi = torch.full_like(target, 1.0 - eps)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        b = (1.0 - mid).clamp_min(tiny)
        bt = torch.pow(b, theta)
        D = (1.0 - bt).clamp_min(tiny)
        val = -theta * torch.pow(b, theta - 1.0) / D
        lo = torch.where(val < target, mid, lo)
        hi = torch.where(val >= target, mid, hi)
    return (lo + hi) * 0.5


def _bisect_u2_for_hfunc1(
    u1: torch.Tensor,
    w: torch.Tensor,
    *,
    hfunc1_fn: Callable[[torch.Tensor], torch.Tensor],
    max_iter: int = 50,
    eps: float = 1e-10,
) -> torch.Tensor:
    # Solve hfunc1([u1,u2]) = w for u2 in (0,1).
    w = w.clamp(eps, 1.0 - eps)
    lo = torch.full_like(w, eps)
    hi = torch.full_like(w, 1.0 - eps)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = hfunc1_fn(torch.stack([u1, mid], dim=1))
        lo = torch.where(val < w, mid, lo)
        hi = torch.where(val >= w, mid, hi)
    return 0.5 * (lo + hi)


def _bisect_u1_for_hfunc2(
    u2: torch.Tensor,
    w: torch.Tensor,
    *,
    hfunc2_fn: Callable[[torch.Tensor], torch.Tensor],
    max_iter: int = 50,
    eps: float = 1e-10,
) -> torch.Tensor:
    # Solve hfunc2([u1,u2]) = w for u1 in (0,1).
    w = w.clamp(eps, 1.0 - eps)
    lo = torch.full_like(w, eps)
    hi = torch.full_like(w, 1.0 - eps)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = hfunc2_fn(torch.stack([mid, u2], dim=1))
        lo = torch.where(val < w, mid, lo)
        hi = torch.where(val >= w, mid, hi)
    return 0.5 * (lo + hi)


def _invert_unit_interval(target: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor], *, max_iter: int = 80, eps: float = 1e-10) -> torch.Tensor:
    # Invert a monotone increasing function f on (0,1) by bisection.
    target = target.clamp(eps, 1.0 - eps)
    lo = torch.full_like(target, eps)
    hi = torch.full_like(target, 1.0 - eps)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = f(mid)
        lo = torch.where(val < target, mid, lo)
        hi = torch.where(val >= target, mid, hi)
    return 0.5 * (lo + hi)


def _winsorize_tau_like_vinecopulib(tau: float) -> float:
    # Port of ParBicop::winsorize_tau()
    sign = -1.0 if tau < 0 else 1.0
    at = abs(float(tau))
    if at < 0.01:
        at = 0.01
    elif at > 0.9:
        at = 0.9
    return sign * at


def _tau_approx_from_normal_scores(
    u1: torch.Tensor,
    u2: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> float:
    z1 = stats.qnorm(stats.clamp_unit(u1))
    z2 = stats.qnorm(stats.clamp_unit(u2))
    rho = stats.pearson_cor(z1, z2, weights=weights)
    rho = max(-0.999999, min(0.999999, float(rho)))
    return float((2.0 / math.pi) * math.asin(rho))


def _estimate_tau_for_fit(
    u1: torch.Tensor,
    u2: torch.Tensor,
    controls: FitControlsBicop,
) -> float:
    mode = str(getattr(controls, "tau_estimation", "auto"))
    w = controls.weights
    if mode == "exact":
        return stats.kendall_tau(u1, u2, weights=w)
    if mode == "approx":
        return _tau_approx_from_normal_scores(u1, u2, weights=w)
    n = int(u1.numel())
    if (w is None or w.numel() == 0) and n >= 800:
        return _tau_approx_from_normal_scores(u1, u2, weights=None)
    return stats.kendall_tau(u1, u2, weights=w)


def _debye1(x: float) -> float:
    # Port of vinecopulib's debye1() in frank.ipp (double precision).
    if x <= 0.0:
        return 0.0
    m_1_2pi = 0.159154943091895335768883763373  # 1/(2pi)
    if x >= 3.0:
        s = 1.64493406684822643647241516665
        kLim = {3: 13, 4: 10, 5: 8, 6: 7, 7: 6, 8: 5, 9: 5, 10: 4, 11: 4, 12: 4, 13: 3}
        kmax = kLim.get(int(x), 3) if x < 14.0 else 3
        for k in range(1, kmax + 1):
            xk = x * k
            ksum = 1.0 / xk
            ksum += ksum / xk
            s -= math.exp(-xk) * ksum * x * x
        return s
    # x < 3.0
    koeff = [
        0.0,
        1.289868133696452872944830333292e00,
        1.646464674222763830320073930823e-01,
        3.468612396889827942903585958184e-02,
        8.154712395888678757370477017305e-03,
        1.989150255636170674291917800638e-03,
        4.921731066160965972759960954793e-04,
        1.224962701174096585170902102707e-04,
        3.056451881730374346514297527344e-05,
        7.634586529999679712923289243879e-06,
        1.907924067745592226304077366899e-06,
        4.769010054554659800072963735060e-07,
        1.192163781025189592248804158716e-07,
        2.980310965673008246931701326140e-08,
        7.450668049576914109638408036805e-09,
        1.862654864839336365743529470042e-09,
        4.656623667353010984002911951881e-10,
        1.164154417580540177848737197821e-10,
        2.910384378208396847185926449064e-11,
        7.275959094757302380474472711747e-12,
        1.818989568052777856506623677390e-12,
        4.547473691649305030453643155957e-13,
        1.136868397525517121855436593505e-13,
        2.842170965606321353966861428348e-14,
        7.105427382674227346596939068119e-15,
        1.776356842186163180619218277278e-15,
        4.440892101596083967998640188409e-16,
        1.110223024969096248744747318102e-16,
        2.775557561945046552567818981300e-17,
        6.938893904331845249488542992219e-18,
        1.734723476023986745668411013469e-18,
        4.336808689994439570027820336642e-19,
        1.084202172491329082183740080878e-19,
        2.710505431220232916297046799365e-20,
        6.776263578041593636171406200902e-21,
        1.694065894509399669649398521836e-21,
        4.235164736272389463688418879636e-22,
        1.058791184067974064762782460584e-22,
        2.646977960169798160618902050189e-23,
        6.617444900424343177893912768629e-24,
        1.654361225106068880734221123349e-24,
        4.135903062765153408791935838694e-25,
        1.033975765691286264082026643327e-25,
        2.584939414228213340076225223666e-26,
        6.462348535570530772269628236053e-27,
        1.615587133892632406631747637268e-27,
        4.038967834731580698317525293132e-28,
        1.009741958682895139216954234507e-28,
        2.524354896707237808750799932127e-29,
        6.310887241768094478219682436680e-30,
        1.577721810442023614704107565240e-30,
        3.944304526105059031370476640000e-31,
        9.860761315262647572437533499000e-32,
        2.465190328815661892443976898000e-32,
        6.162975822039154730370601500000e-33,
        1.540743955509788682510501190000e-33,
        3.851859888774471706184973900000e-34,
        9.629649721936179265360991000000e-35,
        2.407412430484044816328953000000e-35,
        6.018531076210112040809600000000e-36,
        1.504632769052528010200750000000e-36,
        3.761581922631320025497600000000e-37,
        9.403954806578300063715000000000e-38,
        2.350988701644575015901000000000e-38,
        5.877471754111437539470000000000e-39,
        1.469367938527859384580000000000e-39,
        3.673419846319648458500000000000e-40,
        9.183549615799121117000000000000e-41,
        2.295887403949780249000000000000e-41,
        5.739718509874450320000000000000e-42,
        1.434929627468612270000000000000e-42,
    ]
    s = 0.0
    sold = 1.0
    x2pi = x * m_1_2pi
    k = 1
    while (k < 70) and (s != sold):
        sold = s
        s += (2.0 + koeff[k]) * (x2pi ** (2.0 * k)) / (2.0 * k + 1.0)
        k += 1
        s -= (2.0 + koeff[k]) * (x2pi ** (2.0 * k)) / (2.0 * k + 1.0)
        k += 1
    return x * (s + 1.0 - x / 4.0)


def _invert_monotone_1d(target: float, f: Callable[[float], float], *, lo: float, hi: float, max_iter: int = 28) -> float:
    # Bisection inversion for monotone increasing f on [lo,hi].
    a = float(lo)
    b = float(hi)
    ta = f(a)
    tb = f(b)
    if ta > tb:
        # flip if decreasing
        def f2(x: float) -> float:
            return -f(x)

        return _invert_monotone_1d(-target, f2, lo=lo, hi=hi, max_iter=max_iter)
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        tm = f(m)
        if tm < target:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def _arch_cdf(u_rot: torch.Tensor, *, phi, psi) -> torch.Tensor:
    u1 = u_rot[:, 0]
    u2 = u_rot[:, 1]
    return stats.clamp_unit(psi(phi(u1) + phi(u2)))


def _arch_hfunc1(u_rot: torch.Tensor, *, phi, psi, phi_p) -> torch.Tensor:
    u1 = u_rot[:, 0]
    u2 = u_rot[:, 1]
    C = stats.clamp_unit(psi(phi(u1) + phi(u2)))
    out = phi_p(u1) / phi_p(C)
    return torch.where(torch.isnan(out), u2, out)


def _arch_pdf(u_rot: torch.Tensor, *, phi, psi, phi_p, phi_pp) -> torch.Tensor:
    u1 = u_rot[:, 0]
    u2 = u_rot[:, 1]
    C = stats.clamp_unit(psi(phi(u1) + phi(u2)))
    tiny = torch.finfo(u_rot.dtype).tiny
    num = torch.abs(phi_pp(C)) * torch.abs(phi_p(u1)) * torch.abs(phi_p(u2))
    den = torch.pow(torch.abs(phi_p(C)).clamp_min(tiny), 3.0)
    return num / den


def _arch_hinv1(u_rot: torch.Tensor, *, phi, psi, phi_p) -> torch.Tensor:
    # Input is (u1, w) and output u2.
    u1 = u_rot[:, 0]
    w = u_rot[:, 1].clamp(1e-10, 1.0 - 1e-10)
    target = phi_p(u1) / w
    C = stats.clamp_unit(_bisect_in_01_for_target_phi_prime(phi_p, target))
    phi_u2 = (phi(C) - phi(u1)).clamp_min(torch.finfo(u_rot.dtype).tiny)
    return stats.clamp_unit(psi(phi_u2))


@dataclass
class Bicop:
    family: BicopFamily = BicopFamily.indep
    rotation: int = 0
    parameters: torch.Tensor | None = None
    var_types: tuple[str, str] = ("c", "c")
    nobs: int = 0
    _interp_grid: InterpolationGrid | None = field(default=None, init=False, repr=False)
    _fit_loglik: float | None = field(default=None, init=False, repr=False)

    def to(self, *args, **kwargs) -> "Bicop":
        """Move parameters and interpolation grid to device/dtype (in-place)."""
        if self.parameters is not None:
            self.parameters = self.parameters.to(*args, **kwargs)
        if self._interp_grid is not None:
            self._interp_grid = self._interp_grid.to(*args, **kwargs)
        return self

    def __post_init__(self):
        self.family = normalize_family(self.family)
        _check_rotation(int(self.rotation))
        if not family_can_rotate(self.family) and int(self.rotation) != 0:
            raise ValueError(f"family {self.family} does not support rotation")
        if self.parameters is None:
            if self.family == BicopFamily.tll:
                self.parameters = torch.ones((30, 30))
            else:
                self.parameters = torch.empty((0,))

    def to_json(self) -> dict[str, Any]:
        return {
            "family": self.family.value,
            "rotation": int(self.rotation),
            "parameters": self.parameters.detach().cpu().tolist(),
            "var_types": list(self.var_types),
            "nobs": int(self.nobs),
        }

    def str(self) -> str:
        """Human-readable string representation (mirrors pyvinecopulib)."""
        p = torch.as_tensor(self.parameters).reshape(-1)
        pstr = ", ".join(f"{v:.4f}" for v in p.tolist()) if p.numel() > 0 and self.family != BicopFamily.tll else ""
        parts = [f"<torchvine.Bicop>"]
        parts.append(f"  family: {self.family.value}")
        if int(self.rotation) != 0:
            parts.append(f"  rotation: {self.rotation}")
        if pstr:
            parts.append(f"  parameters: [{pstr}]")
        parts.append(f"  var_types: {list(self.var_types)}")
        if self.nobs > 0:
            parts.append(f"  nobs: {self.nobs}")
        return "\n".join(parts)

    @staticmethod
    def from_json(obj: dict[str, Any]) -> "Bicop":
        return Bicop(
            family=normalize_family(obj["family"]),
            rotation=int(obj.get("rotation", 0)),
            parameters=torch.as_tensor(obj.get("parameters", [])),
            var_types=tuple(obj.get("var_types", ["c", "c"])),
            nobs=int(obj.get("nobs", 0)),
        )

    @classmethod
    def from_family(
        cls,
        family: str | BicopFamily,
        *,
        rotation: int = 0,
        parameters: torch.Tensor | None = None,
        var_types: tuple[str, str] = ("c", "c"),
    ) -> "Bicop":
        return cls(family=normalize_family(family), rotation=int(rotation), parameters=parameters, var_types=var_types)

    @classmethod
    def from_data(cls, data: torch.Tensor, controls: FitControlsBicop | None = None) -> "Bicop":
        # pyvinecopulib-style convenience constructor: build+select on data.
        c = cls()
        c.select(data, controls=controls)
        return c

    @classmethod
    def from_file(cls, path: str) -> "Bicop":
        import json

        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return cls.from_json(obj)

    def to_file(self, path: str) -> None:
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2, sort_keys=True)

    @property
    def npars(self) -> float:
        return float(self.get_npars())

    @property
    def tau(self) -> float:
        # Kendall's tau implied by current parameters.
        return float(self.parameters_to_tau())

    @property
    def parameters_lower_bounds(self) -> torch.Tensor:
        lb, _ub = self._get_parameter_bounds()
        return lb

    @property
    def parameters_upper_bounds(self) -> torch.Tensor:
        _lb, ub = self._get_parameter_bounds()
        return ub

    def plot(self, *args, **kwargs):
        import matplotlib.pyplot as plt

        n = int(kwargs.pop("n", 80))
        device = kwargs.pop("device", "cpu")
        dtype = kwargs.pop("dtype", self.parameters.dtype if self.parameters is not None else torch.float64)
        title = kwargs.pop("title", f"{self.family.value} (rot={int(self.rotation)})")

        u1 = torch.linspace(1e-3, 1.0 - 1e-3, n, device=device, dtype=dtype)
        u2 = torch.linspace(1e-3, 1.0 - 1e-3, n, device=device, dtype=dtype)
        U1, U2 = torch.meshgrid(u1, u2, indexing="ij")
        grid = torch.stack([U1.reshape(-1), U2.reshape(-1)], dim=1)
        z = self.pdf(grid).reshape(n, n).detach().cpu().tolist()

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = ax.imshow(z, origin="lower", extent=[0, 1, 0, 1], aspect="auto")
        ax.set_xlabel("u1")
        ax.set_ylabel("u2")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="pdf")
        return fig

    def _check_var_types(self) -> None:
        if len(self.var_types) != 2:
            raise ValueError("var_types must have length 2")
        for t in self.var_types:
            if t not in ("c", "d"):
                raise ValueError("var type must be 'c' or 'd'")

    def _n_discrete(self) -> int:
        self._check_var_types()
        return int(self.var_types[0] == "d") + int(self.var_types[1] == "d")

    def _format_data(self, u: torch.Tensor) -> torch.Tensor:
        # Port of C++ Bicop::format_data.
        u = torch.as_tensor(u)
        if u.ndim != 2:
            raise ValueError("u must be 2D")

        n_disc = self._n_discrete()
        if n_disc == 0:
            if u.shape[1] < 2:
                raise ValueError("u must have at least 2 columns")
            return u[:, :2]
        if n_disc == 2:
            if u.shape[1] != 4:
                raise ValueError("for var_types=('d','d'), u must have 4 columns")
            return u

        # n_disc == 1
        if u.shape[1] not in (3, 4):
            raise ValueError("for one discrete variable, u must have 3 or 4 columns")
        u_new = torch.empty((u.shape[0], 4), device=u.device, dtype=u.dtype)
        u_new[:, :2] = u[:, :2]
        disc_col = 1 if self.var_types[1] == "d" else 0
        cont_col = 1 - disc_col
        old_disc_col = 2 + (1 if u.shape[1] == 4 else 0) * disc_col
        u_new[:, 2 + disc_col] = u[:, old_disc_col]
        u_new[:, 2 + cont_col] = u[:, cont_col]
        return u_new

    def _prep_for_abstract(self, u: torch.Tensor) -> torch.Tensor:
        # Equivalent to C++ prep_for_abstract(): format + trim + rotate.
        u = self._format_data(u)
        u = stats.clamp_unit(torch.as_tensor(u))
        return _rotate_data_like_vinecopulib(u, int(self.rotation))

    def as_continuous(self) -> "Bicop":
        # Discrete handling isn't implemented; keep API parity.
        return Bicop(family=self.family, rotation=int(self.rotation), parameters=self.parameters, var_types=("c", "c"))

    def _get_interp_grid(self, *, device, dtype) -> InterpolationGrid:
        # For BicopFamily.tll
        if self._interp_grid is None or self._interp_grid.values.device != device or self._interp_grid.values.dtype != dtype:
            grid = make_normal_grid(30, boundary_to_01=True, device=device, dtype=dtype)
            vals = torch.as_tensor(self.parameters, device=device, dtype=dtype)
            if vals.ndim != 2 or vals.shape[0] != vals.shape[1]:
                raise ValueError("tll parameters must be a square (m,m) grid")
            self._interp_grid = InterpolationGrid(grid, vals)
        return self._interp_grid

    def _arch_parts(self):
        # Return (phi, psi, phi_p, phi_pp) for Archimedean families.
        dt = self.parameters
        assert dt is not None

        if self.family == BicopFamily.gumbel:
            theta = torch.as_tensor(dt[..., 0])
            if torch.any(theta < 1):
                raise ValueError("gumbel parameter theta must be >= 1")

            phi = lambda x: torch.pow((-torch.log(x)).clamp_min(torch.finfo(x.dtype).tiny), theta)
            psi = lambda t: torch.exp(-torch.pow(t.clamp_min(torch.finfo(theta.dtype).tiny), 1.0 / theta))
            phi_p = lambda x: -theta * torch.pow((-torch.log(x)).clamp_min(torch.finfo(x.dtype).tiny), theta - 1.0) / x
            phi_pp = lambda x: theta * torch.pow((-torch.log(x)).clamp_min(torch.finfo(x.dtype).tiny), theta - 2.0) * (((-torch.log(x)).clamp_min(torch.finfo(x.dtype).tiny)) + theta - 1.0) / (x * x)
            return phi, psi, phi_p, phi_pp

        if self.family == BicopFamily.joe:
            theta = torch.as_tensor(dt[..., 0])
            if torch.any(theta < 1):
                raise ValueError("joe parameter theta must be >= 1")

            phi = lambda x: -torch.log((1.0 - torch.pow((1.0 - x).clamp_min(torch.finfo(x.dtype).tiny), theta)).clamp_min(torch.finfo(x.dtype).tiny))
            psi = lambda t: 1.0 - torch.pow((1.0 - torch.exp(-t)).clamp_min(torch.finfo(theta.dtype).tiny), 1.0 / theta)
            phi_p = lambda x: -theta * torch.pow((1.0 - x).clamp_min(torch.finfo(x.dtype).tiny), theta - 1.0) / (1.0 - torch.pow((1.0 - x).clamp_min(torch.finfo(x.dtype).tiny), theta)).clamp_min(torch.finfo(x.dtype).tiny)

            def phi_pp(x):
                b = (1.0 - x).clamp_min(torch.finfo(x.dtype).tiny)
                D = (1.0 - torch.pow(b, theta)).clamp_min(torch.finfo(x.dtype).tiny)
                term1 = theta * (theta - 1.0) * torch.pow(b, theta - 2.0) / D
                term2 = (theta * theta) * torch.pow(b, 2.0 * theta - 2.0) / (D * D)
                return term1 + term2

            return phi, psi, phi_p, phi_pp

        if self.family == BicopFamily.bb1:
            theta = torch.as_tensor(dt[..., 0])
            delta = torch.as_tensor(dt[..., 1])
            if torch.any(theta <= 0) or torch.any(delta < 1):
                raise ValueError("bb1 parameters must satisfy theta > 0 and delta >= 1")

            phi = lambda x: torch.pow(torch.pow(x, -theta) - 1.0, delta)
            psi = lambda t: torch.pow(torch.pow(t.clamp_min(torch.finfo(theta.dtype).tiny), 1.0 / delta) + 1.0, -1.0 / theta)
            phi_p = lambda x: (-delta * theta) * torch.pow(x, -(1.0 + theta)) * torch.pow(torch.pow(x, -theta) - 1.0, delta - 1.0)

            def phi_pp(x):
                # From vinecopulib comment in bb1.ipp
                t1 = torch.pow(x, -theta) - 1.0
                num = delta * theta * torch.pow(t1, delta)
                den = torch.pow(torch.pow(x, theta) - 1.0, 2.0) * (x * x)
                core = (1.0 + delta * theta) - (1.0 + theta) * torch.pow(x, theta)
                return num * core / den.clamp_min(torch.finfo(x.dtype).tiny)

            return phi, psi, phi_p, phi_pp

        if self.family == BicopFamily.bb6:
            theta = torch.as_tensor(dt[..., 0])
            delta = torch.as_tensor(dt[..., 1])
            if torch.any(theta < 1) or torch.any(delta < 1):
                raise ValueError("bb6 parameters must satisfy theta >= 1 and delta >= 1")

            tiny = torch.finfo(theta.dtype).tiny

            phi = lambda x: torch.pow((-torch.log1p(-torch.pow((1.0 - x).clamp_min(tiny), theta))).clamp_min(tiny), delta)
            psi = lambda t: 1.0 - torch.pow((-torch.expm1(-torch.pow(t.clamp_min(tiny), 1.0 / delta))).clamp_min(tiny), 1.0 / theta)

            def phi_p(x):
                tmp = torch.pow((1.0 - x).clamp_min(tiny), theta)
                lg = torch.log1p(-tmp)
                num = delta * theta * torch.pow((-lg).clamp_min(tiny), delta - 1.0) * torch.pow((1.0 - x).clamp_min(tiny), theta - 1.0)
                # (tmp - 1) is negative
                return num / (tmp - 1.0).clamp_max(-tiny)

            def phi_pp(x):
                tmp = torch.pow((1.0 - x).clamp_min(tiny), theta)
                tmp2 = torch.log1p(-tmp)  # negative
                res = torch.pow((-tmp2).clamp_min(tiny), delta - 2.0)
                core = (delta - 1.0) * theta * tmp - (tmp + theta - 1.0) * tmp2
                return res * delta * theta * torch.pow((1.0 - x).clamp_min(tiny), theta - 2.0) * core / torch.pow((tmp - 1.0).clamp_max(-tiny), 2.0)

            return phi, psi, phi_p, phi_pp

        if self.family == BicopFamily.bb7:
            theta = torch.as_tensor(dt[..., 0])
            delta = torch.as_tensor(dt[..., 1])
            if torch.any(theta < 1) or torch.any(delta <= 0):
                raise ValueError("bb7 parameters must satisfy theta >= 1 and delta > 0")

            tiny = torch.finfo(theta.dtype).tiny

            phi = lambda x: torch.pow(1.0 - torch.pow((1.0 - x).clamp_min(tiny), theta), -delta) - 1.0
            psi = lambda t: 1.0 - torch.pow(1.0 - torch.pow(1.0 + t, -1.0 / delta), 1.0 / theta)
            phi_p = lambda x: -(delta * theta) * torch.pow(1.0 - torch.pow((1.0 - x).clamp_min(tiny), theta), -1.0 - delta) * torch.pow((1.0 - x).clamp_min(tiny), theta - 1.0)

            def phi_pp(x):
                tmp = torch.pow((1.0 - x).clamp_min(tiny), theta)
                res = delta * theta * torch.pow((1.0 - tmp).clamp_min(tiny), -2.0 - delta) * torch.pow((1.0 - x).clamp_min(tiny), theta - 2.0)
                return res * (theta - 1.0 + (1.0 + delta * theta) * tmp)

            return phi, psi, phi_p, phi_pp

        if self.family == BicopFamily.bb8:
            theta = torch.as_tensor(dt[..., 0])
            delta = torch.as_tensor(dt[..., 1])
            if torch.any(theta < 1) or torch.any(delta <= 0) or torch.any(delta > 1):
                raise ValueError("bb8 parameters must satisfy theta >= 1 and 0 < delta <= 1")

            tiny = torch.finfo(theta.dtype).tiny
            norm = (1.0 - torch.pow((1.0 - delta).clamp_min(tiny), theta)).clamp_min(tiny)

            phi = lambda x: -torch.log(((1.0 - torch.pow((1.0 - delta * x).clamp_min(tiny), theta)).clamp_min(tiny)) / norm)

            def psi(t):
                res = torch.exp(-t) * (torch.pow((1.0 - delta).clamp_min(tiny), theta) - 1.0)
                return (1.0 - torch.pow((1.0 + res).clamp_min(tiny), 1.0 / theta)) / delta

            def phi_p(x):
                tmp = torch.pow((1.0 - delta * x).clamp_min(tiny), theta)
                num = delta * theta * torch.pow((1.0 - delta * x).clamp_min(tiny), theta - 1.0)
                return -num / (1.0 - tmp).clamp_min(tiny)

            def phi_pp(x):
                tmp = torch.pow((1.0 - delta * x).clamp_min(tiny), theta)
                res = (delta * delta) * theta * torch.pow((1.0 - delta * x).clamp_min(tiny), theta - 2.0)
                return res * (theta - 1.0 + tmp) / torch.pow((tmp - 1.0).clamp_max(-tiny), 2.0)

            return phi, psi, phi_p, phi_pp

        return None

    # ---- rotationless family implementations (operate on already-rotated u) ----

    def _pdf0(self, u_rot: torch.Tensor) -> torch.Tensor:
        if self.family == BicopFamily.indep:
            # Use (u*0 + 1) instead of torch.ones to preserve autograd graph.
            return u_rot[:, 0] * 0.0 + 1.0

        if self.family == BicopFamily.tll:
            g = self._get_interp_grid(device=u_rot.device, dtype=u_rot.dtype)
            return g.interpolate(u_rot).clamp_min(1e-20)

        if self.family == BicopFamily.gaussian:
            rho = torch.as_tensor(self.parameters[..., 0], device=u_rot.device, dtype=u_rot.dtype).clamp(-0.999999, 0.999999)
            x = stats.qnorm(u_rot)
            x1 = x[:, 0]
            x2 = x[:, 1]
            den = torch.sqrt(1.0 - rho * rho)
            expo = -0.5 * (x1 * x1 - 2.0 * rho * x1 * x2 + x2 * x2) / (1.0 - rho * rho)
            expo = expo + 0.5 * (x1 * x1 + x2 * x2)
            return torch.exp(expo) / den

        if self.family == BicopFamily.student:
            rho = torch.as_tensor(self.parameters[..., 0], device=u_rot.device, dtype=u_rot.dtype).clamp(-0.999999, 0.999999)
            nu = torch.as_tensor(self.parameters[..., 1], device=u_rot.device, dtype=u_rot.dtype).clamp(2.01, 100.0)
            # Fast Hill approximation is substantially faster than iterative qt().
            u_cat = torch.cat([u_rot[:, 0], u_rot[:, 1]])
            x_cat = stats._qt_hill(u_cat, nu)
            n = u_rot.shape[0]
            x1, x2 = x_cat[:n], x_cat[n:]
            # c(u1,u2) = f2(x1,x2;rho,nu) / (f_nu(x1)*f_nu(x2))
            log_c = (torch.lgamma((nu + 2.0) * 0.5) + torch.lgamma(nu * 0.5)
                     - 2.0 * torch.lgamma((nu + 1.0) * 0.5)
                     - 0.5 * torch.log((1.0 - rho * rho).clamp_min(1e-20)))
            log_c = log_c + (nu + 1.0) * 0.5 * (torch.log1p(x1 * x1 / nu) + torch.log1p(x2 * x2 / nu))
            Q = (x1 * x1 - 2.0 * rho * x1 * x2 + x2 * x2) / (nu * (1.0 - rho * rho).clamp_min(1e-20))
            log_c = log_c - (nu + 2.0) * 0.5 * torch.log1p(Q)
            return torch.exp(log_c)

        if self.family == BicopFamily.frank:
            theta = torch.as_tensor(self.parameters[..., 0], device=u_rot.device, dtype=u_rot.dtype)
            if torch.any(theta == 0):
                return torch.ones((u_rot.shape[0],), device=u_rot.device, dtype=u_rot.dtype)
            u1 = u_rot[:, 0]
            u2 = u_rot[:, 1]
            eu = torch.expm1(-theta * u1)
            ev = torch.expm1(-theta * u2)
            ed = torch.expm1(-theta)
            A = eu + 1.0
            B = ev + 1.0
            denom = (ed + eu * ev)
            num = (-theta) * ed * A * B
            return num / (denom * denom)

        if self.family == BicopFamily.clayton:
            theta = torch.as_tensor(self.parameters[..., 0], device=u_rot.device, dtype=u_rot.dtype)
            if torch.any(theta <= 0):
                raise ValueError("clayton parameter theta must be > 0")
            u1 = u_rot[:, 0]
            u2 = u_rot[:, 1]
            t1 = torch.pow(u1, -theta)
            t2 = torch.pow(u2, -theta)
            s = (t1 + t2 - 1.0).clamp_min(torch.finfo(u_rot.dtype).tiny)
            logc = torch.log1p(theta)
            logc = logc + (-1.0 - theta) * (torch.log(u1) + torch.log(u2))
            logc = logc + (-2.0 - 1.0 / theta) * torch.log(s)
            return torch.exp(logc)

        if self.family == BicopFamily.tawn:
            # Extreme value copula using Pickands function (see extreme_value.ipp).
            psi1 = torch.as_tensor(self.parameters[..., 0], device=u_rot.device, dtype=u_rot.dtype).clamp(0.0, 1.0)
            psi2 = torch.as_tensor(self.parameters[..., 1], device=u_rot.device, dtype=u_rot.dtype).clamp(0.0, 1.0)
            theta = torch.as_tensor(self.parameters[..., 2], device=u_rot.device, dtype=u_rot.dtype).clamp(1.0, 1e9)

            tiny = torch.finfo(u_rot.dtype).tiny
            u1 = u_rot[:, 0]
            u2 = u_rot[:, 1]
            L = torch.log((u1 * u2).clamp_min(tiny))
            t = (torch.log(u2.clamp_min(tiny)) / L).clamp(0.0, 1.0)

            tmp = torch.pow((psi2 * t).clamp_min(tiny), theta) + torch.pow((psi1 * (1.0 - t)).clamp_min(tiny), theta)
            A = (1.0 - psi1) * (1.0 - t) + (1.0 - psi2) * t + torch.pow(tmp.clamp_min(tiny), 1.0 / theta)

            tmp2 = psi2 * torch.pow((psi2 * t).clamp_min(tiny), theta - 1.0) - psi1 * torch.pow((psi1 * (1.0 - t)).clamp_min(tiny), theta - 1.0)
            A1 = psi1 - psi2 + torch.pow(tmp.clamp_min(tiny), 1.0 / theta - 1.0) * tmp2

            tmp3 = (psi2 * psi2) * torch.pow((psi2 * t).clamp_min(tiny), theta - 2.0) + (psi1 * psi1) * torch.pow((psi1 * (1.0 - t)).clamp_min(tiny), theta - 2.0)
            A2 = (1.0 - theta) * torch.pow(tmp.clamp_min(tiny), 1.0 / theta - 2.0) * (tmp2 * tmp2) + torch.pow(tmp.clamp_min(tiny), 1.0 / theta - 1.0) * (theta - 1.0) * tmp3

            t3 = A * A + (1.0 - 2.0 * t) * A1 * A - (1.0 - t) * t * ((A1 * A1) + A2 / L)
            expo = (torch.log(u1.clamp_min(tiny)) + torch.log(u2.clamp_min(tiny))) * A
            return torch.exp(expo) * t3 / (u1 * u2).clamp_min(tiny)

        if self.family == BicopFamily.gumbel:
            theta = torch.as_tensor(self.parameters[..., 0], device=u_rot.device, dtype=u_rot.dtype)
            tiny = torch.finfo(u_rot.dtype).tiny
            u1, u2 = u_rot[:, 0], u_rot[:, 1]
            neg_log_u1 = (-torch.log(u1)).clamp_min(tiny)
            neg_log_u2 = (-torch.log(u2)).clamp_min(tiny)
            t1 = torch.pow(neg_log_u1, theta)
            t2 = torch.pow(neg_log_u2, theta)
            s = t1 + t2
            C = torch.exp(-torch.pow(s.clamp_min(tiny), 1.0 / theta))
            neg_log_C = (-torch.log(C)).clamp_min(tiny)
            phi_p_u1 = -theta * torch.pow(neg_log_u1, theta - 1.0) / u1
            phi_p_u2 = -theta * torch.pow(neg_log_u2, theta - 1.0) / u2
            phi_p_C = -theta * torch.pow(neg_log_C, theta - 1.0) / C
            phi_pp_C = theta * torch.pow(neg_log_C, theta - 2.0) * (neg_log_C + theta - 1.0) / (C * C)
            num = torch.abs(phi_pp_C) * torch.abs(phi_p_u1) * torch.abs(phi_p_u2)
            den = torch.pow(torch.abs(phi_p_C).clamp_min(tiny), 3.0)
            return num / den

        if self.family == BicopFamily.joe:
            theta = torch.as_tensor(self.parameters[..., 0], device=u_rot.device, dtype=u_rot.dtype)
            tiny = torch.finfo(u_rot.dtype).tiny
            u1, u2 = u_rot[:, 0], u_rot[:, 1]
            b1 = (1.0 - u1).clamp_min(tiny)
            b2 = (1.0 - u2).clamp_min(tiny)
            b1t = torch.pow(b1, theta)
            b2t = torch.pow(b2, theta)
            D1 = (1.0 - b1t).clamp_min(tiny)
            D2 = (1.0 - b2t).clamp_min(tiny)
            phi_u1 = -torch.log(D1.clamp_min(tiny))
            phi_u2 = -torch.log(D2.clamp_min(tiny))
            s = phi_u1 + phi_u2
            et = torch.exp(-s)
            C = (1.0 - torch.pow((1.0 - et).clamp_min(tiny), 1.0 / theta))
            bC = (1.0 - C).clamp_min(tiny)
            bCt = torch.pow(bC, theta)
            DC = (1.0 - bCt).clamp_min(tiny)
            phi_p_u1 = -theta * torch.pow(b1, theta - 1.0) / D1
            phi_p_u2 = -theta * torch.pow(b2, theta - 1.0) / D2
            phi_p_C = -theta * torch.pow(bC, theta - 1.0) / DC
            phi_pp_C_term1 = theta * (theta - 1.0) * torch.pow(bC, theta - 2.0) / DC
            phi_pp_C_term2 = (theta * theta) * torch.pow(bC, 2.0 * theta - 2.0) / (DC * DC)
            phi_pp_C = phi_pp_C_term1 + phi_pp_C_term2
            num = torch.abs(phi_pp_C) * torch.abs(phi_p_u1) * torch.abs(phi_p_u2)
            den = torch.pow(torch.abs(phi_p_C).clamp_min(tiny), 3.0)
            return num / den

        parts = self._arch_parts()
        if parts is not None:
            phi, psi, phi_p, phi_pp = parts
            return _arch_pdf(u_rot, phi=phi, psi=psi, phi_p=phi_p, phi_pp=phi_pp)

        raise NotImplementedError(f"pdf not implemented for family={self.family}")

    def _cdf0(self, u_rot: torch.Tensor) -> torch.Tensor:
        if self.family == BicopFamily.indep:
            return u_rot[:, 0] * u_rot[:, 1]

        if self.family == BicopFamily.tll:
            g = self._get_interp_grid(device=u_rot.device, dtype=u_rot.dtype)
            return g.integrate_2d(u_rot)

        if self.family == BicopFamily.gaussian:
            rho = torch.as_tensor(self.parameters[..., 0], device=u_rot.device, dtype=u_rot.dtype).clamp(-0.999999, 0.999999)
            z = stats.qnorm(u_rot)
            return stats.pbvnorm_drezner(z[:, 0], z[:, 1], rho)

        if self.family == BicopFamily.student:
            rho = torch.as_tensor(self.parameters[..., 0], device=u_rot.device, dtype=u_rot.dtype).clamp(-0.999999, 0.999999)
            nu = torch.as_tensor(self.parameters[..., 1], device=u_rot.device, dtype=u_rot.dtype).clamp(2.01, 100.0)
            u_cat = torch.cat([u_rot[:, 0], u_rot[:, 1]])
            x_cat = stats._qt_hill(u_cat, nu)
            n = u_rot.shape[0]
            x1, x2 = x_cat[:n], x_cat[n:]
            return stats.pbvt(x1, x2, rho, nu)

        if self.family == BicopFamily.frank:
            theta = torch.as_tensor(self.parameters[..., 0], device=u_rot.device, dtype=u_rot.dtype)
            if torch.any(theta == 0):
                return u_rot[:, 0] * u_rot[:, 1]
            u1 = u_rot[:, 0]
            u2 = u_rot[:, 1]
            eu = torch.expm1(-theta * u1)
            ev = torch.expm1(-theta * u2)
            ed = torch.expm1(-theta)
            q = 1.0 + (eu * ev) / ed
            return (-1.0 / theta) * torch.log(q)

        if self.family == BicopFamily.clayton:
            theta = torch.as_tensor(self.parameters[..., 0], device=u_rot.device, dtype=u_rot.dtype)
            if torch.any(theta <= 0):
                raise ValueError("clayton parameter theta must be > 0")
            u1 = u_rot[:, 0]
            u2 = u_rot[:, 1]
            s = (torch.pow(u1, -theta) + torch.pow(u2, -theta) - 1.0).clamp_min(torch.finfo(u_rot.dtype).tiny)
            return torch.pow(s, -1.0 / theta)

        if self.family == BicopFamily.tawn:
            psi1 = torch.as_tensor(self.parameters[..., 0], device=u_rot.device, dtype=u_rot.dtype).clamp(0.0, 1.0)
            psi2 = torch.as_tensor(self.parameters[..., 1], device=u_rot.device, dtype=u_rot.dtype).clamp(0.0, 1.0)
            theta = torch.as_tensor(self.parameters[..., 2], device=u_rot.device, dtype=u_rot.dtype).clamp(1.0, 1e9)

            tiny = torch.finfo(u_rot.dtype).tiny
            u1 = u_rot[:, 0]
            u2 = u_rot[:, 1]
            L = torch.log((u1 * u2).clamp_min(tiny))
            t = (torch.log(u2.clamp_min(tiny)) / L).clamp(0.0, 1.0)
            tmp = torch.pow((psi2 * t).clamp_min(tiny), theta) + torch.pow((psi1 * (1.0 - t)).clamp_min(tiny), theta)
            A = (1.0 - psi1) * (1.0 - t) + (1.0 - psi2) * t + torch.pow(tmp.clamp_min(tiny), 1.0 / theta)
            expo = (torch.log(u1.clamp_min(tiny)) + torch.log(u2.clamp_min(tiny))) * A
            return torch.exp(expo)

        parts = self._arch_parts()
        if parts is not None:
            phi, psi, _, _ = parts
            return _arch_cdf(u_rot, phi=phi, psi=psi)

        raise NotImplementedError(f"cdf not implemented for family={self.family}")

    def _hfunc1_0(self, u_rot: torch.Tensor) -> torch.Tensor:
        # h1(u1,u2) = P(U2<=u2 | U1=u1)
        if self.family == BicopFamily.indep:
            return u_rot[:, 1]

        if self.family == BicopFamily.tll:
            g = self._get_interp_grid(device=u_rot.device, dtype=u_rot.dtype)
            return g.integrate_1d(u_rot, 1)

        if self.family == BicopFamily.gaussian:
            rho = torch.as_tensor(self.parameters[..., 0], device=u_rot.device, dtype=u_rot.dtype).clamp(-0.999999, 0.999999)
            x = stats.qnorm(u_rot)
            h = (x[:, 1] - rho * x[:, 0]) / torch.sqrt(1.0 - rho * rho)
            return stats.pnorm(h)

        if self.family == BicopFamily.student:
            rho = torch.as_tensor(self.parameters[..., 0], device=u_rot.device, dtype=u_rot.dtype).clamp(-0.999999, 0.999999)
            nu = torch.as_tensor(self.parameters[..., 1], device=u_rot.device, dtype=u_rot.dtype).clamp(2.01, 100.0)
            u_cat = torch.cat([u_rot[:, 0], u_rot[:, 1]])
            x_cat = stats._qt_hill(u_cat, nu)
            n = u_rot.shape[0]
            x1, x2 = x_cat[:n], x_cat[n:]
            arg = (x2 - rho * x1) / torch.sqrt((nu + x1 * x1) * (1.0 - rho * rho) / (nu + 1.0))
            return stats.pt(arg, nu + 1.0)

        if self.family == BicopFamily.frank:
            theta = torch.as_tensor(self.parameters[..., 0], device=u_rot.device, dtype=u_rot.dtype)
            if torch.any(theta == 0):
                return u_rot[:, 1]
            u1 = u_rot[:, 0]
            u2 = u_rot[:, 1]
            eu = torch.expm1(-theta * u1)
            ev = torch.expm1(-theta * u2)
            ed = torch.expm1(-theta)
            A = eu + 1.0
            denom = (ed + eu * ev)
            return A * ev / denom

        if self.family == BicopFamily.clayton:
            theta = torch.as_tensor(self.parameters[..., 0], device=u_rot.device, dtype=u_rot.dtype)
            if torch.any(theta <= 0):
                raise ValueError("clayton parameter theta must be > 0")
            u1 = u_rot[:, 0]
            u2 = u_rot[:, 1]
            s = (torch.pow(u1, -theta) + torch.pow(u2, -theta) - 1.0).clamp_min(torch.finfo(u_rot.dtype).tiny)
            return torch.pow(u1, -theta - 1.0) * torch.pow(s, -1.0 / theta - 1.0)

        if self.family == BicopFamily.tawn:
            psi1 = torch.as_tensor(self.parameters[..., 0], device=u_rot.device, dtype=u_rot.dtype).clamp(0.0, 1.0)
            psi2 = torch.as_tensor(self.parameters[..., 1], device=u_rot.device, dtype=u_rot.dtype).clamp(0.0, 1.0)
            theta = torch.as_tensor(self.parameters[..., 2], device=u_rot.device, dtype=u_rot.dtype).clamp(1.0, 1e9)

            tiny = torch.finfo(u_rot.dtype).tiny
            u1 = u_rot[:, 0]
            u2 = u_rot[:, 1]
            L = torch.log((u1 * u2).clamp_min(tiny))
            t = (torch.log(u2.clamp_min(tiny)) / L).clamp(0.0, 1.0)

            tmp = torch.pow((psi2 * t).clamp_min(tiny), theta) + torch.pow((psi1 * (1.0 - t)).clamp_min(tiny), theta)
            A = (1.0 - psi1) * (1.0 - t) + (1.0 - psi2) * t + torch.pow(tmp.clamp_min(tiny), 1.0 / theta)
            tmp2 = psi2 * torch.pow((psi2 * t).clamp_min(tiny), theta - 1.0) - psi1 * torch.pow((psi1 * (1.0 - t)).clamp_min(tiny), theta - 1.0)
            A1 = psi1 - psi2 + torch.pow(tmp.clamp_min(tiny), 1.0 / theta - 1.0) * tmp2

            t3 = A - t * A1
            expo = (torch.log(u1.clamp_min(tiny)) + torch.log(u2.clamp_min(tiny))) * A
            return torch.exp(expo) * t3 / u1.clamp_min(tiny)

        if self.family == BicopFamily.gumbel:
            theta = torch.as_tensor(self.parameters[..., 0], device=u_rot.device, dtype=u_rot.dtype)
            tiny = torch.finfo(u_rot.dtype).tiny
            u1, u2 = u_rot[:, 0], u_rot[:, 1]
            neg_log_u1 = (-torch.log(u1)).clamp_min(tiny)
            neg_log_u2 = (-torch.log(u2)).clamp_min(tiny)
            t1 = torch.pow(neg_log_u1, theta)
            t2 = torch.pow(neg_log_u2, theta)
            s = t1 + t2
            C = stats.clamp_unit(torch.exp(-torch.pow(s.clamp_min(tiny), 1.0 / theta)))
            phi_p_u1 = -theta * torch.pow(neg_log_u1, theta - 1.0) / u1
            neg_log_C = (-torch.log(C)).clamp_min(tiny)
            phi_p_C = -theta * torch.pow(neg_log_C, theta - 1.0) / C
            out = phi_p_u1 / phi_p_C
            return torch.where(torch.isnan(out), u2, out)

        if self.family == BicopFamily.joe:
            theta = torch.as_tensor(self.parameters[..., 0], device=u_rot.device, dtype=u_rot.dtype)
            tiny = torch.finfo(u_rot.dtype).tiny
            u1, u2 = u_rot[:, 0], u_rot[:, 1]
            b1 = (1.0 - u1).clamp_min(tiny)
            b2 = (1.0 - u2).clamp_min(tiny)
            b1t = torch.pow(b1, theta)
            b2t = torch.pow(b2, theta)
            D1 = (1.0 - b1t).clamp_min(tiny)
            D2 = (1.0 - b2t).clamp_min(tiny)
            phi_u1 = -torch.log(D1.clamp_min(tiny))
            phi_u2 = -torch.log(D2.clamp_min(tiny))
            s = phi_u1 + phi_u2
            et = torch.exp(-s)
            C = stats.clamp_unit(1.0 - torch.pow((1.0 - et).clamp_min(tiny), 1.0 / theta))
            bC = (1.0 - C).clamp_min(tiny)
            bCt = torch.pow(bC, theta)
            DC = (1.0 - bCt).clamp_min(tiny)
            phi_p_u1 = -theta * torch.pow(b1, theta - 1.0) / D1
            phi_p_C = -theta * torch.pow(bC, theta - 1.0) / DC
            out = phi_p_u1 / phi_p_C
            return torch.where(torch.isnan(out), u2, out)

        parts = self._arch_parts()
        if parts is not None:
            phi, psi, phi_p, _ = parts
            return _arch_hfunc1(u_rot, phi=phi, psi=psi, phi_p=phi_p)

        raise NotImplementedError(f"hfunc1 not implemented for family={self.family}")

    def _hfunc2_0(self, u_rot: torch.Tensor) -> torch.Tensor:
        if self.family == BicopFamily.tll:
            g = self._get_interp_grid(device=u_rot.device, dtype=u_rot.dtype)
            return g.integrate_1d(u_rot, 2)

        if self.family == BicopFamily.tawn:
            # ExtremeValueBicop::hfunc2_raw
            psi1 = torch.as_tensor(self.parameters[..., 0], device=u_rot.device, dtype=u_rot.dtype).clamp(0.0, 1.0)
            psi2 = torch.as_tensor(self.parameters[..., 1], device=u_rot.device, dtype=u_rot.dtype).clamp(0.0, 1.0)
            theta = torch.as_tensor(self.parameters[..., 2], device=u_rot.device, dtype=u_rot.dtype).clamp(1.0, 1e9)

            tiny = torch.finfo(u_rot.dtype).tiny
            u1 = u_rot[:, 0]
            u2 = u_rot[:, 1]
            L = torch.log((u1 * u2).clamp_min(tiny))
            t = (torch.log(u2.clamp_min(tiny)) / L).clamp(0.0, 1.0)

            tmp = torch.pow((psi2 * t).clamp_min(tiny), theta) + torch.pow((psi1 * (1.0 - t)).clamp_min(tiny), theta)
            A = (1.0 - psi1) * (1.0 - t) + (1.0 - psi2) * t + torch.pow(tmp.clamp_min(tiny), 1.0 / theta)
            tmp2 = psi2 * torch.pow((psi2 * t).clamp_min(tiny), theta - 1.0) - psi1 * torch.pow((psi1 * (1.0 - t)).clamp_min(tiny), theta - 1.0)
            A1 = psi1 - psi2 + torch.pow(tmp.clamp_min(tiny), 1.0 / theta - 1.0) * tmp2

            t3 = A + (1.0 - t) * A1
            expo = (torch.log(u1.clamp_min(tiny)) + torch.log(u2.clamp_min(tiny))) * A
            return torch.exp(expo) * t3 / u2.clamp_min(tiny)

        uu = torch.stack([u_rot[:, 1], u_rot[:, 0]], dim=1)
        return self._hfunc1_0(uu)

    def _hinv1_0(self, u_rot: torch.Tensor) -> torch.Tensor:
        # inverse of h1 in second argument: input (u1, w) -> u2
        if self.family == BicopFamily.indep:
            return u_rot[:, 1]

        if self.family == BicopFamily.tll:
            u1 = u_rot[:, 0]
            w = u_rot[:, 1].clamp(1e-10, 1.0 - 1e-10)
            return _bisect_u2_for_hfunc1(u1, w, hfunc1_fn=self._hfunc1_0)

        if self.family == BicopFamily.gaussian:
            rho = torch.as_tensor(self.parameters[..., 0], device=u_rot.device, dtype=u_rot.dtype).clamp(-0.999999, 0.999999)
            x = stats.qnorm(u_rot)
            hinv = x[:, 1] * torch.sqrt(1.0 - rho * rho) + rho * x[:, 0]
            return stats.pnorm(hinv)

        if self.family == BicopFamily.student:
            rho = torch.as_tensor(self.parameters[..., 0], device=u_rot.device, dtype=u_rot.dtype).clamp(-0.999999, 0.999999)
            nu = torch.as_tensor(self.parameters[..., 1], device=u_rot.device, dtype=u_rot.dtype).clamp(2.01, 100.0)
            # Fast Hill qt approximation for inverse h-function.
            u_cat = torch.cat([u_rot[:, 0], u_rot[:, 1]])
            nu_cat = torch.cat([nu.expand(u_rot.shape[0]), (nu + 1.0).expand(u_rot.shape[0])])
            x_cat = stats._qt_hill(u_cat, nu_cat)
            n = u_rot.shape[0]
            x1, q = x_cat[:n], x_cat[n:]
            x2 = rho * x1 + q * torch.sqrt((nu + x1 * x1) * (1.0 - rho * rho) / (nu + 1.0))
            return stats.pt(x2, nu)

        if self.family == BicopFamily.frank:
            theta = torch.as_tensor(self.parameters[..., 0], device=u_rot.device, dtype=u_rot.dtype)
            if torch.any(theta == 0):
                return u_rot[:, 1]
            u1 = u_rot[:, 0]
            w = u_rot[:, 1]
            eu = torch.expm1(-theta * u1)
            ed = torch.expm1(-theta)
            A = eu + 1.0
            denom = (A - w * eu).clamp_min(torch.finfo(u_rot.dtype).tiny)
            ev = (w * ed) / denom
            return (-1.0 / theta) * torch.log1p(ev)

        if self.family == BicopFamily.clayton:
            theta = torch.as_tensor(self.parameters[..., 0], device=u_rot.device, dtype=u_rot.dtype)
            if torch.any(theta <= 0):
                raise ValueError("clayton parameter theta must be > 0")
            u1 = u_rot[:, 0]
            w = u_rot[:, 1]
            A = torch.pow(u1, -theta)
            S = torch.pow((w * torch.pow(u1, theta + 1.0)).clamp_min(torch.finfo(u_rot.dtype).tiny), -theta / (theta + 1.0))
            u2m = (S - A + 1.0).clamp_min(torch.finfo(u_rot.dtype).tiny)
            return torch.pow(u2m, -1.0 / theta)

        if self.family == BicopFamily.tawn:
            u1 = u_rot[:, 0]
            w = u_rot[:, 1]
            return _bisect_u2_for_hfunc1(u1, w, hfunc1_fn=self._hfunc1_0)

        if self.family == BicopFamily.gumbel:
            # Newton's method on the Archimedean hinv1 equation: find C s.t. phi'(u1)/phi'(C) = w
            theta = torch.as_tensor(self.parameters[..., 0], device=u_rot.device, dtype=u_rot.dtype)
            tiny = torch.finfo(u_rot.dtype).tiny
            u1 = u_rot[:, 0]
            w = u_rot[:, 1].clamp(1e-10, 1.0 - 1e-10)
            # target = phi'(u1) / w
            neg_log_u1 = (-torch.log(u1)).clamp_min(tiny)
            phi_p_u1 = -theta * torch.pow(neg_log_u1, theta - 1.0) / u1
            target = phi_p_u1 / w
            # Bisect to find C s.t. phi'(C) = target (phi' is monotonically decreasing toward -inf)
            C = _bisect_in_01_for_target_phi_prime_gumbel(theta, target)
            phi_C = torch.pow((-torch.log(C)).clamp_min(tiny), theta)
            phi_u1 = torch.pow(neg_log_u1, theta)
            phi_u2 = (phi_C - phi_u1).clamp_min(tiny)
            return stats.clamp_unit(torch.exp(-torch.pow(phi_u2, 1.0 / theta)))

        if self.family == BicopFamily.joe:
            # Newton's method for joe hinv1
            theta = torch.as_tensor(self.parameters[..., 0], device=u_rot.device, dtype=u_rot.dtype)
            tiny = torch.finfo(u_rot.dtype).tiny
            u1 = u_rot[:, 0]
            w = u_rot[:, 1].clamp(1e-10, 1.0 - 1e-10)
            b1 = (1.0 - u1).clamp_min(tiny)
            b1t = torch.pow(b1, theta)
            D1 = (1.0 - b1t).clamp_min(tiny)
            phi_p_u1 = -theta * torch.pow(b1, theta - 1.0) / D1
            target = phi_p_u1 / w
            C = _bisect_in_01_for_target_phi_prime_joe(theta, target)
            bC = (1.0 - C).clamp_min(tiny)
            bCt = torch.pow(bC, theta)
            DC = (1.0 - bCt).clamp_min(tiny)
            phi_C = -torch.log(DC.clamp_min(tiny))
            phi_u1 = -torch.log(D1.clamp_min(tiny))
            phi_u2 = (phi_C - phi_u1).clamp_min(tiny)
            et = torch.exp(-phi_u2)
            return stats.clamp_unit(1.0 - torch.pow((1.0 - et).clamp_min(tiny), 1.0 / theta))

        parts = self._arch_parts()
        if parts is not None:
            phi, psi, phi_p, _ = parts
            return _arch_hinv1(u_rot, phi=phi, psi=psi, phi_p=phi_p)

        raise NotImplementedError(f"hinv1 not implemented for family={self.family}")

    def _hinv2_0(self, u_rot: torch.Tensor) -> torch.Tensor:
        if self.family == BicopFamily.tawn:
            # Numeric inversion via bisection on hfunc2.
            w = u_rot[:, 0]
            u2 = u_rot[:, 1]
            return _bisect_u1_for_hfunc2(u2, w, hfunc2_fn=self._hfunc2_0)

        uu = torch.stack([u_rot[:, 1], u_rot[:, 0]], dim=1)
        return self._hinv1_0(uu)

    # ---- public wrappers with rotation logic (mirror C++ Bicop) ----

    # ---- abstract (discrete-aware) wrappers (operate on prepped, already-rotated u) ----

    def _pdf_abstract(self, u_prepped: torch.Tensor) -> torch.Tensor:
        if self.var_types == ("c", "c"):
            return self._pdf0(u_prepped[:, :2])

        if u_prepped.shape[1] != 4:
            raise ValueError("discrete models require 4 columns after formatting")

        umax = u_prepped[:, :2]
        umin = u_prepped[:, 2:4]

        # c,d or d,c
        if self.var_types != ("d", "d"):
            if self.var_types[0] != "c":
                udiff = (u_prepped[:, 0] - u_prepped[:, 2]).abs()
                out = torch.empty((u_prepped.shape[0],), device=u_prepped.device, dtype=u_prepped.dtype)
                mask = udiff > 5e-3
                if mask.any():
                    out[mask] = (self._hfunc2_0(umax[mask]) - self._hfunc2_0(umin[mask])) / udiff[mask]
                if (~mask).any():
                    mid = 0.5 * (umax[~mask] + umin[~mask])
                    out[~mask] = self._pdf0(mid)
                return out.abs()
            else:
                udiff = (u_prepped[:, 1] - u_prepped[:, 3]).abs()
                out = torch.empty((u_prepped.shape[0],), device=u_prepped.device, dtype=u_prepped.dtype)
                mask = udiff > 5e-3
                if mask.any():
                    out[mask] = (self._hfunc1_0(umax[mask]) - self._hfunc1_0(umin[mask])) / udiff[mask]
                if (~mask).any():
                    mid = 0.5 * (umax[~mask] + umin[~mask])
                    out[~mask] = self._pdf0(mid)
                return out.abs()

        # d,d
        udiff = (umax - umin).abs()
        out = torch.empty((u_prepped.shape[0],), device=u_prepped.device, dtype=u_prepped.dtype)

        mask_both_small = udiff.max(dim=1).values < 5e-3
        if mask_both_small.any():
            mid = 0.5 * (umax[mask_both_small] + umin[mask_both_small])
            out[mask_both_small] = self._pdf0(mid)

        mask_u1_small = (~mask_both_small) & (udiff[:, 0] < 5e-3)
        if mask_u1_small.any():
            uma = umax[mask_u1_small].clone()
            umi = umin[mask_u1_small].clone()
            uma[:, 0] = 0.5 * (uma[:, 0] + umi[:, 0])
            umi[:, 0] = uma[:, 0]
            out[mask_u1_small] = (self._hfunc1_0(uma) - self._hfunc1_0(umi)) / udiff[mask_u1_small, 1]

        mask_u2_small = (~mask_both_small) & (~mask_u1_small) & (udiff[:, 1] < 5e-3)
        if mask_u2_small.any():
            uma = umax[mask_u2_small].clone()
            umi = umin[mask_u2_small].clone()
            uma[:, 1] = 0.5 * (uma[:, 1] + umi[:, 1])
            umi[:, 1] = uma[:, 1]
            out[mask_u2_small] = (self._hfunc2_0(uma) - self._hfunc2_0(umi)) / udiff[mask_u2_small, 0]

        mask_full = (~mask_both_small) & (~mask_u1_small) & (~mask_u2_small)
        if mask_full.any():
            uma = umax[mask_full]
            umi = umin[mask_full]
            p = self._cdf0(uma) + self._cdf0(umi)
            uma2 = uma.clone()
            umi2 = umi.clone()
            uma2[:, 0], umi2[:, 0] = umi[:, 0], uma[:, 0]
            p = p - self._cdf0(uma2) - self._cdf0(umi2)
            out[mask_full] = p / (udiff[mask_full, 0] * udiff[mask_full, 1])

        return out.abs()

    def _hfunc1_abstract(self, u_prepped: torch.Tensor) -> torch.Tensor:
        if self.var_types[0] != "d":
            return self._hfunc1_0(u_prepped[:, :2])
        if u_prepped.shape[1] != 4:
            raise ValueError("discrete models require 4 columns after formatting")
        uu = u_prepped.clone()
        uu[:, 3] = uu[:, 1]
        u1diff = (uu[:, 0] - uu[:, 2]).abs()
        out = torch.empty((uu.shape[0],), device=uu.device, dtype=uu.dtype)
        mask = u1diff > 5e-3
        if mask.any():
            out[mask] = (self._cdf0(uu[mask, :2]) - self._cdf0(uu[mask, 2:4])) / u1diff[mask]
        if (~mask).any():
            mid = uu[~mask, :2].clone()
            mid[:, 0] = 0.5 * (uu[~mask, 0] + uu[~mask, 2])
            out[~mask] = self._hfunc1_0(mid)
        return out.abs()

    def _hfunc2_abstract(self, u_prepped: torch.Tensor) -> torch.Tensor:
        if self.var_types[1] != "d":
            return self._hfunc2_0(u_prepped[:, :2])
        if u_prepped.shape[1] != 4:
            raise ValueError("discrete models require 4 columns after formatting")
        uu = u_prepped.clone()
        uu[:, 2] = uu[:, 0]
        u2diff = (uu[:, 1] - uu[:, 3]).abs()
        out = torch.empty((uu.shape[0],), device=uu.device, dtype=uu.dtype)
        mask = u2diff > 5e-3
        if mask.any():
            out[mask] = (self._cdf0(uu[mask, :2]) - self._cdf0(uu[mask, 2:4])) / u2diff[mask]
        if (~mask).any():
            mid = uu[~mask, :2].clone()
            mid[:, 1] = 0.5 * (uu[~mask, 1] + uu[~mask, 3])
            out[~mask] = self._hfunc2_0(mid)
        return out.abs()

    def _hinv1_abstract(self, u_prepped: torch.Tensor) -> torch.Tensor:
        if self.var_types[0] == "c":
            return self._hinv1_0(u_prepped[:, :2])
        target = u_prepped[:, 1]

        def f(v: torch.Tensor) -> torch.Tensor:
            uu = u_prepped.clone()
            uu[:, 1] = v
            return self._hfunc1_abstract(uu)

        return _invert_unit_interval(target, f)

    def _hinv2_abstract(self, u_prepped: torch.Tensor) -> torch.Tensor:
        if self.var_types[1] == "c":
            return self._hinv2_0(u_prepped[:, :2])
        target = u_prepped[:, 0]

        def f(v: torch.Tensor) -> torch.Tensor:
            uu = u_prepped.clone()
            uu[:, 0] = v
            return self._hfunc2_abstract(uu)

        return _invert_unit_interval(target, f)

    def pdf(self, u: torch.Tensor) -> torch.Tensor:
        u_prepped = self._prep_for_abstract(u)
        out = self._pdf_abstract(u_prepped)
        return out.clamp_min(torch.finfo(out.dtype).tiny)

    def cdf(self, u: torch.Tensor) -> torch.Tensor:
        u = torch.as_tensor(u)
        if u.ndim != 2 or u.shape[1] < 2:
            raise ValueError("u must have at least 2 columns")
        u0 = stats.clamp_unit(u[:, :2])
        u_prepped = self._prep_for_abstract(u)
        p = self._cdf0(u_prepped[:, :2])
        rot = int(self.rotation)
        if rot == 0:
            return p
        if rot == 90:
            return u0[:, 1] - p
        if rot == 180:
            return p - 1.0 + (u0[:, 0] + u0[:, 1])
        if rot == 270:
            return u0[:, 0] - p
        _check_rotation(rot)
        raise AssertionError("unreachable")

    def hfunc1(self, u: torch.Tensor) -> torch.Tensor:
        u_prepped = self._prep_for_abstract(u)
        rot = int(self.rotation)
        if rot == 0:
            h = self._hfunc1_abstract(u_prepped)
        elif rot == 90:
            h = self._hfunc2_abstract(u_prepped)
        elif rot == 180:
            h = 1.0 - self._hfunc1_abstract(u_prepped)
        elif rot == 270:
            h = 1.0 - self._hfunc2_abstract(u_prepped)
        else:
            _check_rotation(rot)
            raise AssertionError("unreachable")
        return h.clamp(0.0, 1.0)

    def hfunc2(self, u: torch.Tensor) -> torch.Tensor:
        u_prepped = self._prep_for_abstract(u)
        rot = int(self.rotation)
        if rot == 0:
            h = self._hfunc2_abstract(u_prepped)
        elif rot == 90:
            h = 1.0 - self._hfunc1_abstract(u_prepped)
        elif rot == 180:
            h = 1.0 - self._hfunc2_abstract(u_prepped)
        elif rot == 270:
            h = self._hfunc1_abstract(u_prepped)
        else:
            _check_rotation(rot)
            raise AssertionError("unreachable")
        return h.clamp(0.0, 1.0)

    def hinv1(self, u: torch.Tensor) -> torch.Tensor:
        u_prepped = self._prep_for_abstract(u)
        rot = int(self.rotation)
        if rot == 0:
            out = self._hinv1_abstract(u_prepped)
        elif rot == 90:
            out = self._hinv2_abstract(u_prepped)
        elif rot == 180:
            out = 1.0 - self._hinv1_abstract(u_prepped)
        elif rot == 270:
            out = 1.0 - self._hinv2_abstract(u_prepped)
        else:
            _check_rotation(rot)
            raise AssertionError("unreachable")
        return out.clamp(0.0, 1.0)

    def hinv2(self, u: torch.Tensor) -> torch.Tensor:
        u_prepped = self._prep_for_abstract(u)
        rot = int(self.rotation)
        if rot == 0:
            out = self._hinv2_abstract(u_prepped)
        elif rot == 90:
            out = 1.0 - self._hinv1_abstract(u_prepped)
        elif rot == 180:
            out = 1.0 - self._hinv2_abstract(u_prepped)
        elif rot == 270:
            out = self._hinv1_abstract(u_prepped)
        else:
            _check_rotation(rot)
            raise AssertionError("unreachable")
        return out.clamp(0.0, 1.0)

    def simulate(self, n: int, *, device=None, dtype=None, seeds=None) -> torch.Tensor:
        if device is None:
            device = self.parameters.device if self.parameters is not None else None
        if dtype is None:
            dtype = self.parameters.dtype if (self.parameters is not None and self.parameters.numel() > 0) else torch.float32
        # Basic simulation via inverse Rosenblatt for implemented families.
        if n <= 0:
            raise ValueError("n must be positive")
        g = None
        if seeds:
            g = torch.Generator(device=device)
            g.manual_seed(int(seeds[0]))
        u = torch.rand((n, 2), generator=g, device=device, dtype=dtype)
        U2 = self.as_continuous().hinv1(u)
        return torch.stack([u[:, 0], U2], dim=1)

    def flip(self) -> None:
        """Flip variable order in-place (mirrors C++ Bicop::flip()).

        This is used internally when converting a selected vine tree set into an
        R-vine structure representation.
        """
        self._check_var_types()
        self.var_types = (self.var_types[1], self.var_types[0])
        rot = int(self.rotation)
        if rot == 90:
            self.rotation = 270
        elif rot == 270:
            self.rotation = 90
        # Family-specific shape changes.
        if self.family == BicopFamily.tawn:
            p = torch.as_tensor(self.parameters).reshape(-1)
            if p.numel() >= 2:
                p = p.clone()
                p[0], p[1] = p[1], p[0]
                self.parameters = p

    # ---- Fitting / Selection (Torch port; single-threaded) ----

    def loglik(self, data: torch.Tensor, *, weights: torch.Tensor | None = None) -> float:
        pdf = self.pdf(data)
        lp = torch.log(pdf)
        if weights is not None and weights.numel() > 0:
            w = torch.as_tensor(weights, dtype=lp.dtype, device=lp.device)
            lp = lp * w
        lp = lp[torch.isfinite(lp)]
        return float(lp.sum().item())

    def aic(self, data: torch.Tensor) -> float:
        return -2.0 * self.loglik(data) + 2.0 * self.get_npars()

    def bic(self, data: torch.Tensor) -> float:
        data = torch.as_tensor(data)
        n = float(data.shape[0])
        return -2.0 * self.loglik(data) + math.log(n) * self.get_npars()

    def mbic(self, data: torch.Tensor, *, psi0: float = 0.9) -> float:
        data = torch.as_tensor(data)
        n = float(data.shape[0])
        npars = self.get_npars()
        is_indep = self.family == BicopFamily.indep
        log_prior = (0.0 if is_indep else math.log(float(psi0))) + (math.log(1.0 - float(psi0)) if is_indep else 0.0)
        return -2.0 * self.loglik(data) + math.log(n) * npars - 2.0 * log_prior

    def get_npars(self) -> float:
        if self.family == BicopFamily.indep:
            return 0.0
        if self.family == BicopFamily.tll:
            # set by fit; fallback to grid size
            return float(torch.as_tensor(self.parameters).numel())
        return float(torch.as_tensor(self.parameters).reshape(-1).numel())

    def parameters_to_tau(self) -> float:
        # Mirrors Bicop::parameters_to_tau() including rotation sign flip.
        fam = self.family
        p = torch.as_tensor(self.parameters).reshape(-1).detach().cpu().tolist()
        if fam == BicopFamily.indep:
            tau = 0.0
        elif fam == BicopFamily.gaussian:
            rho = float(p[0])
            tau = (2.0 / math.pi) * math.asin(max(-1.0, min(1.0, rho)))
        elif fam == BicopFamily.student:
            rho = float(p[0])
            tau = (2.0 / math.pi) * math.asin(max(-1.0, min(1.0, rho)))
        elif fam == BicopFamily.clayton:
            th = abs(float(p[0]))
            tau = th / (2.0 + th)
        elif fam == BicopFamily.gumbel:
            th = float(p[0])
            tau = (th - 1.0) / th
        elif fam == BicopFamily.frank:
            th = float(p[0])
            ath = abs(th)
            if ath < 1e-5:
                tau = 0.0
            else:
                t = 1.0 - 4.0 / ath + (4.0 / ath) * _debye1(ath) / ath
                tau = t if th >= 0.0 else -t
        elif fam == BicopFamily.joe:
            th = float(p[0])
            tmp = 2.0 / th + 1.0
            dig = float(torch.digamma(torch.tensor(2.0)).item()) - float(torch.digamma(torch.tensor(tmp)).item())
            tau = 1.0 + 2.0 * dig / (2.0 - th)
        elif fam == BicopFamily.bb1:
            theta, delta = float(p[0]), float(p[1])
            tau = 1.0 - 2.0 / (delta * (theta + 2.0))
        else:
            # Used only for diagnostics/preselection; not required for fitting.
            raise NotImplementedError(f"parameters_to_tau not implemented for {fam}")

        if int(self.rotation) in (90, 270):
            tau *= -1.0
        return float(max(-1.0, min(1.0, tau)))

    def tau_to_parameters(self, tau: float) -> torch.Tensor:
        # Mirrors Bicop::tau_to_parameters() for one-parameter families.
        fam = self.family
        t = float(tau)
        dev = self.parameters.device
        dt = self.parameters.dtype
        if fam == BicopFamily.gaussian:
            return torch.tensor([math.sin(t * math.pi / 2.0)], device=dev, dtype=dt)
        if fam == BicopFamily.student:
            rho = math.sin(t * math.pi / 2.0)
            return torch.tensor([rho, 4.0], device=dev, dtype=dt)
        if fam == BicopFamily.clayton:
            at = abs(t)
            th = 2.0 * at / max(1e-12, 1.0 - at)
            return torch.tensor([min(max(th, 1e-10), 28.0)], device=dev, dtype=dt)
        if fam == BicopFamily.gumbel:
            at = abs(t)
            th = 1.0 / max(1e-12, 1.0 - at)
            return torch.tensor([min(max(th, 1.0), 50.0)], device=dev, dtype=dt)
        if fam == BicopFamily.frank:
            if abs(t) < 1e-12:
                return torch.tensor([0.0], device=dev, dtype=dt)
            def tau_of(theta: float) -> float:
                ath = abs(theta)
                if ath < 1e-5:
                    return 0.0
                tt = 1.0 - 4.0 / ath + (4.0 / ath) * _debye1(ath) / ath
                return tt if theta >= 0.0 else -tt

            lo = -35.0 + 1e-6
            hi = 35.0 - 1e-5
            th = _invert_monotone_1d(t, tau_of, lo=lo, hi=hi)
            return torch.tensor([th], device=dev, dtype=dt)
        if fam == BicopFamily.joe:
            at = abs(t)
            taus, thetas = _init_joe_tau_inverse_grid()
            th = _interp_monotone(at, taus, thetas)
            return torch.tensor([th], device=dev, dtype=dt)
        if fam == BicopFamily.indep:
            return torch.empty((0,), device=dev, dtype=dt)
        raise NotImplementedError(f"tau_to_parameters not implemented for {fam}")

    def _get_parameter_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        fam = self.family
        dev = self.parameters.device
        dt = self.parameters.dtype
        if fam == BicopFamily.indep:
            return torch.empty((0,), device=dev, dtype=dt), torch.empty((0,), device=dev, dtype=dt)
        if fam == BicopFamily.gaussian:
            return torch.tensor([-1.0], device=dev, dtype=dt), torch.tensor([1.0], device=dev, dtype=dt)
        if fam == BicopFamily.student:
            return torch.tensor([-1.0, 2.0], device=dev, dtype=dt), torch.tensor([1.0, 30.0], device=dev, dtype=dt)
        if fam == BicopFamily.clayton:
            return torch.tensor([1e-10], device=dev, dtype=dt), torch.tensor([28.0], device=dev, dtype=dt)
        if fam == BicopFamily.gumbel:
            return torch.tensor([1.0], device=dev, dtype=dt), torch.tensor([50.0], device=dev, dtype=dt)
        if fam == BicopFamily.frank:
            return torch.tensor([-35.0], device=dev, dtype=dt), torch.tensor([35.0], device=dev, dtype=dt)
        if fam == BicopFamily.joe:
            return torch.tensor([1.0], device=dev, dtype=dt), torch.tensor([30.0], device=dev, dtype=dt)
        if fam == BicopFamily.bb1:
            return torch.tensor([0.0, 1.0], device=dev, dtype=dt), torch.tensor([7.0, 7.0], device=dev, dtype=dt)
        if fam == BicopFamily.bb6:
            return torch.tensor([1.0, 1.0], device=dev, dtype=dt), torch.tensor([6.0, 8.0], device=dev, dtype=dt)
        if fam == BicopFamily.bb7:
            return torch.tensor([1.0, 0.01], device=dev, dtype=dt), torch.tensor([6.0, 25.0], device=dev, dtype=dt)
        if fam == BicopFamily.bb8:
            return torch.tensor([1.0, 1e-4], device=dev, dtype=dt), torch.tensor([8.0, 1.0], device=dev, dtype=dt)
        if fam == BicopFamily.tawn:
            return torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=dt), torch.tensor([1.0, 1.0, 60.0], device=dev, dtype=dt)
        raise NotImplementedError(f"bounds not implemented for {fam}")

    def _get_start_parameters(self, tau: float) -> torch.Tensor:
        fam = self.family
        dev = self.parameters.device
        dt = self.parameters.dtype
        if fam == BicopFamily.gaussian:
            return torch.tensor([math.sin(tau * math.pi / 2.0)], device=dev, dtype=dt)
        if fam == BicopFamily.student:
            return torch.tensor([math.sin(tau * math.pi / 2.0), 4.0], device=dev, dtype=dt)
        if fam in (BicopFamily.clayton, BicopFamily.gumbel, BicopFamily.frank, BicopFamily.joe):
            return self.tau_to_parameters(tau)
        if fam in (BicopFamily.bb1, BicopFamily.bb6, BicopFamily.bb7, BicopFamily.bb8):
            lb, _ub = self._get_parameter_bounds()
            return lb + 0.1
        if fam == BicopFamily.tawn:
            lb, _ub = self._get_parameter_bounds()
            return lb + 0.5
        # Fallback: keep current parameters (if any), otherwise lower bound + eps.
        if torch.as_tensor(self.parameters).numel() > 0:
            return torch.as_tensor(self.parameters, device=dev, dtype=dt).reshape(-1)
        lb, _ub = self._get_parameter_bounds()
        return lb + 0.1

    def _adjust_bounds_like_vinecopulib(self, lb: torch.Tensor, ub: torch.Tensor, *, tau: float, method: str) -> tuple[torch.Tensor, torch.Tensor]:
        fam = self.family
        tau = float(tau)
        # Refine 1D bounds for one-parameter families (Brent interval refinement).
        if fam in (BicopFamily.gaussian, BicopFamily.clayton, BicopFamily.gumbel, BicopFamily.frank, BicopFamily.joe):
            lb2 = lb.clone()
            ub2 = ub.clone()
            if fam in (BicopFamily.gaussian, BicopFamily.frank):
                lo_tau = max(tau - 0.1, -0.99)
                hi_tau = min(tau + 0.1, 0.99)
            else:
                lo_tau = max(abs(tau) - 0.1, 1e-10)
                hi_tau = min(abs(tau) + 0.1, 0.95)
            try:
                lb = self.tau_to_parameters(lo_tau)
                ub = self.tau_to_parameters(hi_tau)
            except Exception:
                lb = lb2
                ub = ub2
            lb = torch.max(lb2, lb)
            ub = torch.min(ub2, ub)

        if fam == BicopFamily.student:
            # Refine rho bounds from tau, keep nu bounds
            lo_tau = max(tau - 0.1, -0.99)
            hi_tau = min(tau + 0.1, 0.99)
            rho_lo = math.sin(lo_tau * math.pi / 2.0)
            rho_hi = math.sin(hi_tau * math.pi / 2.0)
            lb = torch.tensor([max(rho_lo, -0.999), 2.01], device=dev, dtype=dt)
            ub = torch.tensor([min(rho_hi, 0.999), 30.0], device=dev, dtype=dt)

        if fam == BicopFamily.tawn:
            lb = torch.tensor([0.3, 0.3, 1.5], device=dev, dtype=dt)
            ub = torch.tensor([1.0, 1.0, 7.0], device=dev, dtype=dt)
        return lb, ub

    def fit(self, data: torch.Tensor, controls: FitControlsBicop | None = None) -> "Bicop":
        # For parametric families, this currently keeps the provided parameters
        # (or defaults). For the nonparametric TLL family, it estimates the grid.
        if controls is None:
            controls = FitControlsBicop()

        data = torch.as_tensor(data)
        self.nobs = int(data.shape[0])
        self.to(device=data.device, dtype=data.dtype)
        self._fit_loglik = None  # cached loglik from fitting
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("data must have shape (n, >=2)")

        if self.family == BicopFamily.indep:
            self.parameters = torch.empty((0,), device=data.device, dtype=data.dtype)
            self._fit_loglik = 0.0
            return self

        if self.family == BicopFamily.tll:
            from .tll_fit import fit_tll

            vals, interp, _ll, _npars = fit_tll(
                data,
                method=controls.nonparametric_method,
                mult=controls.nonparametric_mult,
                weights=controls.weights,
                grid_size=30,
            )
            self.parameters = vals.detach().to(device=data.device, dtype=data.dtype)
            self._interp_grid = interp.to(device=data.device, dtype=data.dtype)
            self._fit_loglik = float(_ll)
            return self

        # ---- Parametric fitting (port of ParBicop::fit) ----
        if controls.parametric_method == "itau" and self.family not in _ITAU_FAMILIES:
            raise ValueError(f"parametric_method='itau' not available for family {self.family}")
        compute_fit_loglik = bool(getattr(controls, "compute_fit_loglik", False))

        # Kendall tau on rotated, formatted data (fits the rotated model).
        u_prepped = self._prep_for_abstract(data)
        tau = _estimate_tau_for_fit(u_prepped[:, 0], u_prepped[:, 1], controls)

        method = controls.parametric_method
        if self.family == BicopFamily.indep:
            self.parameters = torch.empty((0,), device=data.device, dtype=data.dtype)
            self._fit_loglik = 0.0
            return self

        # Method itau: if one-parameter family, no optimization required.
        if method == "itau":
            if self.family == BicopFamily.student:
                # Student-t itau: rho from tau, profile-optimize nu using fast Hill qt
                rho = math.sin(tau * math.pi / 2.0)
                rho_f = float(rho)
                rr = 1.0 - rho_f * rho_f
                log_rr = -0.5 * math.log(max(rr, 1e-20))
                rr_safe = max(rr, 1e-20)
                u1 = u_prepped[:, 0]
                u2 = u_prepped[:, 1]
                w_data = controls.weights

                def obj_nu(nu: torch.Tensor) -> torch.Tensor:
                    nu = nu.clamp(2.01, 30.0)
                    # Fast qt via Hill approximation (no Newton/betainc_reg)
                    x1 = stats._qt_hill(u1, nu)
                    x2 = stats._qt_hill(u2, nu)
                    # Inline copula log-density
                    log_c = (torch.lgamma((nu + 2.0) * 0.5) + torch.lgamma(nu * 0.5)
                             - 2.0 * torch.lgamma((nu + 1.0) * 0.5) + log_rr)
                    log_c = log_c + (nu + 1.0) * 0.5 * (torch.log1p(x1 * x1 / nu) + torch.log1p(x2 * x2 / nu))
                    Q = (x1 * x1 - 2.0 * rho_f * x1 * x2 + x2 * x2) / (nu * rr_safe)
                    log_c = log_c - (nu + 2.0) * 0.5 * torch.log1p(Q)
                    if w_data is not None and w_data.numel() > 0:
                        log_c = log_c * w_data.to(log_c.device, log_c.dtype)
                    lp = log_c[torch.isfinite(log_c)]
                    return lp.sum()

                res = torch_maximize_1d_with_grad(obj_nu, a=2.01, b=30.0, x0=4.0)
                self.parameters = torch.tensor([rho, float(res.x)], device=dev, dtype=dt)
                self._fit_loglik = float(res.fun) if compute_fit_loglik else None
                return self
            if self.family in (BicopFamily.gaussian, BicopFamily.clayton, BicopFamily.gumbel, BicopFamily.frank, BicopFamily.joe):
                self.parameters = self.tau_to_parameters(tau)
                self._fit_loglik = self.loglik(data, weights=controls.weights) if compute_fit_loglik else None
                return self
            raise NotImplementedError(f"itau not implemented for family {self.family}")

        # MLE / pseudo-MLE via optimized fitting.
        tau_w = _winsorize_tau_like_vinecopulib(tau)
        lb, ub = self._get_parameter_bounds()
        lb, ub = self._adjust_bounds_like_vinecopulib(lb, ub, tau=tau, method="mle")
        x0 = self._get_start_parameters(tau_w)
        x0 = torch.max(torch.min(x0, ub), lb)

        # Aggressive fast-MLE mode:
        # use tau-parameterized estimate as MLE approximation for 1-parameter families.
        if controls.aggressive_fast_mle and self.family in _FAST_MLE_1P_FAMILIES:
            p_fast = self.tau_to_parameters(tau)
            p_fast = torch.max(torch.min(torch.as_tensor(p_fast).reshape(-1), ub), lb)
            self.parameters = p_fast
            self._fit_loglik = self.loglik(data, weights=controls.weights) if compute_fit_loglik else None
            return self

        # Fast closed-form Gaussian copula MLE.
        if self.family == BicopFamily.gaussian:
            z = stats.qnorm(u_prepped[:, :2])
            w = controls.weights
            if w is not None and w.numel() > 0:
                w = w.to(z.device, z.dtype).reshape(-1)
                ww = w / w.sum().clamp_min(torch.finfo(z.dtype).tiny)
                m1 = (ww * z[:, 0]).sum()
                m2 = (ww * z[:, 1]).sum()
                c12 = (ww * (z[:, 0] - m1) * (z[:, 1] - m2)).sum()
                v1 = (ww * (z[:, 0] - m1) * (z[:, 0] - m1)).sum().clamp_min(1e-20)
                v2 = (ww * (z[:, 1] - m2) * (z[:, 1] - m2)).sum().clamp_min(1e-20)
                rho = (c12 / torch.sqrt(v1 * v2)).clamp(-0.999, 0.999)
            else:
                z1 = z[:, 0] - z[:, 0].mean()
                z2 = z[:, 1] - z[:, 1].mean()
                rho = ((z1 * z2).mean() / torch.sqrt((z1 * z1).mean().clamp_min(1e-20) * (z2 * z2).mean().clamp_min(1e-20))).clamp(-0.999, 0.999)
            self.parameters = torch.tensor([float(rho.item())], device=dev, dtype=dt)
            self._fit_loglik = self.loglik(data, weights=controls.weights) if compute_fit_loglik else None
            return self

        # Fast one-parameter MLE: tau init + short bounded autograd refinement.
        if self.family in (BicopFamily.clayton, BicopFamily.gumbel, BicopFamily.frank, BicopFamily.joe):
            try:
                p0 = self.tau_to_parameters(tau)
                p0 = torch.max(torch.min(torch.as_tensor(p0), ub), lb)
                u_fit_fast = self._prep_for_abstract(data)[:, :2]
                w_fit_fast = controls.weights

                def obj1_fast(pars: torch.Tensor) -> torch.Tensor:
                    self.parameters = pars.reshape(-1)
                    pdf_v = self._pdf0(u_fit_fast)
                    lp = torch.log(pdf_v.clamp_min(1e-300))
                    if w_fit_fast is not None and w_fit_fast.numel() > 0:
                        lp = lp * w_fit_fast.to(lp.device, lp.dtype)
                    return lp[torch.isfinite(lp)].sum()

                res_fast = torch_maximize_bounded_with_grad(obj1_fast, x0=p0.reshape(-1), lb=lb, ub=ub, max_iter=24)
                if math.isfinite(res_fast.fun):
                    self.parameters = res_fast.x.detach().clone()
                    self._fit_loglik = res_fast.fun
                    return self
            except Exception:
                if torchmin_strict_enabled():
                    raise

        # ---------- Student-t: fast profile approach using Hill qt approximation ----------
        # Strategy: use _qt_hill (84x faster than full qt) for all optimization calls,
        # then compute the final accurate loglik with the full qt at the end.
        # Profile approach replaces 2D coordinate descent (~840 loglik calls) with
        # two 1D searches (~60 total calls), giving ~14x speedup on top of the fast qt.
        if self.family == BicopFamily.student:
            rho0 = float(x0[0].item())
            nu_lo = float(lb[1].item())
            nu_hi = float(ub[1].item())
            nu0 = float(x0[1].item())
            rho_lo = float(lb[0].item())
            rho_hi = float(ub[0].item())
            w_student = controls.weights
            u_prepped_st = self._prep_for_abstract(data)[:, :2]
            u1_st = u_prepped_st[:, 0]
            u2_st = u_prepped_st[:, 1]

            def _fast_student_ll(rho_t: torch.Tensor, nu_t: torch.Tensor) -> torch.Tensor:
                """Fast Student-t loglik using Hill qt approximation."""
                rho_t = rho_t.clamp(-0.999999, 0.999999)
                nu_t = nu_t.clamp(2.01, 100.0)
                x1 = stats._qt_hill(u1_st, nu_t)
                x2 = stats._qt_hill(u2_st, nu_t)
                log_c = (torch.lgamma((nu_t + 2.0) * 0.5) + torch.lgamma(nu_t * 0.5)
                         - 2.0 * torch.lgamma((nu_t + 1.0) * 0.5)
                         - 0.5 * torch.log((1.0 - rho_t * rho_t).clamp_min(1e-20)))
                log_c = log_c + (nu_t + 1.0) * 0.5 * (torch.log1p(x1 * x1 / nu_t) + torch.log1p(x2 * x2 / nu_t))
                Q = (x1 * x1 - 2.0 * rho_t * x1 * x2 + x2 * x2) / (nu_t * (1.0 - rho_t * rho_t).clamp_min(1e-20))
                log_c = log_c - (nu_t + 2.0) * 0.5 * torch.log1p(Q)
                if w_student is not None and w_student.numel() > 0:
                    log_c = log_c * w_student.to(log_c.device, log_c.dtype)
                return log_c[torch.isfinite(log_c)].sum()

            # Pass 1: optimize nu with rho fixed at tau-based estimate
            rho0_t = torch.tensor(rho0, dtype=dt, device=dev)

            def obj_nu(nu_t: torch.Tensor) -> torch.Tensor:
                return _fast_student_ll(rho0_t, nu_t)

            res_nu = torch_maximize_1d_with_grad(obj_nu, a=nu_lo, b=nu_hi, x0=nu0, max_iter=24)
            nu_opt = float(res_nu.x.item())
            nu_opt_t = torch.tensor(nu_opt, dtype=dt, device=dev)

            # Pass 2: optimize rho with nu fixed
            def obj_rho(rho_t: torch.Tensor) -> torch.Tensor:
                return _fast_student_ll(rho_t, nu_opt_t)

            res_rho = torch_maximize_1d_with_grad(obj_rho, a=rho_lo, b=rho_hi, x0=rho0, max_iter=24)
            rho_opt = float(res_rho.x.item())
            rho_opt_t = torch.tensor(rho_opt, dtype=dt, device=dev)

            # Pass 3: refine nu with optimal rho
            def obj_nu2(nu_t: torch.Tensor) -> torch.Tensor:
                return _fast_student_ll(rho_opt_t, nu_t)

            res2 = torch_maximize_1d_with_grad(obj_nu2, a=nu_lo, b=nu_hi, x0=nu_opt, max_iter=24)
            nu_final = float(res2.x.item())

            self.parameters = torch.tensor([rho_opt, nu_final], device=dev, dtype=dt)
            # Final accurate loglik using full qt
            self._fit_loglik = self.loglik(data, weights=w_student)
            return self

        # ---------- 1-parameter families: gradient-based MLE via autograd ----------
        # Uses torch-native bounded L-BFGS (pytorch-minimize backend when available)
        # (~5-10 iterations vs 30 for golden section = 3-6x speedup).
        if x0.numel() == 1:
            a = float(lb[0].item())
            b = float(ub[0].item())

            _grad_fams = (
                BicopFamily.gaussian, BicopFamily.clayton, BicopFamily.gumbel,
                BicopFamily.frank, BicopFamily.joe,
            )
            if self.family in _grad_fams:
                u_fit = self._prep_for_abstract(data)[:, :2]
                w_fit = controls.weights

                def obj_grad(pars: torch.Tensor) -> torch.Tensor:
                    self.parameters = pars
                    pdf_v = self._pdf0(u_fit)
                    lp = torch.log(pdf_v.clamp_min(1e-300))
                    if w_fit is not None and w_fit.numel() > 0:
                        lp = lp * w_fit.to(lp.device, lp.dtype)
                    return lp[torch.isfinite(lp)].sum()

                try:
                    res = torch_maximize_bounded_with_grad(obj_grad, x0=x0, lb=lb, ub=ub, max_iter=24)
                    if math.isfinite(res.fun):
                        self.parameters = res.x.detach().clone()
                        self._fit_loglik = res.fun
                        return self
                except Exception:
                    if torchmin_strict_enabled():
                        raise

            # Fallback: black-box 1D search.
            def obj1(v: float) -> float:
                self.parameters = torch.tensor([float(v)], device=dev, dtype=dt)
                return self.loglik(data, weights=controls.weights)

            res = torch_maximize_1d(obj1, a=a, b=b, x0=float(x0[0].item()))
            self.parameters = torch.tensor([float(res.x.item())], device=dev, dtype=dt)
            self._fit_loglik = res.fun
            return self

        # ---------- Multi-parameter families: bounded gradient-based MLE ----------
        u_fit = self._prep_for_abstract(data)[:, :2]
        w_fit = controls.weights

        def obj_vec_grad(pars: torch.Tensor) -> torch.Tensor:
            self.parameters = torch.as_tensor(pars, device=dev, dtype=dt).reshape(-1)
            pdf_v = self._pdf0(u_fit)
            lp = torch.log(pdf_v.clamp_min(1e-300))
            if w_fit is not None and w_fit.numel() > 0:
                lp = lp * w_fit.to(lp.device, lp.dtype)
            return lp[torch.isfinite(lp)].sum()

        try:
            res = torch_maximize_bounded_with_grad(obj_vec_grad, x0=x0, lb=lb, ub=ub, max_iter=32)
        except Exception:
            if torchmin_strict_enabled():
                raise
            def obj_vec(pars: torch.Tensor) -> float:
                self.parameters = torch.as_tensor(pars, device=dev, dtype=dt).reshape(-1)
                return self.loglik(data, weights=controls.weights)

            res = torch_maximize_bounded(obj_vec, x0=x0, lb=lb, ub=ub)
        self.parameters = res.x.detach().clone()
        self._fit_loglik = res.fun
        return self

    def select(self, data: torch.Tensor, controls: FitControlsBicop | None = None) -> "Bicop":
        # Optimized port: precompute shared quantities once, then evaluate each candidate cheaply.
        if controls is None:
            controls = FitControlsBicop()

        data = torch.as_tensor(data)
        self.nobs = int(data.shape[0])
        self.to(device=data.device, dtype=data.dtype)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("data must have shape (n, >=2)")

        # Candidate family set depends on parametric_method.
        fams = list(controls.family_set) if controls.family_set else []
        if not fams:
            fams = list(_ITAU_FAMILIES) if controls.parametric_method == "itau" else list(BicopFamily)
        else:
            if controls.parametric_method == "itau":
                fams = [f for f in fams if f in _ITAU_FAMILIES]
                if not fams:
                    raise RuntimeError("No family with method itau provided")

        # Precompute tau ONCE on unrotated data — derive rotated tau by sign flip.
        tau0 = _estimate_tau_for_fit(data[:, 0], data[:, 1], controls)
        which_rot = (0, 180) if (tau0 > 0.0) else (90, 270)

        candidates: list[Bicop] = []
        for fam in fams:
            if not family_can_rotate(fam):
                candidates.append(Bicop(family=fam, rotation=0, var_types=self.var_types))
            else:
                if controls.allow_rotations:
                    candidates.append(Bicop(family=fam, rotation=which_rot[0], var_types=self.var_types))
                    candidates.append(Bicop(family=fam, rotation=which_rot[1], var_types=self.var_types))
                elif tau0 > 0.0:
                    candidates.append(Bicop(family=fam, rotation=0, var_types=self.var_types))

        # Remove combinations based on symmetry characteristics (preselection).
        if controls.preselect_families:
            z = stats.qnorm(stats.clamp_unit(data[:, :2]))
            x1 = z[:, 0]
            x2 = z[:, 1]
            if tau0 > 0.0:
                m1 = (x1 > 0) & (x2 > 0)
                m2 = (x1 < 0) & (x2 < 0)
            else:
                m1 = (x1 < 0) & (x2 > 0)
                m2 = (x1 > 0) & (x2 < 0)

            def _c_on_mask(mask: torch.Tensor) -> float:
                if not bool(mask.any()):
                    return 0.0
                idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
                if idx.numel() <= 1:
                    return 0.0
                idx = idx[:-1]
                w = None
                if controls.weights is not None and controls.weights.numel() > 0:
                    w = controls.weights[idx]
                return stats.pearson_cor(x1[idx], x2[idx], weights=w)

            c1 = _c_on_mask(m1)
            c2 = _c_on_mask(m2)
            cdiff = c1 - c2

            def keep(cop: Bicop) -> bool:
                fam = cop.family
                rot = int(cop.rotation)
                if not family_can_rotate(fam):
                    if (abs(cdiff) > 0.3) and (fam == BicopFamily.frank):
                        return False
                    return True
                is_bb = fam in (BicopFamily.bb1, BicopFamily.bb6, BicopFamily.bb7, BicopFamily.bb8)
                if is_bb:
                    if (tau0 > 0.0) and (rot in (0, 180)):
                        return True
                    if (tau0 < 0.0) and (rot in (90, 270)):
                        return True
                    return False
                is_90or180 = rot in (90, 180)
                lt = fam in (BicopFamily.clayton, BicopFamily.bb1, BicopFamily.bb7, BicopFamily.tawn)
                ut = fam in (BicopFamily.gumbel, BicopFamily.joe, BicopFamily.bb1, BicopFamily.bb6, BicopFamily.bb7, BicopFamily.bb8, BicopFamily.tawn)
                if cdiff > 0.05:
                    if lt and is_90or180:
                        return True
                    if ut and (not is_90or180):
                        return True
                    return False
                if cdiff < -0.05:
                    if lt and (not is_90or180):
                        return True
                    if ut and is_90or180:
                        return True
                    return False
                if (tau0 > 0.0) and (rot in (0, 180)):
                    return True
                if (tau0 < 0.0) and (rot in (90, 270)):
                    return True
                return False

            candidates = [c for c in candidates if keep(c)]

        # Precompute rotated data for each distinct rotation (avoid recomputation).
        u_clamped = stats.clamp_unit(data[:, :2])
        rotated_cache: dict[int, torch.Tensor] = {}
        for cand in candidates:
            r = int(cand.rotation)
            if r not in rotated_cache:
                rotated_cache[r] = _rotate_data_like_vinecopulib(u_clamped, r)

        # Precompute tau for each rotation: for rotation 90/270, tau flips sign; for 0/180 it stays.
        tau_for_rot: dict[int, float] = {}
        for r in rotated_cache:
            if r in (90, 270):
                tau_for_rot[r] = -tau0
            else:
                tau_for_rot[r] = tau0

        best = None
        best_score = -float("inf")
        n_eff = float(data.shape[0])
        method = controls.parametric_method

        for cand in candidates:
            try:
                rot = int(cand.rotation)
                cand.nobs = int(data.shape[0])
                u_rot = rotated_cache[rot]
                tau = tau_for_rot[rot]

                if cand.family == BicopFamily.indep:
                    cand.parameters = torch.empty((0,), device=data.device, dtype=data.dtype)
                    ll = 0.0
                elif cand.family == BicopFamily.tll:
                    cand.fit(data, controls)
                    ll = cand._fit_loglik if cand._fit_loglik is not None else cand.loglik(data, weights=controls.weights)
                elif method == "itau" and cand.family in _ITAU_FAMILIES:
                    if cand.family == BicopFamily.student:
                        # Student-t needs profile optimization for nu; use full fit
                        cand.fit(data, controls)
                        ll = cand._fit_loglik if cand._fit_loglik is not None else cand.loglik(data, weights=controls.weights)
                    else:
                        # Fast path: tau_to_parameters + loglik in one pass
                        cand.parameters = cand.tau_to_parameters(tau)
                        # Compute loglik directly on pre-rotated data to avoid re-rotating
                        pdf_vals = cand._pdf0(u_rot)
                        pdf_vals = pdf_vals.clamp_min(torch.finfo(pdf_vals.dtype).tiny)
                        lp = torch.log(pdf_vals)
                        if controls.weights is not None and controls.weights.numel() > 0:
                            w = torch.as_tensor(controls.weights, dtype=lp.dtype, device=lp.device)
                            lp = lp * w
                        lp = lp[torch.isfinite(lp)]
                        ll = float(lp.sum().item())
                elif method == "mle" and controls.aggressive_fast_mle and cand.family in _FAST_MLE_1P_FAMILIES:
                    # Fast MLE approximation for 1-parameter families: tau-parameterized estimate + direct loglik.
                    cand.parameters = cand.tau_to_parameters(tau)
                    pdf_vals = cand._pdf0(u_rot)
                    pdf_vals = pdf_vals.clamp_min(torch.finfo(pdf_vals.dtype).tiny)
                    lp = torch.log(pdf_vals)
                    if controls.weights is not None and controls.weights.numel() > 0:
                        w = torch.as_tensor(controls.weights, dtype=lp.dtype, device=lp.device)
                        lp = lp * w
                    lp = lp[torch.isfinite(lp)]
                    ll = float(lp.sum().item())
                else:
                    cand.fit(data, controls)
                    ll = cand._fit_loglik if cand._fit_loglik is not None else cand.loglik(data, weights=controls.weights)

                npars = cand.get_npars()
            except (NotImplementedError, Exception):
                continue

            # Convert criterion to a "score" to maximize.
            if controls.selection_criterion == "loglik":
                score = ll
            elif controls.selection_criterion == "aic":
                score = -(-2.0 * ll + 2.0 * npars)
            elif controls.selection_criterion == "bic":
                score = -(-2.0 * ll + math.log(n_eff) * npars)
            else:  # mbic
                is_indep = cand.family == BicopFamily.indep
                log_prior = (0.0 if is_indep else math.log(float(controls.psi0))) + (math.log(1.0 - float(controls.psi0)) if is_indep else 0.0)
                score = -(-2.0 * ll + math.log(n_eff) * npars - 2.0 * log_prior)

            if score > best_score:
                best_score = score
                best = cand

        if best is None:
            raise RuntimeError("no candidate model could be fitted/evaluated")

        self.family = best.family
        self.rotation = best.rotation
        self.parameters = best.parameters
        self._interp_grid = best._interp_grid
        self._fit_loglik = best_score  # cache the best score
        return self
