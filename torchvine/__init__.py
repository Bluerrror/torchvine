"""torchvine — Pure-PyTorch vine copula modelling.

GPU-ready, differentiable, and API-compatible with pyvinecopulib.
"""

from __future__ import annotations

__version__ = "0.2.2"

from .families import BicopFamily
from .bicop import Bicop
from .fit_controls import FitControlsBicop, FitControlsVinecop
from .rvine_structure import RVineStructure, DVineStructure, CVineStructure
from .vinecop import Vinecop
from .tll_fit import to_pseudo_obs
from .kde1d import Kde1d
from .pair_copuladata import pairs_copula_data
from .stats import (
    pearson_cor, kendall_tau, spearman_rho, blomqvist_beta, hoeffding_d, wdm,
)

import torch


def get_device(verbose: bool = False) -> torch.device:
    """Return the best available device (CUDA if available, else CPU).

    Parameters
    ----------
    verbose : bool, optional
        If ``True``, print which device was selected.

    Returns
    -------
    torch.device
    """
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    if verbose:
        print(f"torchvine: using device '{dev}'")
    return dev


def simulate_uniform(
    n: int,
    d: int,
    *,
    qrng: bool = False,
    seeds: list[int] | tuple[int, ...] = (),
) -> torch.Tensor:
    """Simulate from the multivariate uniform distribution.

    If ``qrng=True``, uses Sobol sequences for quasi-random numbers.
    Mirrors pyvinecopulib.simulate_uniform.
    """
    n = int(n)
    d = int(d)
    if qrng:
        eng = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=int(seeds[0]) if seeds else 0)
        return eng.draw(n)
    g = None
    if seeds:
        g = torch.Generator()
        g.manual_seed(int(seeds[0]))
    return torch.rand((n, d), generator=g)


def sobol(
    n: int,
    d: int,
    seeds: list[int] | tuple[int, ...] = (),
) -> torch.Tensor:
    """Simulate from the multivariate Sobol sequence.

    Mirrors pyvinecopulib.sobol.
    """
    eng = torch.quasirandom.SobolEngine(
        dimension=int(d), scramble=True, seed=int(seeds[0]) if seeds else 0,
    )
    return eng.draw(int(n))


def ghalton(
    n: int,
    d: int,
    seeds: list[int] | tuple[int, ...] = (),
) -> torch.Tensor:
    """Simulate from a quasi-random Halton-like sequence.

    Mirrors pyvinecopulib.ghalton. Uses scrambled Sobol as a high-quality
    quasi-random substitute (PyTorch does not include a native Halton engine).
    """
    eng = torch.quasirandom.SobolEngine(
        dimension=int(d), scramble=True, seed=int(seeds[0]) if seeds else 1,
    )
    return eng.draw(int(n))


# ---------------------------------------------------------------------------
# Individual family shortcut names (mirrors pyvinecopulib module-level names)
# ---------------------------------------------------------------------------
indep = BicopFamily.indep
gaussian = BicopFamily.gaussian
student = BicopFamily.student
clayton = BicopFamily.clayton
gumbel = BicopFamily.gumbel
frank = BicopFamily.frank
joe = BicopFamily.joe
bb1 = BicopFamily.bb1
bb6 = BicopFamily.bb6
bb7 = BicopFamily.bb7
bb8 = BicopFamily.bb8
tawn = BicopFamily.tawn
tll = BicopFamily.tll

# ---------------------------------------------------------------------------
# Family convenience lists (mirrors pyvinecopulib module-level constants)
# ---------------------------------------------------------------------------
one_par = [BicopFamily.gaussian, BicopFamily.clayton, BicopFamily.gumbel, BicopFamily.frank, BicopFamily.joe]
two_par = [BicopFamily.student, BicopFamily.bb1, BicopFamily.bb6, BicopFamily.bb7, BicopFamily.bb8]
three_par = [BicopFamily.tawn]
parametric = one_par + two_par + three_par
nonparametric = [BicopFamily.indep, BicopFamily.tll]
rotationless = [BicopFamily.indep, BicopFamily.gaussian, BicopFamily.student, BicopFamily.frank, BicopFamily.tll]
archimedean = [BicopFamily.clayton, BicopFamily.gumbel, BicopFamily.frank, BicopFamily.joe,
               BicopFamily.bb1, BicopFamily.bb6, BicopFamily.bb7, BicopFamily.bb8]
elliptical = [BicopFamily.gaussian, BicopFamily.student]
extreme_value = [BicopFamily.tawn, BicopFamily.gumbel]
bb = [BicopFamily.bb1, BicopFamily.bb6, BicopFamily.bb7, BicopFamily.bb8]
lt = [BicopFamily.clayton, BicopFamily.bb1, BicopFamily.bb7, BicopFamily.tawn]
ut = [BicopFamily.gumbel, BicopFamily.joe, BicopFamily.bb1, BicopFamily.bb6,
      BicopFamily.bb7, BicopFamily.bb8, BicopFamily.tawn]
itau = [BicopFamily.indep, BicopFamily.gaussian, BicopFamily.student, BicopFamily.clayton,
        BicopFamily.gumbel, BicopFamily.frank, BicopFamily.joe]
all = list(BicopFamily)


__all__ = [
    "BicopFamily",
    "Bicop",
    "FitControlsBicop",
    "FitControlsVinecop",
    "RVineStructure",
    "DVineStructure",
    "CVineStructure",
    "Vinecop",
    "Kde1d",
    "pairs_copula_data",
    "to_pseudo_obs",
    "get_device",
    "simulate_uniform",
    "sobol",
    "ghalton",
    # Dependence measures
    "pearson_cor",
    "kendall_tau",
    "spearman_rho",
    "blomqvist_beta",
    "hoeffding_d",
    "wdm",
    # Individual family shortcut names
    "indep",
    "gaussian",
    "student",
    "clayton",
    "gumbel",
    "frank",
    "joe",
    "bb1",
    "bb6",
    "bb7",
    "bb8",
    "tawn",
    "tll",
    # Family convenience lists
    "one_par",
    "two_par",
    "three_par",
    "parametric",
    "nonparametric",
    "rotationless",
    "archimedean",
    "elliptical",
    "extreme_value",
    "bb",
    "lt",
    "ut",
    "itau",
    "all",
]
