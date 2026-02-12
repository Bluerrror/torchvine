"""torchvine — Pure-PyTorch vine copula modelling.

GPU-ready, differentiable, and API-compatible with pyvinecopulib.
"""

__version__ = "0.1.0"

from .families import BicopFamily
from .bicop import Bicop
from .fit_controls import FitControlsBicop, FitControlsVinecop
from .rvine_structure import RVineStructure, DVineStructure, CVineStructure
from .vinecop import Vinecop
from .tll_fit import to_pseudo_obs

import torch


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
        return eng.draw(n).to(torch.float64)
    g = None
    if seeds:
        g = torch.Generator()
        g.manual_seed(int(seeds[0]))
    return torch.rand((n, d), generator=g, dtype=torch.float64)


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
    return eng.draw(int(n)).to(torch.float64)


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
    return eng.draw(int(n)).to(torch.float64)


# ---------------------------------------------------------------------------
# Individual family shortcut names (mirrors pyvinecopulib module-level names)
# ---------------------------------------------------------------------------
indep = BicopFamily.indep
gaussian = BicopFamily.gaussian
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
# Note: student copula is intentionally excluded from this package.
# ---------------------------------------------------------------------------
one_par = [BicopFamily.gaussian, BicopFamily.clayton, BicopFamily.gumbel, BicopFamily.frank, BicopFamily.joe]
two_par = [BicopFamily.bb1, BicopFamily.bb6, BicopFamily.bb7, BicopFamily.bb8]
three_par = [BicopFamily.tawn]
parametric = one_par + two_par + three_par
nonparametric = [BicopFamily.indep, BicopFamily.tll]
rotationless = [BicopFamily.indep, BicopFamily.gaussian, BicopFamily.frank, BicopFamily.tll]
archimedean = [BicopFamily.clayton, BicopFamily.gumbel, BicopFamily.frank, BicopFamily.joe,
               BicopFamily.bb1, BicopFamily.bb6, BicopFamily.bb7, BicopFamily.bb8]
elliptical = [BicopFamily.gaussian]
extreme_value = [BicopFamily.tawn, BicopFamily.gumbel]
bb = [BicopFamily.bb1, BicopFamily.bb6, BicopFamily.bb7, BicopFamily.bb8]
lt = [BicopFamily.clayton, BicopFamily.bb1, BicopFamily.bb7, BicopFamily.tawn]
ut = [BicopFamily.gumbel, BicopFamily.joe, BicopFamily.bb1, BicopFamily.bb6,
      BicopFamily.bb7, BicopFamily.bb8, BicopFamily.tawn]
itau = [BicopFamily.indep, BicopFamily.gaussian, BicopFamily.clayton,
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
    "to_pseudo_obs",
    "simulate_uniform",
    "sobol",
    "ghalton",
    # Individual family shortcut names
    "indep",
    "gaussian",
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
