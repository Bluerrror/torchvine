"""Fitting controls for bivariate and vine copulas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from .families import BicopFamily


@dataclass
class FitControlsBicop:
    family_set: list[BicopFamily] = field(default_factory=list)
    parametric_method: str = "mle"  # "mle" or "itau"
    nonparametric_method: str = "constant"  # "constant", "linear", "quadratic"
    nonparametric_mult: float = 1.0
    selection_criterion: str = "bic"  # "loglik", "aic", "bic", "mbic"
    tau_estimation: str = "auto"  # "auto", "exact", "approx"
    weights: torch.Tensor | None = None
    psi0: float = 0.9
    preselect_families: bool = True
    allow_rotations: bool = True
    aggressive_fast_mle: bool = True  # use tau-parameterized fast MLE approximation for 1-parameter families
    compute_fit_loglik: bool = False  # skip loglik evaluation in fit() when speed is preferred
    num_threads: int = 4  # parallel edge fitting threads (1 = sequential)

    def __post_init__(self):
        if self.parametric_method not in ("mle", "itau"):
            raise ValueError("parametric_method must be 'mle' or 'itau'")
        if self.nonparametric_method not in ("constant", "linear", "quadratic"):
            raise ValueError("nonparametric_method must be 'constant', 'linear', or 'quadratic'")
        if self.nonparametric_mult <= 0:
            raise ValueError("nonparametric_mult must be positive")
        if self.selection_criterion not in ("loglik", "aic", "bic", "mbic", "mbicv"):
            raise ValueError("selection_criterion must be one of 'loglik','aic','bic','mbic','mbicv'")
        if self.tau_estimation not in ("auto", "exact", "approx"):
            raise ValueError("tau_estimation must be one of 'auto','exact','approx'")
        if not (0.0 < float(self.psi0) < 1.0):
            raise ValueError("psi0 must be in (0,1)")
        if self.weights is not None:
            self.weights = torch.as_tensor(self.weights)

    def str(self) -> str:
        """Human-readable summary (mirrors pyvinecopulib)."""
        fam_names = ", ".join(f.value.capitalize() for f in self.family_set) if self.family_set else "all"
        parts = [
            f"Family set: {fam_names}",
            f"Parametric method: {self.parametric_method}",
            f"Nonparametric method: {self.nonparametric_method}",
            f"Nonparametric multiplier: {self.nonparametric_mult}",
            f"Weights: {'yes' if self.weights is not None else 'no'}",
            f"Selection criterion: {self.selection_criterion}",
            f"Tau estimation: {self.tau_estimation}",
            f"Preselect families: {self.preselect_families}",
            f"Aggressive fast MLE: {self.aggressive_fast_mle}",
            f"Compute fit loglik: {self.compute_fit_loglik}",
            f"psi0: {self.psi0}",
            f"Number of threads: {self.num_threads}",
        ]
        return "\n".join(parts)


@dataclass
class FitControlsVinecop(FitControlsBicop):
    trunc_lvl: int | None = None
    tree_criterion: str = "tau"
    threshold: float = 0.0
    select_trunc_lvl: bool = False
    select_threshold: bool = False
    select_families: bool = True
    show_trace: bool = False
    tree_algorithm: str = "mst_prim"
    seeds: Sequence[int] = ()

    def __post_init__(self):
        super().__post_init__()
        if self.trunc_lvl is not None and int(self.trunc_lvl) < 0:
            raise ValueError("trunc_lvl must be >= 0")
        if self.tree_criterion not in ("tau", "rho", "joe", "hoeffd", "mcor"):
            raise ValueError("tree_criterion must be one of 'tau','rho','joe','hoeffd','mcor'")
        if self.threshold < 0:
            raise ValueError("threshold must be >= 0")
        if self.tree_algorithm not in ("mst_prim", "mst_kruskal", "random_weighted", "random_unweighted"):
            raise ValueError("tree_algorithm must be one of 'mst_prim','mst_kruskal','random_weighted','random_unweighted'")

    @property
    def mst_algorithm(self) -> str:
        """Alias for tree_algorithm (pyvinecopulib compatibility)."""
        return self.tree_algorithm

    @mst_algorithm.setter
    def mst_algorithm(self, value: str) -> None:
        self.tree_algorithm = value

    def str(self) -> str:
        """Human-readable summary (mirrors pyvinecopulib)."""
        base = super().str()
        parts = [
            base,
            f"Tree criterion: {self.tree_criterion}",
            f"Threshold: {self.threshold}",
            f"Truncation level: {self.trunc_lvl if self.trunc_lvl is not None else 'none'}",
            f"Select truncation level: {self.select_trunc_lvl}",
            f"Select threshold: {self.select_threshold}",
            f"Select families: {self.select_families}",
            f"Show trace: {self.show_trace}",
            f"Tree algorithm: {self.tree_algorithm}",
        ]
        return "\n".join(parts)
