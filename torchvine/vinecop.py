"""Vine copula model — fitting, evaluation, simulation, and transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import json
import torch

from .bicop import Bicop
from .stats import clamp_unit
from .rvine_structure import RVineStructure, DVineStructure, CVineStructure
from .fit_controls import FitControlsVinecop, FitControlsBicop
from .families import BicopFamily
import math


def _check_pair_copulas_shape(pair_copulas: Sequence[Sequence[Bicop]], d: int, trunc_lvl: int) -> None:
    if len(pair_copulas) != trunc_lvl:
        raise ValueError(f"pair_copulas must have {trunc_lvl} trees, got {len(pair_copulas)}")
    for t in range(trunc_lvl):
        expected = d - 1 - t
        if len(pair_copulas[t]) != expected:
            raise ValueError(f"pair_copulas[{t}] must have {expected} edges, got {len(pair_copulas[t])}")


@dataclass
class Vinecop:
    """Torch implementation of continuous vine copula *pdf* using an RVineStructure.

    Current scope:
    - Continuous models only (var_types = all 'c').
    - `pdf()` implemented for arbitrary R-vine structures given by `RVineStructure`.

    Not implemented (yet):
    - Discrete variables.
    - cdf/rosenblatt/inverse_rosenblatt for general vines.
    - fit/select.
    """

    structure: RVineStructure
    pair_copulas: list[list[Bicop]]
    var_types: list[str]
    nobs: int = 0
    threshold_: float = 0.0

    @classmethod
    def from_order(  # D-vine
        cls,
        order: Sequence[int],
        *,
        pair_copulas: Sequence[Sequence[Bicop]] | None = None,
        trunc_lvl: int | None = None,
        var_types: Sequence[str] | None = None,
    ) -> "Vinecop":
        structure = DVineStructure.from_order(order, trunc_lvl=trunc_lvl)
        return cls.from_structure(structure=structure, pair_copulas=pair_copulas, var_types=var_types)

    @classmethod
    def from_cvine_order(
        cls,
        order: Sequence[int],
        *,
        pair_copulas: Sequence[Sequence[Bicop]] | None = None,
        trunc_lvl: int | None = None,
        var_types: Sequence[str] | None = None,
    ) -> "Vinecop":
        structure = CVineStructure.from_order(order, trunc_lvl=trunc_lvl)
        return cls.from_structure(structure=structure, pair_copulas=pair_copulas, var_types=var_types)

    @classmethod
    def from_structure(
        cls,
        *,
        structure: RVineStructure | None = None,
        matrix=None,
        pair_copulas: Sequence[Sequence[Bicop]] | None = None,
        var_types: Sequence[str] | None = None,
    ) -> "Vinecop":
        if structure is None and matrix is not None:
            structure = RVineStructure.from_matrix(matrix)
        if structure is None:
            raise ValueError("either structure or matrix must be provided")
        d = int(structure.d)
        trunc_lvl = int(structure.trunc_lvl)

        if var_types is None:
            vt = ["c"] * d
        else:
            vt = [str(x) for x in var_types]
            if len(vt) != d:
                raise ValueError(f"var_types must have length {d}, got {len(vt)}")
        # Discrete variables are supported for pdf/fit/select/cdf.

        if pair_copulas is None:
            pcs = cls.make_pair_copula_store(d, trunc_lvl=trunc_lvl)
        else:
            pcs = [[pc for pc in tree] for tree in pair_copulas]
            _check_pair_copulas_shape(pcs, d, trunc_lvl)

        return cls(structure=structure, pair_copulas=pcs, var_types=vt, nobs=0)

    def _format_data(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Format input data into (u_main, u_sub_full) with shape (n,d) each.

        Accepts:
        - (n,d) continuous-only (u_sub_full = u_main)
        - (n,2d) where second block is u_sub for all variables
        - (n,d+k) where k = number of discrete variables and the extra columns
          contain u_sub for discrete variables in the order they appear.
        """
        u = torch.as_tensor(u, dtype=torch.float64)
        if u.ndim != 2:
            raise ValueError("u must be 2D")
        d = int(self.structure.d)
        k = sum(1 for t in self.var_types if t == "d")
        if u.shape[1] == d:
            # No discrete left-limits provided; treat sub as identical.
            return u, u
        if u.shape[1] == 2 * d:
            return u[:, :d], u[:, d:]
        if u.shape[1] == d + k:
            u_main = u[:, :d]
            u_sub = u_main.clone()
            disc_count = 0
            for j, t in enumerate(self.var_types):
                if t == "d":
                    u_sub[:, j] = u[:, d + disc_count]
                    disc_count += 1
            return u_main, u_sub
        raise ValueError(f"u must have shape (n,{d}), (n,{2*d}), or (n,{d+k}); got {tuple(u.shape)}")

    @classmethod
    def from_dimension(cls, d: int, *, var_types: Sequence[str] | None = None) -> "Vinecop":
        d = int(d)
        if d <= 0:
            raise ValueError("d must be positive")
        return cls.from_order(list(range(1, d + 1)), var_types=var_types)

    @classmethod
    def from_json(cls, s: str | dict[str, Any], *, check: bool = True) -> "Vinecop":
        # Minimal loader for the vinecopulib/pyvinecopulib structure JSON for RVineStructure.
        # This is only meant for test/cross-check use right now.
        if isinstance(s, str):
            obj = json.loads(s)
        else:
            obj = s

        if "structure" not in obj or "pair copulas" not in obj:
            raise ValueError("invalid Vinecop json: expected keys 'structure' and 'pair copulas'")

        structure = RVineStructure.from_vinecopulib_json(obj["structure"])
        # Pair copulas: only support the subset that Bicop.from_json understands.
        pcs: list[list[Bicop]] = []
        pc_node = obj["pair copulas"]
        for tree in range(structure.trunc_lvl):
            tree_key = f"tree{tree}"
            if tree_key not in pc_node:
                break
            tree_json = pc_node[tree_key]
            row: list[Bicop] = []
            for edge in range(structure.d - tree - 1):
                pc_key = f"pc{edge}"
                row.append(Bicop.from_json(tree_json[pc_key]))
            pcs.append(row)
        vc = cls.from_structure(structure=structure, pair_copulas=pcs, var_types=obj.get("var_types", None))
        vc.nobs = int(obj.get("nobs", 0))
        vc.threshold_ = float(obj.get("threshold", 0.0))
        return vc

    def to_json(self) -> dict[str, Any]:
        # Interop-friendly JSON (similar to pyvinecopulib/vinecopulib).
        structure = {
            "order": list(self.structure.order),
            "array": {
                "d": int(self.structure.d),
                "t": int(self.structure.trunc_lvl),
                "data": [row[:] for row in self.structure.struct_array],
            },
        }
        pcs: dict[str, Any] = {}
        for t, tree in enumerate(self.pair_copulas):
            node: dict[str, Any] = {}
            for e, pc in enumerate(tree):
                node[f"pc{e}"] = pc.to_json()
            pcs[f"tree{t}"] = node
        return {
            "structure": structure,
            "pair copulas": pcs,
            "var_types": list(self.var_types),
            "nobs": int(self.nobs),
            "threshold": float(self.threshold_),
        }

    @classmethod
    def from_file(cls, path: str) -> "Vinecop":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return cls.from_json(obj)

    def to_file(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2, sort_keys=True)

    @classmethod
    def from_data(
        cls,
        data: torch.Tensor,
        controls: FitControlsVinecop | None = None,
        *,
        var_types: Sequence[str] | None = None,
    ) -> "Vinecop":
        data = torch.as_tensor(data, dtype=torch.float64)
        if data.ndim != 2:
            raise ValueError("data must be 2D")
        if controls is None:
            controls = FitControlsVinecop()
        # d is inferred from the first block; var_types disambiguates (n,d+k).
        d = int(len(var_types)) if var_types is not None else int(data.shape[1])
        vc = cls.from_order(list(range(1, d + 1)), trunc_lvl=0, var_types=var_types)
        vc.select(data, controls)
        return vc

    @staticmethod
    def make_pair_copula_store(d: int, trunc_lvl: int | None = None) -> list[list[Bicop]]:
        d = int(d)
        if d <= 0:
            raise ValueError("d must be positive")
        if trunc_lvl is None:
            trunc_lvl = d - 1
        trunc = max(0, min(d - 1, int(trunc_lvl)))
        return [[Bicop() for _ in range(d - 1 - t)] for t in range(trunc)]

    def get_dim(self) -> int:
        return int(self.structure.d)

    @property
    def dim(self) -> int:
        return int(self.structure.d)

    @property
    def trunc_lvl(self) -> int:
        return int(self.structure.trunc_lvl)

    @property
    def order(self) -> list[int]:
        return list(self.structure.order)

    @property
    def matrix(self):
        return self.structure.matrix

    @property
    def npars(self) -> float:
        return float(self.get_npars())

    @property
    def parameters(self) -> list[list[torch.Tensor]]:
        return [[torch.as_tensor(pc.parameters, dtype=torch.float64) for pc in tree] for tree in self.pair_copulas]

    @property
    def families(self) -> list[list[str]]:
        return [[pc.family.value for pc in tree] for tree in self.pair_copulas]

    @property
    def rotations(self) -> list[list[int]]:
        return [[int(pc.rotation) for pc in tree] for tree in self.pair_copulas]

    @property
    def taus(self) -> list[list[float]]:
        return [[float(pc.tau) for pc in tree] for tree in self.pair_copulas]

    def get_tau(self, tree: int, edge: int) -> float:
        return float(self.get_pair_copula(tree, edge).tau)

    @property
    def threshold(self) -> float:
        return float(self.threshold_)

    def truncate(self, trunc_lvl: int) -> None:
        self._truncate(int(trunc_lvl))

    def _truncate(self, trunc_lvl: int) -> None:
        trunc_req = max(0, min(int(self.structure.d) - 1, int(trunc_lvl)))
        if trunc_req == int(self.structure.trunc_lvl):
            return
        self.structure = RVineStructure.from_order_and_struct_array(
            order=list(self.structure.order),
            struct_array=[row[:] for row in self.structure.struct_array[:trunc_req]],
            trunc_lvl=int(trunc_req),
            check=False,
        )
        self.pair_copulas = [tree[:] for tree in self.pair_copulas[:trunc_req]]

    def format(self) -> str:
        return json.dumps(self.to_json(), indent=2, sort_keys=True)

    def str(self) -> str:
        """Human-readable string representation (mirrors pyvinecopulib)."""
        d = int(self.structure.d)
        lines = [f"<torchvine.Vinecop> Vinecop model with {d} variables"]
        header = f"{'tree':>4}  {'edge':>4}  {'conditioned variables':>22}  {'conditioning variables':>23}  {'var_types':>10}  {'family':>10}  {'rotation':>8}  {'parameters':>20}  {'tau':>8}"
        lines.append(header)
        for t in range(len(self.pair_copulas)):
            for e in range(len(self.pair_copulas[t])):
                pc = self.pair_copulas[t][e]
                p = torch.as_tensor(pc.parameters, dtype=torch.float64).reshape(-1)
                pstr = ", ".join(f"{v:.2f}" for v in p.tolist()) if p.numel() > 0 and pc.family != BicopFamily.tll else ""
                try:
                    tau = f"{pc.tau:.2f}"
                except Exception:
                    tau = ""
                vt = str(list(pc.var_types))
                lines.append(
                    f"{t:>4}  {e:>4}  {'':>22}  {'':>23}  {vt:>10}  {pc.family.value:>10}  {pc.rotation:>8}  {pstr:>20}  {tau:>8}"
                )
        return "\n".join(lines)

    def plot(self, *args, **kwargs):
        import matplotlib.pyplot as plt

        title = kwargs.pop("title", "Vinecop structure")
        M = self.matrix.detach().cpu().to(torch.int64)
        M_shape0 = int(M.shape[0])
        M_shape1 = int(M.shape[1])

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = ax.imshow(M.tolist(), origin="upper", aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        for i in range(M_shape0):
            for j in range(M_shape1):
                val = int(M[i, j].item())
                if val != 0:
                    ax.text(j, i, str(val), ha="center", va="center", color="w", fontsize=9)
        fig.colorbar(im, ax=ax, label="matrix entries")
        return fig

    # --- API parity helpers (pyvinecopulib) ---
    def get_pair_copula(self, tree: int, edge: int) -> Bicop:
        return self.pair_copulas[int(tree)][int(edge)]

    def get_family(self, tree: int, edge: int) -> BicopFamily:
        return self.get_pair_copula(tree, edge).family

    def get_rotation(self, tree: int, edge: int) -> int:
        return int(self.get_pair_copula(tree, edge).rotation)

    def get_parameters(self, tree: int, edge: int) -> torch.Tensor:
        return torch.as_tensor(self.get_pair_copula(tree, edge).parameters, dtype=torch.float64)

    def pdf(self, u: torch.Tensor) -> torch.Tensor:
        """Evaluate continuous vine copula density (pdf)."""
        u = torch.as_tensor(u)
        d = self.structure.d
        trunc_lvl = self.structure.trunc_lvl
        if u.ndim != 2:
            raise ValueError("u must be 2D")

        u_main, u_sub = self._format_data(u)
        u_main = clamp_unit(u_main)
        u_sub = clamp_unit(u_sub)

        n = int(u_main.shape[0])
        device = u_main.device
        dtype = u_main.dtype

        if trunc_lvl == 0 or d == 1:
            return torch.ones((n,), device=device, dtype=dtype)

        # Reorder evaluation points to natural order (vectorized).
        order = self.structure.order
        order_idx = torch.tensor([o - 1 for o in order], dtype=torch.long, device=device)
        _hfunc2_init = u_main[:, order_idx]
        _hfunc2_sub_init = u_sub[:, order_idx]

        # Use lists of 1-D tensors (one per column) to avoid in-place
        # slice assignments that break autograd.
        hfunc2 = [_hfunc2_init[:, i] for i in range(d)]
        hfunc1 = [h.clone() for h in hfunc2]
        hfunc2_sub = [_hfunc2_sub_init[:, i] for i in range(d)]
        hfunc1_sub = [h.clone() for h in hfunc2_sub]

        pdf = torch.ones((n,), device=device, dtype=dtype)
        tiny = torch.finfo(dtype).tiny

        for tree in range(trunc_lvl):
            for edge in range(d - tree - 1):
                cop = self.pair_copulas[tree][edge]
                m = self.structure.min_at(tree, edge)
                u1 = hfunc2[edge]
                if m == self.structure.struct_at(tree, edge):
                    u2 = hfunc2[m - 1]
                    u2_sub = hfunc2_sub[m - 1]
                else:
                    u2 = hfunc1[m - 1]
                    u2_sub = hfunc1_sub[m - 1]

                if "d" in cop.var_types:
                    u1_sub = hfunc2_sub[edge]
                    uu = torch.stack([u1, u2, u1_sub, u2_sub], dim=1)
                else:
                    uu = torch.stack([u1, u2], dim=1)
                dens = cop.pdf(uu)
                pdf = pdf * dens

                # Compute h-functions only if needed (mirrors vinecopulib).
                if self.structure.needed_hfunc1_at(tree, edge):
                    hfunc1[edge] = cop.hfunc1(uu)
                    if cop.var_types[1] == "d" and uu.shape[1] == 4:
                        uu_sub = torch.stack([uu[:, 0], uu[:, 3], uu[:, 2], uu[:, 3]], dim=1)
                        hfunc1_sub[edge] = cop.hfunc1(uu_sub)
                    else:
                        hfunc1_sub[edge] = hfunc1[edge]
                if self.structure.needed_hfunc2_at(tree, edge):
                    hfunc2[edge] = cop.hfunc2(uu)
                    if cop.var_types[0] == "d" and uu.shape[1] == 4:
                        uu_sub = torch.stack([uu[:, 2], uu[:, 1], uu[:, 2], uu[:, 3]], dim=1)
                        hfunc2_sub[edge] = cop.hfunc2(uu_sub)
                    else:
                        hfunc2_sub[edge] = hfunc2[edge]

        # Prevent negative zeros from numerical noise (and avoid underflow to 0 from product)
        return pdf.clamp_min(tiny)

    def loglik(self, u: torch.Tensor) -> float:
        p = self.pdf(u)
        lp = torch.log(p)
        lp = lp[torch.isfinite(lp)]
        return float(lp.sum().item())

    def get_npars(self) -> float:
        return float(sum(pc.get_npars() for tree in self.pair_copulas for pc in tree))

    def aic(self, u: torch.Tensor) -> float:
        return -2.0 * self.loglik(u) + 2.0 * self.get_npars()

    def bic(self, u: torch.Tensor) -> float:
        n = float(torch.as_tensor(u).shape[0])
        return -2.0 * self.loglik(u) + math.log(n) * self.get_npars()

    def mbicv(self, u: torch.Tensor, *, psi0: float = 0.9) -> float:
        # Port of Vinecop::mbicv(u, psi0) for continuous data.
        n = float(torch.as_tensor(u).shape[0])
        ll = self.loglik(u)
        penalty = self.calculate_mbicv_penalty(int(n), float(psi0))
        return -2.0 * ll + penalty

    def calculate_mbicv_penalty(self, nobs: int, psi0: float) -> float:
        d = int(self.structure.d)
        if not (0.0 < psi0 < 1.0):
            raise ValueError("psi0 must be in (0,1)")
        non_indeps = [0] * (d - 1)
        for t in range(min(d - 1, len(self.pair_copulas))):
            for e in range(d - 1 - t):
                if self.pair_copulas[t][e].family != BicopFamily.indep:
                    non_indeps[t] += 1
        npars = self.get_npars()
        log_prior = 0.0
        for t in range(d - 1):
            ps = psi0 ** float(t + 1)
            log_prior += float(non_indeps[t]) * math.log(ps) + float(d - non_indeps[t] - (t + 1)) * math.log(1.0 - ps)
        return math.log(float(nobs)) * float(npars) - 2.0 * float(log_prior)

    def cdf(self, u: torch.Tensor, *, N: int = 10000, seeds=None, batch_size: int = 256) -> torch.Tensor:
        """Monte Carlo CDF estimate (mirrors vinecopulib's approach).

        For discrete models, only the first d columns are used as evaluation points.
        """
        u = torch.as_tensor(u, dtype=torch.float64)
        u_main, _u_sub = self._format_data(u)
        u_main = clamp_unit(u_main)
        n = int(u_main.shape[0])
        d = int(u_main.shape[1])
        if d != int(self.structure.d):
            raise ValueError("u has wrong dimension")
        # Simulate quasi-random numbers from the model.
        u_sim = self.simulate(int(N), seeds=seeds)
        out = torch.empty((n,), device=u_main.device, dtype=u_main.dtype)
        for i0 in range(0, n, int(batch_size)):
            i1 = min(n, i0 + int(batch_size))
            temp = u_main[i0:i1]  # (b,d)
            # count rows where all coordinates <= temp
            # equivalent to max(u_sim - temp) <= 0
            diff = u_sim[:, None, :] - temp[None, :, :]
            mx = torch.amax(diff, dim=2)
            out[i0:i1] = (mx <= 0.0).to(u_main.dtype).mean(dim=0)
        return out

    def fit(self, data: torch.Tensor, controls: FitControlsBicop | None = None, num_threads: int = 1) -> None:
        """Fit parameters of a pre-specified vine copula model (fixed structure + families).

        Matches pyvinecopulib's `Vinecop.fit()` signature; `num_threads` is accepted
        for API parity but ignored (torch port is single-threaded here).
        """
        _ = num_threads
        if controls is None:
            controls = FitControlsBicop()

        from .vine_select import VinecopSelectorTorch

        ctr = FitControlsVinecop(
            trunc_lvl=int(self.structure.trunc_lvl),
            tree_criterion="tau",
            threshold=0.0,
            select_threshold=False,
            select_trunc_lvl=False,
            select_families=False,
            tree_algorithm="mst_prim",
            seeds=(),
            family_set=list(controls.family_set),
            parametric_method=controls.parametric_method,
            nonparametric_method=controls.nonparametric_method,
            nonparametric_mult=controls.nonparametric_mult,
            selection_criterion=controls.selection_criterion,
            weights=controls.weights,
            psi0=controls.psi0,
            preselect_families=controls.preselect_families,
            allow_rotations=controls.allow_rotations,
        )
        data = torch.as_tensor(data, dtype=torch.float64)
        self.nobs = int(data.shape[0])
        selector = VinecopSelectorTorch(
            data=clamp_unit(data),
            controls=ctr,
            var_types=list(self.var_types),
            vine_struct=self.structure,
            pair_copulas=self.pair_copulas,
        )
        struct, pcs = selector.select_all_trees()
        self.structure = struct
        self.pair_copulas = pcs
        self.threshold_ = float(ctr.threshold)

    def select(self, data: torch.Tensor, controls: FitControlsVinecop | None = None) -> None:
        """Select families/parameters and optionally structure.

        Matches pyvinecopulib's in-place `Vinecop.select()`.
        """
        if controls is None:
            controls = FitControlsVinecop()
        data = torch.as_tensor(data, dtype=torch.float64)
        d = self.structure.d
        if data.ndim != 2:
            raise ValueError("data must be 2D")

        trunc_req = controls.trunc_lvl if controls.trunc_lvl is not None else (d - 1)
        trunc_req = max(0, min(d - 1, int(trunc_req)))

        cur_trunc = int(self.structure.trunc_lvl)
        if trunc_req < cur_trunc:
            # Truncate in-place.
            self.structure = RVineStructure.from_order_and_struct_array(
                order=list(self.structure.order),
                struct_array=[row[:] for row in self.structure.struct_array[:trunc_req]],
                trunc_lvl=int(trunc_req),
                check=False,
            )
            self.pair_copulas = [tree[:] for tree in self.pair_copulas[:trunc_req]]
            self.nobs = int(data.shape[0])
            self.threshold_ = float(getattr(controls, "threshold", 0.0))
            return

        from .vine_select import VinecopSelectorTorch

        self.nobs = int(data.shape[0])
        selector = VinecopSelectorTorch(
            data=clamp_unit(data),
            controls=controls,
            var_types=list(self.var_types),
            vine_struct=self.structure if cur_trunc > 0 else None,
            pair_copulas=(self.pair_copulas if not controls.select_families else None),
        )
        struct, pcs = selector.select_all_trees()
        self.structure = struct
        self.pair_copulas = pcs
        self.threshold_ = float(controls.threshold)

    def rosenblatt(
        self,
        u: torch.Tensor,
        *,
        num_threads: int = 1,
        randomize_discrete: bool = True,
        seeds=(),
    ) -> torch.Tensor:
        """Rosenblatt transform (supports discrete randomization).

        Mirrors C++ Vinecop::rosenblatt() at a high level (single-threaded).
        If a variable is discrete and `randomize_discrete=True`, the output is
        randomized as:
          W * F + (1-W) * F^-,
        where F^- is the left limit (provided via the discrete data layout).
        """
        _ = num_threads
        u = torch.as_tensor(u, dtype=torch.float64)
        u_main, u_sub = self._format_data(u)
        d = int(self.structure.d)
        trunc_lvl = int(self.structure.trunc_lvl)
        if u_main.ndim != 2 or int(u_main.shape[1]) != d:
            raise ValueError(f"u must have shape (n,{d}) (or discrete layouts); got {tuple(u.shape)}")

        u_main = clamp_unit(u_main)
        u_sub = clamp_unit(u_sub)
        n = int(u_main.shape[0])
        device = u_main.device
        dtype = u_main.dtype

        order = self.structure.order
        order_idx = torch.tensor([o - 1 for o in order], dtype=torch.long, device=device)
        inverse_order = torch.empty(d, dtype=torch.long, device=device)
        inverse_order[order_idx] = torch.arange(d, device=device)

        hfunc1 = torch.zeros((n, d), device=device, dtype=dtype)
        hfunc2 = u_main[:, order_idx]
        hfunc1_sub = torch.zeros((n, d), device=device, dtype=dtype)
        hfunc2_sub = u_sub[:, order_idx]
        hfunc1[:] = hfunc2
        hfunc1_sub[:] = hfunc2_sub

        # Reuse per-edge buffers to avoid repeated allocations.
        uu2 = torch.empty((n, 2), device=device, dtype=dtype)
        uu4 = torch.empty((n, 4), device=device, dtype=dtype)
        uu4_sub = torch.empty((n, 4), device=device, dtype=dtype)

        for tree in range(trunc_lvl):
            for edge in range(d - tree - 1):
                cop = self.pair_copulas[tree][edge]
                m = self.structure.min_at(tree, edge)
                u1 = hfunc2[:, edge]
                if m == self.structure.struct_at(tree, edge):
                    u2 = hfunc2[:, m - 1]
                    u2_sub = hfunc2_sub[:, m - 1]
                else:
                    u2 = hfunc1[:, m - 1]
                    u2_sub = hfunc1_sub[:, m - 1]

                uu2[:, 0] = u1
                uu2[:, 1] = u2
                uu = uu2
                if "d" in cop.var_types:
                    u1_sub = hfunc2_sub[:, edge]
                    uu4[:, 0] = u1
                    uu4[:, 1] = u2
                    uu4[:, 2] = u1_sub
                    uu4[:, 3] = u2_sub
                    uu = uu4

                # Compute h-functions only if needed (mirrors vinecopulib).
                if self.structure.needed_hfunc1_at(tree, edge):
                    hfunc1[:, edge] = cop.hfunc1(uu)
                    if cop.var_types[1] == "d" and uu.shape[1] == 4:
                        uu4_sub[:] = uu  # type: ignore[assignment]
                        uu4_sub[:, 1] = uu4_sub[:, 3]
                        hfunc1_sub[:, edge] = cop.hfunc1(uu4_sub)
                    else:
                        hfunc1_sub[:, edge] = hfunc1[:, edge]

                hfunc2[:, edge] = cop.hfunc2(uu)
                if cop.var_types[0] == "d" and uu.shape[1] == 4:
                    uu4_sub[:] = uu  # type: ignore[assignment]
                    uu4_sub[:, 0] = uu4_sub[:, 2]
                    hfunc2_sub[:, edge] = cop.hfunc2(uu4_sub)
                else:
                    hfunc2_sub[:, edge] = hfunc2[:, edge]

        # back to original order (vectorized)
        out = hfunc2[:, inverse_order]
        out_sub = hfunc2_sub[:, inverse_order]

        if randomize_discrete and any(t == "d" for t in self.var_types):
            g = None
            if seeds:
                g = torch.Generator(device=device)
                g.manual_seed(int(seeds[0]))
            w = torch.rand((n, d), generator=g, device=device, dtype=dtype)
            disc_mask = torch.tensor([t == "d" for t in self.var_types], dtype=torch.bool, device=device)
            out[:, disc_mask] = w[:, disc_mask] * out[:, disc_mask] + (1.0 - w[:, disc_mask]) * out_sub[:, disc_mask]

        return out.clamp(min=1e-10, max=1.0 - 1e-10)

    def inverse_rosenblatt(self, u: torch.Tensor, *, num_threads: int = 1) -> torch.Tensor:
        """Inverse Rosenblatt transform.

        Mirrors C++ Vinecop::inverse_rosenblatt() (single-threaded, no splitting).
        """
        _ = num_threads
        u = torch.as_tensor(u)
        d = self.structure.d
        trunc_lvl = self.structure.trunc_lvl
        if u.ndim != 2 or u.shape[1] != d:
            raise ValueError(f"u must have shape (n, {d}); got {tuple(u.shape)}")

        u = clamp_unit(u)
        n = u.shape[0]
        device = u.device
        dtype = u.dtype

        order = self.structure.order
        order_idx = torch.tensor([o - 1 for o in order], dtype=torch.long, device=device)
        inverse_order = torch.empty(d, dtype=torch.long, device=device)
        inverse_order[order_idx] = torch.arange(d, device=device)

        # Triangular arrays of tensors (tree index 0..trunc_lvl, var index 0..d-1)
        hinv2: list[list[torch.Tensor | None]] = [[None] * d for _ in range(trunc_lvl + 1)]
        hfunc1: list[list[torch.Tensor | None]] = [[None] * d for _ in range(trunc_lvl + 1)]

        # initialize with independent uniforms (natural order)
        for j in range(d):
            idx = min(trunc_lvl, d - j - 1)
            hinv2[idx][j] = u[:, order[j] - 1]
        hinv2[0][d - 1] = hinv2[0][d - 1] if hinv2[0][d - 1] is not None else u[:, order[d - 1] - 1]
        hfunc1[0][d - 1] = hinv2[0][d - 1]

        for var in range(d - 2, -1, -1):
            tree_start = min(trunc_lvl - 1, d - var - 2)
            for tree in range(tree_start, -1, -1):
                # In vinecopulib, simulation/inverse_rosenblatt always returns
                # continuous uniforms even if the model includes discrete vars.
                # We therefore invert using the continuous version of each pair copula.
                cop = self.pair_copulas[tree][var].as_continuous()

                m = self.structure.min_at(tree, var)
                u0 = hinv2[tree + 1][var]
                if u0 is None:
                    raise RuntimeError("internal inverse_rosenblatt error: missing hinv2(tree+1,var)")

                if m == self.structure.struct_at(tree, var):
                    u1 = hinv2[tree][m - 1]
                else:
                    u1 = hfunc1[tree][m - 1]
                if u1 is None:
                    raise RuntimeError("internal inverse_rosenblatt error: missing conditioning value")

                U_e = torch.stack([u0, u1], dim=1)
                hinv2[tree][var] = cop.hinv2(U_e)

                # compute hfunc1 for later use (only if required)
                if var < d - 1 and self.structure.needed_hfunc1_at(tree, var):
                    U_e2 = torch.stack([hinv2[tree][var], u1], dim=1)  # type: ignore[arg-type]
                    hfunc1[tree + 1][var] = cop.hfunc1(U_e2)

        # Vectorized output assembly
        vals = [hinv2[0][int(inverse_order[j].item())] for j in range(d)]
        for j, val in enumerate(vals):
            if val is None:
                raise RuntimeError("internal inverse_rosenblatt error: missing output")
        out = torch.stack(vals, dim=1)  # type: ignore[arg-type]
        return out

    def simulate(self, n: int, *, device=None, dtype=torch.float64, seeds=None) -> torch.Tensor:
        # Mirrors C++: generate uniforms then inverse_rosenblatt.
        g = None
        if seeds:
            g = torch.Generator(device=device)
            g.manual_seed(int(seeds[0]))
        u = torch.rand((int(n), self.structure.d), generator=g, device=device, dtype=dtype)
        return self.inverse_rosenblatt(u)
