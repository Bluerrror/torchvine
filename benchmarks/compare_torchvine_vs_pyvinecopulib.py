"""Extended benchmark: torchvine vs pyvinecopulib (CPU)."""

from __future__ import annotations

import json
import time
import importlib.util
from pathlib import Path

import numpy as np
import pyvinecopulib as pv
import torch

import torchvine as tv
from torchvine import BicopFamily
from torchvine.fit_controls import FitControlsBicop, FitControlsVinecop


OUT_PATH = Path("benchmarks/results_torchvine_vs_pyvinecopulib.json")


def _timeit(fn, repeats: int, warmup: int = 1) -> float:
    for _ in range(max(0, warmup)):
        fn()
    t0 = time.perf_counter()
    for _ in range(max(1, repeats)):
        fn()
    return (time.perf_counter() - t0) / max(1, repeats)


def _timeit_median(fn, repeats: int, warmup: int = 1) -> float:
    for _ in range(max(0, warmup)):
        fn()
    vals = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    vals.sort()
    return float(vals[len(vals) // 2])


def _speedup(tv_t: float, pv_t: float) -> float:
    return float(pv_t / tv_t) if tv_t > 0.0 else float("inf")


def _pv_params(params: tuple[float, ...]) -> np.ndarray:
    return np.asarray(params, dtype=np.float64).reshape(-1, 1)


def _bicop_obj_data(n: int, dtype: torch.dtype) -> tuple[torch.Tensor, np.ndarray]:
    u_th = torch.rand((int(n), 2), dtype=dtype).clamp(1e-10, 1.0 - 1e-10)
    u_np = np.asfortranarray(u_th.detach().cpu().numpy())
    return u_th, u_np


def run() -> dict:
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(1)
    dtype = torch.float64

    families = [
        ("gaussian", (0.6,)),
        ("clayton", (2.0,)),
        ("gumbel", (1.8,)),
        ("frank", (5.0,)),
        ("joe", (2.5,)),
        ("student", (0.5, 5.0)),
    ]
    eval_ns = [2_000, 10_000, 50_000]
    fit_ns = [1_000, 4_000, 12_000]
    vine_dims = [3, 5, 8, 12]
    vine_train_ns = [800, 2_000, 6_000]

    out: dict[str, dict] = {
        "meta": {
            "torch_version": torch.__version__,
            "pyvinecopulib_version": getattr(pv, "__version__", "unknown"),
            "device": "cpu",
            "eval_ns": eval_ns,
            "fit_ns": fit_ns,
            "vine_dims": vine_dims,
            "vine_train_ns": vine_train_ns,
            "torchmin_available": bool(importlib.util.find_spec("torchmin") is not None),
        },
        "bicop_eval_grid": {},
        "bicop_fit_grid": {},
        "vinecop_grid": {},
    }

    # Bicop eval over sizes and families.
    for fam, par in families:
        rec_fam: dict[str, dict] = {}
        pv_fam = getattr(pv.BicopFamily, fam)
        pv_b = pv.Bicop(family=pv_fam, rotation=0, parameters=_pv_params(par))
        tv_b = tv.Bicop.from_family(fam, parameters=torch.tensor(par, dtype=dtype))
        for n in eval_ns:
            u_th, u_np = _bicop_obj_data(n=n, dtype=dtype)
            t_tv_pdf = _timeit(lambda: tv_b.pdf(u_th).sum().item(), repeats=4, warmup=1)
            t_pv_pdf = _timeit(lambda: pv_b.pdf(u_np).sum().item(), repeats=4, warmup=1)
            t_tv_h1 = _timeit(lambda: tv_b.hfunc1(u_th).sum().item(), repeats=4, warmup=1)
            t_pv_h1 = _timeit(lambda: pv_b.hfunc1(u_np).sum().item(), repeats=4, warmup=1)
            rec_fam[str(n)] = {
                "pdf_ms_torchvine": round(t_tv_pdf * 1000, 4),
                "pdf_ms_pyvinecopulib": round(t_pv_pdf * 1000, 4),
                "pdf_speedup_torchvine_over_pyvinecopulib": round(_speedup(t_tv_pdf, t_pv_pdf), 4),
                "hfunc1_ms_torchvine": round(t_tv_h1 * 1000, 4),
                "hfunc1_ms_pyvinecopulib": round(t_pv_h1 * 1000, 4),
                "hfunc1_speedup_torchvine_over_pyvinecopulib": round(_speedup(t_tv_h1, t_pv_h1), 4),
            }
        out["bicop_eval_grid"][fam] = rec_fam

    # Bicop MLE fit over sizes and families.
    for fam, par in families:
        rec_fam = {}
        pv_fam = getattr(pv.BicopFamily, fam)
        ctr_tv = FitControlsBicop(family_set=[getattr(BicopFamily, fam)], parametric_method="mle")
        ctr_pv = pv.FitControlsBicop(family_set=[pv_fam], parametric_method="mle")
        for n in fit_ns:
            obs = tv.Bicop.from_family(fam, parameters=torch.tensor(par, dtype=dtype)).simulate(int(n)).clamp(1e-10, 1.0 - 1e-10)
            obs_np = np.asfortranarray(obs.detach().cpu().numpy())

            def fit_tv():
                b = tv.Bicop.from_family(fam)
                b.fit(obs, controls=ctr_tv)
                return b

            def fit_pv():
                b = pv.Bicop(family=pv_fam)
                b.fit(obs_np, controls=ctr_pv)
                return b

            t_tv = _timeit_median(fit_tv, repeats=5, warmup=1)
            t_pv = _timeit_median(fit_pv, repeats=5, warmup=1)
            rec_fam[str(n)] = {
                "fit_mle_ms_torchvine": round(t_tv * 1000, 4),
                "fit_mle_ms_pyvinecopulib": round(t_pv * 1000, 4),
                "fit_mle_speedup_torchvine_over_pyvinecopulib": round(_speedup(t_tv, t_pv), 4),
            }
        out["bicop_fit_grid"][fam] = rec_fam

    # Vinecop select/pdf/sim over dimensions and train sample sizes.
    tv_fams = [BicopFamily.indep, BicopFamily.gaussian, BicopFamily.clayton, BicopFamily.gumbel, BicopFamily.frank, BicopFamily.joe]
    pv_fams = [pv.BicopFamily.indep, pv.BicopFamily.gaussian, pv.BicopFamily.clayton, pv.BicopFamily.gumbel, pv.BicopFamily.frank, pv.BicopFamily.joe]

    for d in vine_dims:
        rec_dim: dict[str, dict] = {}
        for n_train in vine_train_ns:
            x_th = torch.rand((int(n_train), int(d)), dtype=dtype).clamp(1e-10, 1.0 - 1e-10)
            x_np = np.asfortranarray(x_th.detach().cpu().numpy())

            ctr_tv = FitControlsVinecop(
                family_set=tv_fams,
                parametric_method="mle",
                selection_criterion="aic",
                tree_criterion="tau",
                trunc_lvl=min(4, d - 1),
                threshold=0.0,
                select_families=True,
                allow_rotations=True,
            )
            ctr_pv = pv.FitControlsVinecop(
                family_set=pv_fams,
                parametric_method="mle",
                selection_criterion="aic",
                tree_criterion="tau",
                trunc_lvl=min(4, d - 1),
                threshold=0.0,
                select_families=True,
                allow_rotations=True,
            )

            def sel_tv():
                v = tv.Vinecop.from_dimension(int(d))
                v.select(x_th, controls=ctr_tv)
                return v

            def sel_pv():
                v = pv.Vinecop(d=int(d))
                v.select(x_np, controls=ctr_pv)
                return v

            t_sel_tv = _timeit_median(sel_tv, repeats=3, warmup=1)
            t_sel_pv = _timeit_median(sel_pv, repeats=3, warmup=1)

            m_tv = sel_tv()
            m_pv = sel_pv()

            n_eval = max(2_000, int(n_train))
            x_eval_th = torch.rand((n_eval, int(d)), dtype=dtype).clamp(1e-10, 1.0 - 1.0e-10)
            x_eval_np = np.asfortranarray(x_eval_th.detach().cpu().numpy())

            t_pdf_tv = _timeit(lambda: m_tv.pdf(x_eval_th).sum().item(), repeats=3, warmup=1)
            t_pdf_pv = _timeit(lambda: m_pv.pdf(x_eval_np).sum().item(), repeats=3, warmup=1)

            n_sim = max(2_000, n_eval // 2)
            t_sim_tv = _timeit(lambda: m_tv.simulate(n_sim).sum().item(), repeats=3, warmup=1)
            t_sim_pv = _timeit(lambda: m_pv.simulate(n_sim).sum().item(), repeats=3, warmup=1)

            rec_dim[str(n_train)] = {
                "select_mle_ms_torchvine": round(t_sel_tv * 1000, 4),
                "select_mle_ms_pyvinecopulib": round(t_sel_pv * 1000, 4),
                "select_mle_speedup_torchvine_over_pyvinecopulib": round(_speedup(t_sel_tv, t_sel_pv), 4),
                "pdf_ms_torchvine": round(t_pdf_tv * 1000, 4),
                "pdf_ms_pyvinecopulib": round(t_pdf_pv * 1000, 4),
                "pdf_speedup_torchvine_over_pyvinecopulib": round(_speedup(t_pdf_tv, t_pdf_pv), 4),
                "simulate_ms_torchvine": round(t_sim_tv * 1000, 4),
                "simulate_ms_pyvinecopulib": round(t_sim_pv * 1000, 4),
                "simulate_speedup_torchvine_over_pyvinecopulib": round(_speedup(t_sim_tv, t_sim_pv), 4),
            }
        out["vinecop_grid"][f"d={d}"] = rec_dim

    ratios: list[float] = []
    for fam_rec in out["bicop_eval_grid"].values():
        for n_rec in fam_rec.values():
            ratios.append(float(n_rec["pdf_speedup_torchvine_over_pyvinecopulib"]))
            ratios.append(float(n_rec["hfunc1_speedup_torchvine_over_pyvinecopulib"]))
    for fam_rec in out["bicop_fit_grid"].values():
        for n_rec in fam_rec.values():
            ratios.append(float(n_rec["fit_mle_speedup_torchvine_over_pyvinecopulib"]))
    for dim_rec in out["vinecop_grid"].values():
        for n_rec in dim_rec.values():
            ratios.append(float(n_rec["select_mle_speedup_torchvine_over_pyvinecopulib"]))
            ratios.append(float(n_rec["pdf_speedup_torchvine_over_pyvinecopulib"]))
            ratios.append(float(n_rec["simulate_speedup_torchvine_over_pyvinecopulib"]))

    s = torch.tensor([r for r in ratios if r > 0], dtype=torch.float64)
    gmean = float(torch.exp(torch.log(s).mean()).item()) if s.numel() > 0 else float("nan")
    out["summary"] = {
        "num_speedup_points": int(s.numel()),
        "geometric_mean_speedup_torchvine_over_pyvinecopulib": round(gmean, 4),
    }
    return out


if __name__ == "__main__":
    res = run()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res["summary"], indent=2))
    print(f"saved={OUT_PATH}")
