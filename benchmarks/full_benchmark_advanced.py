"""Full benchmark + advanced plotting for fitting speed comparisons.

Benchmarks:
  - Bicop MLE fit
  - Bicop itau fit
  - Bicop family selection (MLE)
  - Vine selection+fit (MLE)
  - Vine selection+fit (itau)

Libraries:
  - torchvine
  - torchvinecopulib
  - pyvinecopulib
"""

from __future__ import annotations

import gc
import importlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from functools import partial as _partial


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "benchmarks" / "results"
OUT_JSON = OUT_DIR / "full_benchmark_advanced.json"

# Enforce strict torchmin for torchvine optimization.
os.environ.setdefault("TORCHVINE_REQUIRE_TORCHMIN", "1")

sys.path.insert(0, str(ROOT))

import pyvinecopulib as pv
import torchvine as tv
from torchvine import BicopFamily
from torchvine.fit_controls import FitControlsBicop, FitControlsVinecop
from torchvinecopulib import bicop as tvcl_bicop
from torchvinecopulib import vinecop as tvcl_vinecop
from torchvinecopulib.util import ENUM_FUNC_BIDEP as _ENUM_BIDEP, kendall_tau as _kt, chatterjee_xi as _cx

# torchvinecopulib enum compatibility on newer Python versions.
for _name, _fn in [("kendall_tau", _kt), ("chatterjee_xi", _cx)]:
    if _name not in _ENUM_BIDEP._member_map_:
        _ENUM_BIDEP._member_map_[_name] = type("_Fake", (), {"value": _partial(_fn)})()


COMMON_FAMILIES = [
    ("gaussian", "Gaussian", (0.6,)),
    ("clayton", "Clayton", (2.0,)),
    ("gumbel", "Gumbel", (1.8,)),
    ("frank", "Frank", (5.0,)),
    ("joe", "Joe", (2.5,)),
]

BICOP_NS = [500, 1_000, 2_000, 5_000, 10_000]
VINE_DIMS = [3, 5, 8, 10]
VINE_NS = [500, 1_000, 3_000]

BICOP_REPEATS = 3
VINE_REPEATS = 2

COLORS = {
    "torchvine": "#f08c00",
    "torchvinecopulib": "#1d4ed8",
    "pyvinecopulib": "#16a34a",
}


@dataclass
class FlatPoint:
    section: str
    subgroup: str
    x: str
    torchvine_ms: float
    torchvinecopulib_ms: float
    pyvinecopulib_ms: float
    speedup_tv_over_tvcl: float
    speedup_tv_over_pv: float


def _time_median(fn, repeats: int, warmup: int = 1) -> float:
    for _ in range(max(0, warmup)):
        fn()
    gc.collect()
    vals = []
    for _ in range(max(1, repeats)):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    return float(statistics.median(vals))


def _speedup(ref: float, target: float) -> float:
    return float(ref / target) if target > 0.0 else float("inf")


def bench_bicop_fit(method: str) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for tv_name, tvcl_name, true_params in COMMON_FAMILIES:
        out[tv_name] = {}
        for n in BICOP_NS:
            true_cop = tv.Bicop.from_family(tv_name, parameters=torch.tensor(true_params, dtype=torch.float64))
            obs_th = true_cop.simulate(n).clamp(1e-10, 1.0 - 1e-10)
            obs_np = np.asfortranarray(obs_th.detach().cpu().numpy())

            ctr_tv = FitControlsBicop(family_set=[getattr(BicopFamily, tv_name)], parametric_method=method)
            ctr_pv = pv.FitControlsBicop(
                family_set=[getattr(pv.BicopFamily, tv_name)],
                parametric_method=method,
            )

            def fit_tv():
                b = tv.Bicop.from_family(tv_name)
                b.fit(obs_th, controls=ctr_tv)
                return b

            def fit_tvcl():
                return tvcl_bicop.bcp_from_obs(
                    obs_th,
                    mtd_fit=method,
                    mtd_sel="aic",
                    tpl_fam=(tvcl_name,),
                    topk=1,
                )

            def fit_pv():
                b = pv.Bicop(family=getattr(pv.BicopFamily, tv_name))
                b.fit(obs_np, controls=ctr_pv)
                return b

            t_tv = _time_median(fit_tv, BICOP_REPEATS, warmup=1)
            t_tvcl = _time_median(fit_tvcl, BICOP_REPEATS, warmup=1)
            t_pv = _time_median(fit_pv, BICOP_REPEATS, warmup=1)

            out[tv_name][str(n)] = {
                "torchvine_ms": round(t_tv * 1000.0, 3),
                "torchvinecopulib_ms": round(t_tvcl * 1000.0, 3),
                "pyvinecopulib_ms": round(t_pv * 1000.0, 3),
                "speedup_tv_over_tvcl": round(_speedup(t_tvcl, t_tv), 3),
                "speedup_tv_over_pv": round(_speedup(t_pv, t_tv), 3),
            }
    return out


def bench_bicop_family_selection() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    all_tv_fams = [BicopFamily.gaussian, BicopFamily.clayton, BicopFamily.gumbel, BicopFamily.frank, BicopFamily.joe]
    all_pv_fams = [
        pv.BicopFamily.gaussian,
        pv.BicopFamily.clayton,
        pv.BicopFamily.gumbel,
        pv.BicopFamily.frank,
        pv.BicopFamily.joe,
    ]
    all_tvcl_fams = ("Gaussian", "Clayton", "Gumbel", "Frank", "Joe")

    for n in BICOP_NS:
        true_cop = tv.Bicop.from_family("clayton", parameters=torch.tensor([2.0], dtype=torch.float64))
        obs_th = true_cop.simulate(n).clamp(1e-10, 1.0 - 1e-10)
        obs_np = np.asfortranarray(obs_th.detach().cpu().numpy())

        ctr_tv = FitControlsBicop(
            family_set=all_tv_fams,
            parametric_method="mle",
            selection_criterion="aic",
            allow_rotations=True,
        )
        ctr_pv = pv.FitControlsBicop(
            family_set=all_pv_fams,
            parametric_method="mle",
            selection_criterion="aic",
            allow_rotations=True,
        )

        def fit_tv():
            b = tv.Bicop()
            b.select(obs_th, controls=ctr_tv)
            return b

        def fit_tvcl():
            return tvcl_bicop.bcp_from_obs(
                obs_th,
                mtd_fit="mle",
                mtd_sel="aic",
                tpl_fam=all_tvcl_fams,
                topk=2,
            )

        def fit_pv():
            b = pv.Bicop()
            b.select(obs_np, controls=ctr_pv)
            return b

        t_tv = _time_median(fit_tv, BICOP_REPEATS, warmup=1)
        t_tvcl = _time_median(fit_tvcl, BICOP_REPEATS, warmup=1)
        t_pv = _time_median(fit_pv, BICOP_REPEATS, warmup=1)

        out[str(n)] = {
            "torchvine_ms": round(t_tv * 1000.0, 3),
            "torchvinecopulib_ms": round(t_tvcl * 1000.0, 3),
            "pyvinecopulib_ms": round(t_pv * 1000.0, 3),
            "speedup_tv_over_tvcl": round(_speedup(t_tvcl, t_tv), 3),
            "speedup_tv_over_pv": round(_speedup(t_pv, t_tv), 3),
        }
    return out


def _vine_controls_tv(method: str, trunc: int) -> FitControlsVinecop:
    return FitControlsVinecop(
        family_set=[
            BicopFamily.indep,
            BicopFamily.gaussian,
            BicopFamily.clayton,
            BicopFamily.gumbel,
            BicopFamily.frank,
            BicopFamily.joe,
        ],
        parametric_method=method,
        selection_criterion="aic",
        tree_criterion="tau",
        trunc_lvl=trunc,
        threshold=0.0,
        select_families=True,
        allow_rotations=True,
    )


def _vine_controls_pv(method: str, trunc: int) -> pv.FitControlsVinecop:
    return pv.FitControlsVinecop(
        family_set=[
            pv.BicopFamily.indep,
            pv.BicopFamily.gaussian,
            pv.BicopFamily.clayton,
            pv.BicopFamily.gumbel,
            pv.BicopFamily.frank,
            pv.BicopFamily.joe,
        ],
        parametric_method=method,
        selection_criterion="aic",
        tree_criterion="tau",
        trunc_lvl=trunc,
        threshold=0.0,
        select_families=True,
        allow_rotations=True,
    )


def bench_vine(method: str) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    tvcl_fams = ("Independent", "Gaussian", "Clayton", "Gumbel", "Frank", "Joe")
    for d in VINE_DIMS:
        key = f"d={d}"
        out[key] = {}
        for n in VINE_NS:
            obs_th = torch.rand((n, d), dtype=torch.float64).clamp(1e-10, 1.0 - 1e-10)
            obs_np = np.asfortranarray(obs_th.detach().cpu().numpy())
            trunc = min(3, d - 1)
            ctr_tv = _vine_controls_tv(method, trunc)
            ctr_pv = _vine_controls_pv(method, trunc)

            def sel_tv():
                v = tv.Vinecop.from_dimension(d)
                v.select(obs_th, controls=ctr_tv)
                return v

            def sel_tvcl():
                return tvcl_vinecop.vcp_from_obs(
                    obs_th,
                    is_Dissmann=True,
                    mtd_vine="rvine",
                    mtd_fit=method,
                    mtd_sel="aic",
                    tpl_fam=tvcl_fams,
                    topk=1,
                    thresh_trunc=0.0,
                )

            def sel_pv():
                v = pv.Vinecop(d=d)
                v.select(obs_np, controls=ctr_pv)
                return v

            t_tv = _time_median(sel_tv, VINE_REPEATS, warmup=1)
            t_tvcl = _time_median(sel_tvcl, VINE_REPEATS, warmup=1)
            t_pv = _time_median(sel_pv, VINE_REPEATS, warmup=1)

            out[key][str(n)] = {
                "torchvine_ms": round(t_tv * 1000.0, 3),
                "torchvinecopulib_ms": round(t_tvcl * 1000.0, 3),
                "pyvinecopulib_ms": round(t_pv * 1000.0, 3),
                "speedup_tv_over_tvcl": round(_speedup(t_tvcl, t_tv), 3),
                "speedup_tv_over_pv": round(_speedup(t_pv, t_tv), 3),
            }
    return out


def _flatten(results: dict[str, Any]) -> list[FlatPoint]:
    pts: list[FlatPoint] = []
    for section, section_data in results.items():
        if section in {"meta", "summary"}:
            continue
        if not isinstance(section_data, dict):
            continue
        for subgroup, rec in section_data.items():
            if isinstance(rec, dict) and "torchvine_ms" in rec:
                pts.append(
                    FlatPoint(
                        section=section,
                        subgroup="-",
                        x=str(subgroup),
                        torchvine_ms=float(rec["torchvine_ms"]),
                        torchvinecopulib_ms=float(rec["torchvinecopulib_ms"]),
                        pyvinecopulib_ms=float(rec["pyvinecopulib_ms"]),
                        speedup_tv_over_tvcl=float(rec["speedup_tv_over_tvcl"]),
                        speedup_tv_over_pv=float(rec["speedup_tv_over_pv"]),
                    )
                )
            elif isinstance(rec, dict):
                for x, rr in rec.items():
                    if isinstance(rr, dict) and "torchvine_ms" in rr:
                        pts.append(
                            FlatPoint(
                                section=section,
                                subgroup=str(subgroup),
                                x=str(x),
                                torchvine_ms=float(rr["torchvine_ms"]),
                                torchvinecopulib_ms=float(rr["torchvinecopulib_ms"]),
                                pyvinecopulib_ms=float(rr["pyvinecopulib_ms"]),
                                speedup_tv_over_tvcl=float(rr["speedup_tv_over_tvcl"]),
                                speedup_tv_over_pv=float(rr["speedup_tv_over_pv"]),
                            )
                        )
    return pts


def _gmean(vals: list[float]) -> float:
    xs = [v for v in vals if v > 0 and math.isfinite(v)]
    if not xs:
        return float("nan")
    t = torch.tensor(xs, dtype=torch.float64)
    return float(torch.exp(torch.log(t).mean()).item())


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _bar_nested_sharedy(
    nested: dict[str, dict[str, dict[str, float]]],
    *,
    title: str,
    filename: str,
    x_label: str,
    group_label: str,
) -> None:
    groups = list(nested.keys())
    xvals = sorted(list(next(iter(nested.values())).keys()), key=lambda x: int(x))
    fig, axes = plt.subplots(1, len(xvals), figsize=(4.3 * len(xvals), 5.2), sharey=True)
    if len(xvals) == 1:
        axes = [axes]
    for i, xv in enumerate(xvals):
        ax = axes[i]
        tv = [nested[g][xv]["torchvine_ms"] for g in groups]
        tvcl = [nested[g][xv]["torchvinecopulib_ms"] for g in groups]
        pv = [nested[g][xv]["pyvinecopulib_ms"] for g in groups]
        x = np.arange(len(groups))
        w = 0.25
        ax.bar(x - w, pv, width=w, color=COLORS["pyvinecopulib"], label="pyvinecopulib")
        ax.bar(x, tvcl, width=w, color=COLORS["torchvinecopulib"], label="torchvinecopulib")
        ax.bar(x + w, tv, width=w, color=COLORS["torchvine"], label="torchvine")
        ax.set_xticks(x, groups, rotation=40, ha="right")
        ax.set_title(f"{x_label}={xv}")
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.25)
        if i == 0:
            ax.set_ylabel("runtime (ms, log scale)")
            ax.legend(fontsize=8)
    fig.suptitle(f"{title} ({group_label} x {x_label}, shared y-axis)")
    _save(fig, filename)


def _bar_flat_sharedy(
    flat: dict[str, dict[str, float]],
    *,
    title: str,
    filename: str,
    x_label: str,
) -> None:
    xvals = sorted(list(flat.keys()), key=lambda x: int(x))
    tv = [flat[x]["torchvine_ms"] for x in xvals]
    tvcl = [flat[x]["torchvinecopulib_ms"] for x in xvals]
    pv = [flat[x]["pyvinecopulib_ms"] for x in xvals]
    x = np.arange(len(xvals))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - w, pv, width=w, color=COLORS["pyvinecopulib"], label="pyvinecopulib")
    ax.bar(x, tvcl, width=w, color=COLORS["torchvinecopulib"], label="torchvinecopulib")
    ax.bar(x + w, tv, width=w, color=COLORS["torchvine"], label="torchvine")
    ax.set_xticks(x, xvals)
    ax.set_xlabel(x_label)
    ax.set_ylabel("runtime (ms, log scale)")
    ax.set_yscale("log")
    ax.set_title(f"{title} (shared y-axis)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    _save(fig, filename)


def plot_advanced(results: dict[str, Any]) -> list[str]:
    made: list[str] = []
    pts = _flatten(results)

    # 1-5) Bar charts only (no heatmaps), all with shared y-axis within figure.
    _bar_nested_sharedy(
        results["bicop_mle_fit"],
        title="Bicop MLE Fit",
        filename="bar_bicop_mle_fit_sharedy.png",
        x_label="n",
        group_label="family",
    )
    made.append("bar_bicop_mle_fit_sharedy.png")

    _bar_nested_sharedy(
        results["bicop_itau_fit"],
        title="Bicop itau Fit",
        filename="bar_bicop_itau_fit_sharedy.png",
        x_label="n",
        group_label="family",
    )
    made.append("bar_bicop_itau_fit_sharedy.png")

    _bar_flat_sharedy(
        results["bicop_family_selection"],
        title="Bicop Family Selection",
        filename="bar_bicop_family_selection_sharedy.png",
        x_label="n",
    )
    made.append("bar_bicop_family_selection_sharedy.png")

    _bar_nested_sharedy(
        results["vine_mle_select_fit"],
        title="Vine MLE Select+Fit",
        filename="bar_vine_mle_select_fit_sharedy.png",
        x_label="n",
        group_label="dimension",
    )
    made.append("bar_vine_mle_select_fit_sharedy.png")

    _bar_nested_sharedy(
        results["vine_itau_select_fit"],
        title="Vine itau Select+Fit",
        filename="bar_vine_itau_select_fit_sharedy.png",
        x_label="n",
        group_label="dimension",
    )
    made.append("bar_vine_itau_select_fit_sharedy.png")

    # 6) Scaling curves (log-log median runtime by section), shared y-axis.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    sections = ["bicop_mle_fit", "bicop_itau_fit", "bicop_family_selection", "vine_mle_select_fit", "vine_itau_select_fit"]
    for i, lib in enumerate(["torchvine_ms", "torchvinecopulib_ms", "pyvinecopulib_ms"]):
        ax = axes[i]
        for sec in sections:
            sub = [p for p in pts if p.section == sec]
            x_num = np.array([int(p.x) for p in sub], dtype=float)
            y = np.array([getattr(p, lib) for p in sub], dtype=float)
            # aggregate by x
            uniq = sorted(set(x_num.tolist()))
            yy = [float(np.median(y[x_num == u])) for u in uniq]
            ax.plot(uniq, yy, "o-", label=sec)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(lib.replace("_ms", ""))
        ax.set_xlabel("n")
        ax.set_ylabel("runtime (ms)")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc="best")
    _save(fig, "adv_scaling_curves_loglog.png")
    made.append("adv_scaling_curves_loglog.png")

    return made


def compute_summary(results: dict[str, Any]) -> dict[str, Any]:
    pts = _flatten(results)
    s_tvcl = [p.speedup_tv_over_tvcl for p in pts]
    s_pv = [p.speedup_tv_over_pv for p in pts]
    wins_all = sum(1 for p in pts if p.torchvine_ms <= p.torchvinecopulib_ms and p.torchvine_ms <= p.pyvinecopulib_ms)

    by_section: dict[str, dict[str, float]] = {}
    for s in sorted(set(p.section for p in pts)):
        sub = [p for p in pts if p.section == s]
        by_section[s] = {
            "count": float(len(sub)),
            "gmean_tv_over_tvcl": round(_gmean([p.speedup_tv_over_tvcl for p in sub]), 3),
            "gmean_tv_over_pv": round(_gmean([p.speedup_tv_over_pv for p in sub]), 3),
            "win_rate_fastest_all3_pct": round(
                100.0
                * sum(
                    1
                    for p in sub
                    if p.torchvine_ms <= p.torchvinecopulib_ms and p.torchvine_ms <= p.pyvinecopulib_ms
                )
                / max(1, len(sub)),
                2,
            ),
        }

    return {
        "num_points": len(pts),
        "gmean_speedup_tv_over_torchvinecopulib": round(_gmean(s_tvcl), 3),
        "gmean_speedup_tv_over_pyvinecopulib": round(_gmean(s_pv), 3),
        "median_speedup_tv_over_torchvinecopulib": round(float(np.median(np.array(s_tvcl))), 3),
        "median_speedup_tv_over_pyvinecopulib": round(float(np.median(np.array(s_pv))), 3),
        "min_speedup_tv_over_torchvinecopulib": round(float(np.min(np.array(s_tvcl))), 3),
        "max_speedup_tv_over_torchvinecopulib": round(float(np.max(np.array(s_tvcl))), 3),
        "min_speedup_tv_over_pyvinecopulib": round(float(np.min(np.array(s_pv))), 3),
        "max_speedup_tv_over_pyvinecopulib": round(float(np.max(np.array(s_pv))), 3),
        "torchvine_fastest_all3_win_rate_pct": round(100.0 * wins_all / max(1, len(pts)), 2),
        "by_section": by_section,
    }


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    opt_mod = importlib.import_module("torchvine.optimize")
    tv_mod = importlib.import_module("torchvine")
    torchmin_mod = importlib.import_module("torchmin")

    results: dict[str, Any] = {
        "meta": {
            "torch_version": torch.__version__,
            "torchvine_version": getattr(tv_mod, "__version__", "unknown"),
            "pyvinecopulib_version": getattr(pv, "__version__", "unknown"),
            "torchvinecopulib_version": "2024.10.1",
            "torchmin_version": getattr(torchmin_mod, "__version__", "unknown"),
            "strict_torchmin": bool(opt_mod.torchmin_strict_enabled()),
            "torchvine_module_path": str(Path(tv_mod.__file__).resolve()),
            "torchvine_optimize_path": str(Path(opt_mod.__file__).resolve()),
            "torchmin_module_path": str(Path(torchmin_mod.__file__).resolve()),
            "seed": 42,
            "num_threads": 1,
            "bicop_sample_sizes": BICOP_NS,
            "vine_dims": VINE_DIMS,
            "vine_sample_sizes": VINE_NS,
            "bicop_repeats": BICOP_REPEATS,
            "vine_repeats": VINE_REPEATS,
        }
    }

    print("Running bicop MLE fit...")
    results["bicop_mle_fit"] = bench_bicop_fit("mle")

    print("Running bicop itau fit...")
    results["bicop_itau_fit"] = bench_bicop_fit("itau")

    print("Running bicop family selection...")
    results["bicop_family_selection"] = bench_bicop_family_selection()

    print("Running vine MLE select+fit...")
    results["vine_mle_select_fit"] = bench_vine("mle")

    print("Running vine itau select+fit...")
    results["vine_itau_select_fit"] = bench_vine("itau")

    results["summary"] = compute_summary(results)
    plot_files = plot_advanced(results)
    results["plot_files"] = plot_files

    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["summary"], indent=2))
    print(f"Saved benchmark JSON: {OUT_JSON}")
    print("Saved plots:")
    for p in plot_files:
        print(f"  - {OUT_DIR / p}")


if __name__ == "__main__":
    main()
