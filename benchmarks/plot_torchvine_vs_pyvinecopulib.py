"""Generate extended benchmark plots for torchvine vs pyvinecopulib."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


IN_PATH = Path("benchmarks/results_torchvine_vs_pyvinecopulib.json")
OUT_DIR = Path("benchmarks")


def _load() -> dict:
    with IN_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save(fig: plt.Figure, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_bicop_eval_lines(data: dict):
    fams = list(data["bicop_eval_grid"].keys())
    ns = [int(v) for v in data["meta"]["eval_ns"]]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for i, fam in enumerate(fams):
        ax = axes[i // 3, i % 3]
        pdf_s = [data["bicop_eval_grid"][fam][str(n)]["pdf_speedup_torchvine_over_pyvinecopulib"] for n in ns]
        h1_s = [data["bicop_eval_grid"][fam][str(n)]["hfunc1_speedup_torchvine_over_pyvinecopulib"] for n in ns]
        ax.plot(ns, pdf_s, "o-", label="pdf", color="#1f77b4")
        ax.plot(ns, h1_s, "s-", label="hfunc1", color="#ff7f0e")
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_title(fam)
        ax.set_xlabel("n samples")
        ax.set_ylabel("speedup")
    axes[0, 0].legend()
    fig.suptitle("Bicop Eval Speedup vs Sample Size")
    _save(fig, "plot_bicop_eval_lines.png")


def plot_bicop_fit_lines(data: dict):
    fams = list(data["bicop_fit_grid"].keys())
    ns = [int(v) for v in data["meta"]["fit_ns"]]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for i, fam in enumerate(fams):
        ax = axes[i // 3, i % 3]
        s = [data["bicop_fit_grid"][fam][str(n)]["fit_mle_speedup_torchvine_over_pyvinecopulib"] for n in ns]
        ax.plot(ns, s, "o-", color="#2ca02c")
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_title(fam)
        ax.set_xlabel("n fit samples")
        ax.set_ylabel("speedup")
    fig.suptitle("Bicop MLE Fit Speedup vs Sample Size")
    _save(fig, "plot_bicop_fit_lines.png")


def plot_bicop_eval_raw_side_by_side(data: dict):
    fams = list(data["bicop_eval_grid"].keys())
    ns = [int(v) for v in data["meta"]["eval_ns"]]
    x = np.arange(len(ns))
    w = 0.36

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=False)
    for i, fam in enumerate(fams):
        ax = axes[i // 3, i % 3]
        tv_pdf = [data["bicop_eval_grid"][fam][str(n)]["pdf_ms_torchvine"] for n in ns]
        pv_pdf = [data["bicop_eval_grid"][fam][str(n)]["pdf_ms_pyvinecopulib"] for n in ns]
        tv_h = [data["bicop_eval_grid"][fam][str(n)]["hfunc1_ms_torchvine"] for n in ns]
        pv_h = [data["bicop_eval_grid"][fam][str(n)]["hfunc1_ms_pyvinecopulib"] for n in ns]
        ax.bar(x - w / 2, tv_pdf, w, color="#1f77b4", alpha=0.8, label="tv pdf")
        ax.bar(x + w / 2, pv_pdf, w, color="#d62728", alpha=0.8, label="pv pdf")
        ax.plot(x, tv_h, "o--", color="#2ca02c", linewidth=1.4, label="tv hfunc1")
        ax.plot(x, pv_h, "s--", color="#ff7f0e", linewidth=1.4, label="pv hfunc1")
        ax.set_yscale("log")
        ax.set_xticks(x, [str(n) for n in ns])
        ax.set_title(fam)
        ax.set_xlabel("n")
        ax.set_ylabel("ms (log)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Bicop Eval Raw Times (Bars: pdf, Lines: hfunc1)")
    _save(fig, "plot_bicop_eval_raw_side_by_side.png")


def plot_bicop_fit_raw_side_by_side(data: dict):
    fams = list(data["bicop_fit_grid"].keys())
    ns = [int(v) for v in data["meta"]["fit_ns"]]
    x = np.arange(len(ns))
    w = 0.36

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=False)
    for i, fam in enumerate(fams):
        ax = axes[i // 3, i % 3]
        tv = [data["bicop_fit_grid"][fam][str(n)]["fit_mle_ms_torchvine"] for n in ns]
        pv = [data["bicop_fit_grid"][fam][str(n)]["fit_mle_ms_pyvinecopulib"] for n in ns]
        ax.bar(x - w / 2, tv, w, color="#1f77b4", label="torchvine")
        ax.bar(x + w / 2, pv, w, color="#d62728", label="pyvinecopulib")
        ax.set_yscale("log")
        ax.set_xticks(x, [str(n) for n in ns])
        ax.set_title(fam)
        ax.set_xlabel("n fit")
        ax.set_ylabel("ms (log)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Bicop Fit Raw Times (MLE)")
    _save(fig, "plot_bicop_fit_raw_side_by_side.png")


def _vine_matrix(data: dict, metric_key: str) -> tuple[np.ndarray, list[int], list[int]]:
    dims = [int(d.split("=")[1]) for d in data["vinecop_grid"].keys()]
    ns = [int(v) for v in data["meta"]["vine_train_ns"]]
    mat = np.zeros((len(dims), len(ns)), dtype=float)
    for i, d in enumerate(dims):
        rec_d = data["vinecop_grid"][f"d={d}"]
        for j, n in enumerate(ns):
            mat[i, j] = float(rec_d[str(n)][metric_key])
    return mat, dims, ns


def plot_vine_heatmaps(data: dict):
    configs = [
        ("select_mle_speedup_torchvine_over_pyvinecopulib", "Select Speedup"),
        ("pdf_speedup_torchvine_over_pyvinecopulib", "PDF Speedup"),
        ("simulate_speedup_torchvine_over_pyvinecopulib", "Simulate Speedup"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (key, title) in zip(axes, configs):
        mat, dims, ns = _vine_matrix(data, key)
        im = ax.imshow(np.log10(mat), aspect="auto", cmap="RdYlGn", vmin=-2, vmax=2)
        ax.set_xticks(np.arange(len(ns)), [str(n) for n in ns])
        ax.set_yticks(np.arange(len(dims)), [str(d) for d in dims])
        ax.set_xlabel("train n")
        ax.set_ylabel("dimension d")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Vinecop Speedup Heatmaps (log10 scale)")
    _save(fig, "plot_vinecop_heatmaps.png")


def plot_vine_raw_times(data: dict):
    dims = [int(d.split("=")[1]) for d in data["vinecop_grid"].keys()]
    ns = [int(v) for v in data["meta"]["vine_train_ns"]]
    dim_to_pick = dims[len(dims) // 2]
    rec = data["vinecop_grid"][f"d={dim_to_pick}"]

    ops = ["select_mle", "pdf", "simulate"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    for i, op in enumerate(ops):
        tv = [rec[str(n)][f"{op}_ms_torchvine"] for n in ns]
        pv = [rec[str(n)][f"{op}_ms_pyvinecopulib"] for n in ns]
        x = np.arange(len(ns))
        w = 0.35
        axes[i].bar(x - w / 2, tv, w, label="torchvine", color="#1f77b4")
        axes[i].bar(x + w / 2, pv, w, label="pyvinecopulib", color="#d62728")
        axes[i].set_xticks(x, [str(n) for n in ns])
        axes[i].set_yscale("log")
        axes[i].set_xlabel("train n")
        axes[i].set_ylabel("ms (log)")
        axes[i].set_title(op)
    axes[0].legend()
    fig.suptitle(f"Vinecop Raw Times at d={dim_to_pick}")
    _save(fig, "plot_vinecop_raw_times_mid_dim.png")


def plot_vine_lines_by_dim(data: dict):
    dims = [int(d.split("=")[1]) for d in data["vinecop_grid"].keys()]
    ns = [int(v) for v in data["meta"]["vine_train_ns"]]
    metrics = [
        ("select_mle_speedup_torchvine_over_pyvinecopulib", "select"),
        ("pdf_speedup_torchvine_over_pyvinecopulib", "pdf"),
        ("simulate_speedup_torchvine_over_pyvinecopulib", "simulate"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=False)
    for ax, (k, lbl) in zip(axes, metrics):
        for n in ns:
            ys = [data["vinecop_grid"][f"d={d}"][str(n)][k] for d in dims]
            ax.plot(dims, ys, "o-", label=f"n={n}")
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.set_xlabel("dimension d")
        ax.set_ylabel("speedup")
        ax.set_title(lbl)
    axes[0].legend(fontsize=8)
    fig.suptitle("Vinecop Speedup by Dimension (per Train Size)")
    _save(fig, "plot_vinecop_lines_by_dim.png")


def plot_global_speedup_hist(data: dict):
    vals = []
    for fam in data["bicop_eval_grid"].values():
        for rec in fam.values():
            vals.extend([
                rec["pdf_speedup_torchvine_over_pyvinecopulib"],
                rec["hfunc1_speedup_torchvine_over_pyvinecopulib"],
            ])
    for fam in data["bicop_fit_grid"].values():
        for rec in fam.values():
            vals.append(rec["fit_mle_speedup_torchvine_over_pyvinecopulib"])
    for dim in data["vinecop_grid"].values():
        for rec in dim.values():
            vals.extend([
                rec["select_mle_speedup_torchvine_over_pyvinecopulib"],
                rec["pdf_speedup_torchvine_over_pyvinecopulib"],
                rec["simulate_speedup_torchvine_over_pyvinecopulib"],
            ])
    arr = np.asarray(vals, dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.hist(np.log10(arr), bins=28, color="#9467bd", alpha=0.9)
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("log10(speedup)")
    ax.set_ylabel("count")
    ax.set_title("Distribution of All Speedup Points")
    _save(fig, "plot_global_speedup_hist.png")


def plot_summary_bar(data: dict):
    g = float(data["summary"]["geometric_mean_speedup_torchvine_over_pyvinecopulib"])
    fig, ax = plt.subplots(figsize=(7, 2))
    ax.barh(["overall"], [g], color="#2ca02c" if g >= 1.0 else "#d62728")
    ax.axvline(1.0, color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("geometric mean speedup (pyvinecopulib / torchvine)")
    ax.set_title("Overall Benchmark Summary")
    ax.text(g, 0, f" {g:.3f}x", va="center", ha="left", fontsize=10, fontweight="bold")
    _save(fig, "plot_summary_speedup.png")


def plot_worst_best_points(data: dict):
    points = []
    for fam, d_fam in data["bicop_eval_grid"].items():
        for n, rec in d_fam.items():
            points.append((f"eval:{fam}:{n}:pdf", rec["pdf_speedup_torchvine_over_pyvinecopulib"]))
            points.append((f"eval:{fam}:{n}:hfunc1", rec["hfunc1_speedup_torchvine_over_pyvinecopulib"]))
    for fam, d_fam in data["bicop_fit_grid"].items():
        for n, rec in d_fam.items():
            points.append((f"fit:{fam}:{n}", rec["fit_mle_speedup_torchvine_over_pyvinecopulib"]))
    for d, d_dim in data["vinecop_grid"].items():
        for n, rec in d_dim.items():
            points.append((f"vine:{d}:{n}:select", rec["select_mle_speedup_torchvine_over_pyvinecopulib"]))
            points.append((f"vine:{d}:{n}:pdf", rec["pdf_speedup_torchvine_over_pyvinecopulib"]))
            points.append((f"vine:{d}:{n}:sim", rec["simulate_speedup_torchvine_over_pyvinecopulib"]))

    points_sorted = sorted(points, key=lambda x: float(x[1]))
    worst = points_sorted[:15]
    best = points_sorted[-15:]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=False)
    yw = np.arange(len(worst))
    yb = np.arange(len(best))
    axes[0].barh(yw, [w[1] for w in worst], color="#d62728")
    axes[0].axvline(1.0, color="#333333", linestyle="--", linewidth=1)
    axes[0].set_yticks(yw, [w[0] for w in worst], fontsize=8)
    axes[0].set_xscale("log")
    axes[0].set_title("Worst 15 speedup points")
    axes[0].set_xlabel("speedup")

    axes[1].barh(yb, [b[1] for b in best], color="#2ca02c")
    axes[1].axvline(1.0, color="#333333", linestyle="--", linewidth=1)
    axes[1].set_yticks(yb, [b[0] for b in best], fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_title("Best 15 speedup points")
    axes[1].set_xlabel("speedup")
    fig.suptitle("Pointwise Speedup Extremes")
    _save(fig, "plot_worst_best_points.png")


def main():
    data = _load()
    plot_bicop_eval_lines(data)
    plot_bicop_fit_lines(data)
    plot_bicop_eval_raw_side_by_side(data)
    plot_bicop_fit_raw_side_by_side(data)
    plot_vine_heatmaps(data)
    plot_vine_raw_times(data)
    plot_vine_lines_by_dim(data)
    plot_global_speedup_hist(data)
    plot_summary_bar(data)
    plot_worst_best_points(data)
    print("saved extended plots to benchmarks/")


if __name__ == "__main__":
    main()
