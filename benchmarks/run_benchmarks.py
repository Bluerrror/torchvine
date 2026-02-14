"""Benchmark torchvine vs pyvinecopulib speed for Bicop and Vinecop operations.

Generates comparison plots saved to benchmarks/ folder.
"""
import time
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyvinecopulib as pv
import torchvine as tv


def _pv_params(params):
    return np.array(params, dtype=np.float64).reshape(-1, 1)


# ── Bicop benchmarks ─────────────────────────────────────────────────────────

FAMILIES = {
    "gaussian":  [0.6],
    "clayton":   [2.5],
    "gumbel":    [2.0],
    "frank":     [5.0],
    "joe":       [2.5],
    "bb1":       [0.5, 1.5],
    "bb7":       [2.0, 1.0],
    "tawn":      [0.5, 0.5, 2.0],
}

SAMPLE_SIZES = [100, 500, 1_000, 10_000, 100_000]
N_REPEATS = 5


def bench_bicop_pdf():
    """Benchmark Bicop.pdf() for all families and sample sizes."""
    results = {}
    for fam_name, params in FAMILIES.items():
        pv_c = pv.Bicop(getattr(pv.BicopFamily, fam_name), 0, _pv_params(params))
        tv_c = tv.Bicop(fam_name, parameters=torch.tensor(params, dtype=torch.float64))

        pv_times = []
        tv_times = []
        for n in SAMPLE_SIZES:
            rng = np.random.default_rng(42)
            u_np = np.asfortranarray(rng.uniform(0.02, 0.98, (n, 2)))
            u_th = torch.tensor(u_np, dtype=torch.float64)

            # Warmup
            pv_c.pdf(u_np)
            tv_c.pdf(u_th)

            # pyvinecopulib
            t0 = time.perf_counter()
            for _ in range(N_REPEATS):
                pv_c.pdf(u_np)
            pv_t = (time.perf_counter() - t0) / N_REPEATS

            # torchvine
            t0 = time.perf_counter()
            for _ in range(N_REPEATS):
                tv_c.pdf(u_th)
            tv_t = (time.perf_counter() - t0) / N_REPEATS

            pv_times.append(pv_t * 1000)
            tv_times.append(tv_t * 1000)

        results[fam_name] = {"pv": pv_times, "tv": tv_times}
    return results


def bench_bicop_hfunc():
    """Benchmark Bicop.hfunc1() for all families."""
    results = {}
    for fam_name, params in FAMILIES.items():
        pv_c = pv.Bicop(getattr(pv.BicopFamily, fam_name), 0, _pv_params(params))
        tv_c = tv.Bicop(fam_name, parameters=torch.tensor(params, dtype=torch.float64))

        pv_times = []
        tv_times = []
        for n in SAMPLE_SIZES:
            rng = np.random.default_rng(42)
            u_np = np.asfortranarray(rng.uniform(0.02, 0.98, (n, 2)))
            u_th = torch.tensor(u_np, dtype=torch.float64)

            pv_c.hfunc1(u_np)
            tv_c.hfunc1(u_th)

            t0 = time.perf_counter()
            for _ in range(N_REPEATS):
                pv_c.hfunc1(u_np)
            pv_t = (time.perf_counter() - t0) / N_REPEATS

            t0 = time.perf_counter()
            for _ in range(N_REPEATS):
                tv_c.hfunc1(u_th)
            tv_t = (time.perf_counter() - t0) / N_REPEATS

            pv_times.append(pv_t * 1000)
            tv_times.append(tv_t * 1000)

        results[fam_name] = {"pv": pv_times, "tv": tv_times}
    return results


def bench_bicop_simulate():
    """Benchmark Bicop.simulate()."""
    results = {}
    sim_sizes = [100, 500, 1_000, 5_000, 10_000]
    for fam_name, params in FAMILIES.items():
        pv_c = pv.Bicop(getattr(pv.BicopFamily, fam_name), 0, _pv_params(params))
        tv_c = tv.Bicop(fam_name, parameters=torch.tensor(params, dtype=torch.float64))

        pv_times = []
        tv_times = []
        for n in sim_sizes:
            pv_c.simulate(n)
            tv_c.simulate(n)

            t0 = time.perf_counter()
            for _ in range(N_REPEATS):
                pv_c.simulate(n)
            pv_t = (time.perf_counter() - t0) / N_REPEATS

            t0 = time.perf_counter()
            for _ in range(N_REPEATS):
                tv_c.simulate(n)
            tv_t = (time.perf_counter() - t0) / N_REPEATS

            pv_times.append(pv_t * 1000)
            tv_times.append(tv_t * 1000)

        results[fam_name] = {"pv": pv_times, "tv": tv_times}
    return results


# ── Vinecop benchmarks ───────────────────────────────────────────────────────

def bench_vinecop_fit():
    """Benchmark Vinecop.select() for different dimensions."""
    dims = [3, 4, 5, 6]
    n = 500
    results = {"pv": [], "tv": []}

    for d in dims:
        rng = np.random.default_rng(42)
        u_np = np.asfortranarray(rng.uniform(0.01, 0.99, (n, d)))
        u_th = torch.tensor(u_np, dtype=torch.float64)

        pv_fams = [f for f in pv.BicopFamily if f != pv.BicopFamily.student]

        # Warmup
        pv_v = pv.Vinecop(d=d)
        pv_v.select(u_np, controls=pv.FitControlsVinecop(
            family_set=pv_fams, parametric_method="itau"))

        tv_v = tv.Vinecop.from_dimension(d)
        tv_v.select(u_th, controls=tv.FitControlsVinecop(parametric_method="itau"))

        # pyvinecopulib
        t0 = time.perf_counter()
        for _ in range(3):
            pv_v2 = pv.Vinecop(d=d)
            pv_v2.select(u_np, controls=pv.FitControlsVinecop(
                family_set=pv_fams, parametric_method="itau"))
        pv_t = (time.perf_counter() - t0) / 3

        # torchvine
        t0 = time.perf_counter()
        for _ in range(3):
            tv_v2 = tv.Vinecop.from_dimension(d)
            tv_v2.select(u_th, controls=tv.FitControlsVinecop(parametric_method="itau"))
        tv_t = (time.perf_counter() - t0) / 3

        results["pv"].append(pv_t * 1000)
        results["tv"].append(tv_t * 1000)

    results["dims"] = dims
    return results


def bench_vinecop_pdf():
    """Benchmark Vinecop.pdf() for different sample sizes."""
    d = 4
    sizes = [100, 500, 1_000, 5_000]

    # Build a fitted vine
    rng = np.random.default_rng(42)
    u_fit = np.asfortranarray(rng.uniform(0.01, 0.99, (500, d)))
    u_fit_th = torch.tensor(u_fit, dtype=torch.float64)

    pv_fams = [f for f in pv.BicopFamily if f != pv.BicopFamily.student]
    pv_v = pv.Vinecop(d=d)
    pv_v.select(u_fit, controls=pv.FitControlsVinecop(
        family_set=pv_fams, parametric_method="itau"))

    tv_v = tv.Vinecop.from_dimension(d)
    tv_v.select(u_fit_th, controls=tv.FitControlsVinecop(parametric_method="itau"))

    results = {"pv": [], "tv": []}
    for n in sizes:
        u_np = np.asfortranarray(rng.uniform(0.02, 0.98, (n, d)))
        u_th = torch.tensor(u_np, dtype=torch.float64)

        pv_v.pdf(u_np)
        tv_v.pdf(u_th)

        t0 = time.perf_counter()
        for _ in range(N_REPEATS):
            pv_v.pdf(u_np)
        pv_t = (time.perf_counter() - t0) / N_REPEATS

        t0 = time.perf_counter()
        for _ in range(N_REPEATS):
            tv_v.pdf(u_th)
        tv_t = (time.perf_counter() - t0) / N_REPEATS

        results["pv"].append(pv_t * 1000)
        results["tv"].append(tv_t * 1000)

    results["sizes"] = sizes
    return results


def bench_vinecop_simulate():
    """Benchmark Vinecop.simulate()."""
    d = 4
    sizes = [100, 500, 1_000, 5_000]

    pv_v = pv.Vinecop(d=d)
    tv_v = tv.Vinecop.from_dimension(d)

    results = {"pv": [], "tv": []}
    for n in sizes:
        pv_v.simulate(n)
        tv_v.simulate(n)

        t0 = time.perf_counter()
        for _ in range(N_REPEATS):
            pv_v.simulate(n)
        pv_t = (time.perf_counter() - t0) / N_REPEATS

        t0 = time.perf_counter()
        for _ in range(N_REPEATS):
            tv_v.simulate(n)
        tv_t = (time.perf_counter() - t0) / N_REPEATS

        results["pv"].append(pv_t * 1000)
        results["tv"].append(tv_t * 1000)

    results["sizes"] = sizes
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_bicop_comparison(results, title, filename):
    """Plot grouped bar chart of torchvine vs pyvinecopulib times."""
    fam_names = list(results.keys())
    n_fams = len(fam_names)
    n_sizes = len(SAMPLE_SIZES)

    fig, axes = plt.subplots(1, n_sizes, figsize=(4 * n_sizes, 5), sharey=False)
    if n_sizes == 1:
        axes = [axes]

    for idx, n in enumerate(SAMPLE_SIZES):
        ax = axes[idx]
        pv_vals = [results[f]["pv"][idx] for f in fam_names]
        tv_vals = [results[f]["tv"][idx] for f in fam_names]

        x = np.arange(n_fams)
        w = 0.35
        ax.bar(x - w / 2, pv_vals, w, label="pyvinecopulib", color="#2196F3", alpha=0.85)
        ax.bar(x + w / 2, tv_vals, w, label="torchvine", color="#FF9800", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(fam_names, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"n={n:,}", fontsize=10)
        ax.set_ylabel("Time (ms)" if idx == 0 else "")
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_scaling(results, x_vals, x_label, title, filename):
    """Plot line chart of scaling with sample size or dimension."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x_vals, results["pv"], "o-", label="pyvinecopulib", color="#2196F3",
            linewidth=2, markersize=6)
    ax.plot(x_vals, results["tv"], "s-", label="torchvine", color="#FF9800",
            linewidth=2, markersize=6)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Time (ms)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_summary_bar(all_results, filename):
    """Summary bar chart: average speedup ratio across all benchmarks."""
    labels = []
    ratios = []

    for name, res in all_results.items():
        if isinstance(res, dict) and "pv" in res and "tv" in res:
            pv_avg = np.mean(res["pv"])
            tv_avg = np.mean(res["tv"])
            labels.append(name)
            ratios.append(pv_avg / tv_avg if tv_avg > 0 else 1.0)
        elif isinstance(res, dict):
            for fam, vals in res.items():
                if isinstance(vals, dict) and "pv" in vals:
                    pv_avg = np.mean(vals["pv"])
                    tv_avg = np.mean(vals["tv"])
                    labels.append(f"{name}:{fam}")
                    ratios.append(pv_avg / tv_avg if tv_avg > 0 else 1.0)

    fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.4)))
    colors = ["#4CAF50" if r >= 1 else "#f44336" for r in ratios]
    y = np.arange(len(labels))
    ax.barh(y, ratios, color=colors, alpha=0.8)
    ax.axvline(x=1.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Speedup Ratio (pyvinecopulib / torchvine)", fontsize=11)
    ax.set_title("Speed Comparison: torchvine vs pyvinecopulib\n(>1 = torchvine faster)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")


def main():
    import os
    os.makedirs("benchmarks", exist_ok=True)

    all_results = {}

    print("Running Bicop PDF benchmark...")
    pdf_res = bench_bicop_pdf()
    all_results["bicop_pdf"] = pdf_res
    plot_bicop_comparison(pdf_res, "Bicop.pdf() Speed Comparison",
                          "benchmarks/bicop_pdf_benchmark.png")

    print("Running Bicop hfunc benchmark...")
    hfunc_res = bench_bicop_hfunc()
    all_results["bicop_hfunc"] = hfunc_res
    plot_bicop_comparison(hfunc_res, "Bicop.hfunc1() Speed Comparison",
                          "benchmarks/bicop_hfunc_benchmark.png")

    print("Running Bicop simulate benchmark...")
    sim_res = bench_bicop_simulate()
    all_results["bicop_sim"] = sim_res
    plot_bicop_comparison(sim_res, "Bicop.simulate() Speed Comparison",
                          "benchmarks/bicop_simulate_benchmark.png")

    print("Running Vinecop fit benchmark...")
    fit_res = bench_vinecop_fit()
    all_results["vinecop_fit"] = fit_res
    plot_scaling(fit_res, fit_res["dims"], "Dimension",
                 "Vinecop.select() Speed Comparison",
                 "benchmarks/vinecop_fit_benchmark.png")

    print("Running Vinecop PDF benchmark...")
    vpdf_res = bench_vinecop_pdf()
    all_results["vinecop_pdf"] = vpdf_res
    plot_scaling(vpdf_res, vpdf_res["sizes"], "Sample Size",
                 "Vinecop.pdf() Speed Comparison",
                 "benchmarks/vinecop_pdf_benchmark.png")

    print("Running Vinecop simulate benchmark...")
    vsim_res = bench_vinecop_simulate()
    all_results["vinecop_sim"] = vsim_res
    plot_scaling(vsim_res, vsim_res["sizes"], "Sample Size",
                 "Vinecop.simulate() Speed Comparison",
                 "benchmarks/vinecop_simulate_benchmark.png")

    print("\nGenerating summary plot...")
    plot_summary_bar(all_results, "benchmarks/summary_speedup.png")

    # Save raw results
    serializable = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            serializable[k] = {}
            for k2, v2 in v.items():
                if isinstance(v2, dict):
                    serializable[k][k2] = {k3: [float(x) for x in v3] if isinstance(v3, list) else v3
                                           for k3, v3 in v2.items()}
                elif isinstance(v2, list):
                    serializable[k][k2] = [float(x) if isinstance(x, (int, float, np.floating)) else x for x in v2]
                else:
                    serializable[k][k2] = v2

    with open("benchmarks/results.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print("  Saved: benchmarks/results.json")

    print("\nDone! All benchmark results saved to benchmarks/")


if __name__ == "__main__":
    main()
