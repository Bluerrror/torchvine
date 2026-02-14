# torchvine

**Pure-PyTorch vine copula modelling — GPU-ready, differentiable, and fully API-compatible with [pyvinecopulib](https://github.com/vinecopulib/pyvinecopulib).**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

---

## Why torchvine?

| | torchvine | pyvinecopulib |
|---|---|---|
| **Backend** | Pure PyTorch (GPU / CPU) | C++ with Python bindings |
| **Differentiable** | ✅ Autograd-compatible | ❌ |
| **GPU acceleration** | ✅ CUDA tensors | ❌ CPU only |
| **API** | Drop-in replacement | Reference |
| **Copula families** | 13 (full parity) | 13 |

**Zero C/C++ dependencies** — everything is implemented in pure PyTorch, making it easy to install, debug, and extend.

### Supported Copula Families

| Family | Parameters | Type |
|--------|-----------|------|
| Independence | 0 | — |
| Gaussian | 1 (ρ) | Elliptical |
| Student-t | 2 (ρ, ν) | Elliptical |
| Clayton | 1 (θ) | Archimedean |
| Gumbel | 1 (θ) | Archimedean / Extreme-value |
| Frank | 1 (θ) | Archimedean |
| Joe | 1 (θ) | Archimedean |
| BB1 | 2 (θ, δ) | Archimedean |
| BB6 | 2 (θ, δ) | Archimedean |
| BB7 | 2 (θ, δ) | Archimedean |
| BB8 | 2 (θ, δ) | Archimedean |
| Tawn | 3 (ψ₁, ψ₂, θ) | Extreme-value |
| TLL | nonparametric | Kernel-based |

All asymmetric families support rotations (0°, 90°, 180°, 270°).

---

## Installation

```bash
pip install torchvine
```

**From source:**

```bash
git clone https://github.com/Bluerrror/torchvine.git
cd torchvine
pip install -e .
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, matplotlib ≥ 3.5

---

## Quick Start

### Bivariate Copula

```python
import torch
import torchvine as tv

# Create a Gaussian copula with correlation 0.7
cop = tv.Bicop(tv.BicopFamily.gaussian, parameters=torch.tensor([0.7]))
print(cop.str())               # <Bicop> family=gaussian, rotation=0, parameters=[0.7]
print(cop.parameters_to_tau()) # Kendall's tau ≈ 0.494

# Evaluate density and simulate
u = torch.rand(1000, 2, dtype=torch.float64)
pdf_vals = cop.pdf(u)
samples  = cop.simulate(1000)

# Fit from data (automatic family selection)
fitted = tv.Bicop()
fitted.select(samples)
print(fitted.str())
```

### Vine Copula

```python
# Fit a 5-dimensional vine copula
data = torch.rand(500, 5, dtype=torch.float64)
vine = tv.Vinecop(d=5)
vine.select(data, controls=tv.FitControlsVinecop(family_set=tv.parametric))

print(vine.str())
print(f"Log-likelihood: {vine.loglik(data):.2f}")
print(f"AIC: {vine.aic(data):.2f}")

# Simulate and transform
sim = vine.simulate(1000)
pit = vine.rosenblatt(data)      # probability integral transform
```

### Dependence Measures (wdm)

```python
x = torch.randn(1000, dtype=torch.float64)
y = 0.6 * x + 0.8 * torch.randn(1000, dtype=torch.float64)

# All five dependence measures — pure torch, no scipy/numpy
print(tv.kendall_tau(x, y))    # Kendall's tau
print(tv.spearman_rho(x, y))   # Spearman's rho
print(tv.pearson_cor(x, y))    # Pearson correlation
print(tv.blomqvist_beta(x, y)) # Blomqvist's beta
print(tv.hoeffding_d(x, y))    # Hoeffding's D
print(tv.wdm(x, y, "kendall")) # Unified interface
```

### 1-D Kernel Density Estimation

```python
data = torch.randn(500, dtype=torch.float64)
kde = tv.Kde1d()
kde.fit(data)

pts = torch.linspace(-3, 3, 200, dtype=torch.float64)
pdf = kde.pdf(pts)
cdf = kde.cdf(pts)
q   = kde.quantile(torch.tensor([0.025, 0.5, 0.975]))
```

### GPU Acceleration

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
u_gpu = torch.rand(10000, 2, dtype=torch.float64, device=device)

cop = tv.Bicop(tv.BicopFamily.clayton, parameters=torch.tensor([3.0], device=device))
pdf_gpu = cop.pdf(u_gpu)  # runs entirely on GPU
```

---

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `tv.Bicop` | Bivariate copula (create, fit, evaluate, simulate) |
| `tv.Vinecop` | Vine copula model (select, pdf, simulate, rosenblatt) |
| `tv.Kde1d` | 1-D kernel density estimation (fit, pdf, cdf, quantile, simulate) |
| `tv.RVineStructure` | R-vine structure matrix |
| `tv.DVineStructure` | D-vine structure (convenience subclass) |
| `tv.CVineStructure` | C-vine structure (convenience subclass) |
| `tv.FitControlsBicop` | Fitting options for bivariate copulas |
| `tv.FitControlsVinecop` | Fitting options for vine copulas |
| `tv.BicopFamily` | Enum of copula families |

### Bicop Methods

```
Bicop(family, rotation=0, parameters=None, var_types=("c","c"))
  .pdf(u)  .cdf(u)  .loglik(u)  .aic(u)  .bic(u)  .mbic(u)
  .hfunc1(u)  .hfunc2(u)  .hinv1(u)  .hinv2(u)
  .simulate(n)  .fit(data)  .select(data)
  .parameters_to_tau()  .tau_to_parameters(tau)
  .to_json()  .from_json()  .to_file()  .from_file()  .str()
```

### Vinecop Methods

```
Vinecop(d)  |  Vinecop(structure=)  |  Vinecop(matrix=)
  .select(data, controls)  .fit(data)
  .pdf(u)  .loglik(u)  .aic(u)  .bic(u)  .simulate(n)
  .rosenblatt(u)  .inverse_rosenblatt(u)  .cdf(u)
  .truncate(level)  .to_json()  .from_json()  .str()
  .structure  .pair_copulas  .dim  .order  .npars
```

### Dependence Measures

| Function | Description |
|----------|-------------|
| `tv.kendall_tau(x, y)` | Kendall's rank correlation |
| `tv.spearman_rho(x, y)` | Spearman's rank correlation |
| `tv.pearson_cor(x, y)` | Pearson linear correlation |
| `tv.blomqvist_beta(x, y)` | Blomqvist's beta (medial correlation) |
| `tv.hoeffding_d(x, y)` | Hoeffding's D statistic |
| `tv.wdm(x, y, method)` | Unified interface for all measures |

### Visualization

| Function | Description |
|----------|-------------|
| `tv.pairs_copula_data(data)` | Pairs plot with copula density contours and Kendall's τ |

### Utility Functions

| Function | Description |
|----------|-------------|
| `tv.to_pseudo_obs(data)` | Rank-transform to pseudo-observations |
| `tv.simulate_uniform(n, d)` | Uniform random / quasi-random samples |
| `tv.sobol(n, d)` | Sobol quasi-random sequence |
| `tv.ghalton(n, d)` | Generalized Halton-like sequence |

### Family Convenience Lists

```python
tv.one_par        # [gaussian, clayton, gumbel, frank, joe]
tv.two_par        # [student, bb1, bb6, bb7, bb8]
tv.three_par      # [tawn]
tv.parametric     # all parametric families
tv.nonparametric  # [indep, tll]
tv.archimedean    # [clayton, gumbel, frank, joe, bb1, bb6, bb7, bb8]
tv.elliptical     # [gaussian, student]
tv.extreme_value  # [tawn, gumbel]
tv.itau           # families supporting inverse-tau fitting
tv.all            # all families
```

---

## Project Structure

```
torchvine/
├── torchvine/
│   ├── __init__.py          # Public API and exports
│   ├── bicop.py             # Bivariate copula implementation
│   ├── families.py          # BicopFamily enum
│   ├── fit_controls.py      # FitControlsBicop / FitControlsVinecop
│   ├── interpolation.py     # Grid interpolation for TLL
│   ├── kde1d.py             # 1-D kernel density estimation
│   ├── optimize.py          # Parameter optimization (MLE)
│   ├── pair_copuladata.py   # Pairs copula data visualization
│   ├── rvine_structure.py   # R-vine / D-vine / C-vine structures
│   ├── stats.py             # Statistical functions (wdm, tau, rho, etc.)
│   ├── tll_fit.py           # Transformation local likelihood estimator
│   ├── vine_select.py       # Vine structure & family selection
│   └── vinecop.py           # Vine copula model
├── tests/                   # Unit + cross-check tests
├── examples/                # Jupyter notebook tutorials
├── pyproject.toml
├── LICENSE                  # MIT
└── README.md
```

---

## Examples

See the [`examples/`](examples/) directory for Jupyter notebooks:

| Notebook | Topics |
|----------|--------|
| [01 — Getting Started](examples/01_getting_started.ipynb) | Imports, copula basics, simulation, fitting |
| [02 — Bivariate Copulas](examples/02_bivariate_copulas.ipynb) | All families, rotations, Student-t, model selection |
| [03 — Vine Copulas](examples/03_vine_copulas.ipynb) | Vine fitting, structure, simulation, Rosenblatt transform |
| [04 — Kde1d & Statistics](examples/04_kde1d_and_stats.ipynb) | KDE, dependence measures, pairs plot visualization |

---

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

- API design follows [vinecopulib](https://github.com/vinecopulib/vinecopulib) / [pyvinecopulib](https://github.com/vinecopulib/pyvinecopulib) by Thomas Nagler and Thibault Vatter.
- Development assisted by [GitHub Copilot](https://github.com/features/copilot).
