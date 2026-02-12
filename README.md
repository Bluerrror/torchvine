# torchvine

**Pure-PyTorch vine copula modelling — GPU-ready, differentiable, and API-compatible with [pyvinecopulib](https://github.com/vinecopulib/pyvinecopulib).**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

---

## Features

| | torchvine | pyvinecopulib |
|---|---|---|
| **Backend** | Pure PyTorch (GPU / CPU) | C++ with Python bindings |
| **Differentiable** | ✅ Autograd-compatible | ❌ |
| **GPU acceleration** | ✅ CUDA tensors | ❌ CPU only |
| **API** | Drop-in replacement | Reference |
| **Copula families** | 12 (all except Student-t) | 13 |

### Supported Copula Families

| Family | Parameters | Type |
|--------|-----------|------|
| Independence | 0 | — |
| Gaussian | 1 (ρ) | Elliptical |
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
git clone https://github.com/bluerror2710/torchvine.git
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
cop = tv.Bicop("gaussian", parameters=torch.tensor([0.7]))
print(cop.str())               # <Bicop> family=gaussian, rotation=0, parameters=[0.7]
print(cop.parameters_to_tau()) # Kendall's tau ≈ 0.494

# Evaluate density and simulate
u = torch.rand(1000, 2, dtype=torch.float64)
pdf_vals = cop.pdf(u)
samples  = cop.simulate(1000)

# Fit from data
fitted = tv.Bicop.select(samples)
print(fitted.str())
```

### Vine Copula

```python
# Fit a 5-dimensional vine copula
data = torch.rand(500, 5, dtype=torch.float64)
vine = tv.Vinecop.from_dimension(5)
vine.select(data, controls=tv.FitControlsVinecop(parametric_method="itau"))

print(vine.str())
print(f"Log-likelihood: {vine.loglik(data):.2f}")
print(f"AIC: {vine.aic(data):.2f}")

# Simulate and transform
sim = vine.simulate(1000)
pit = vine.rosenblatt(data)      # probability integral transform
```

### GPU Acceleration

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
u_gpu = torch.rand(10000, 2, dtype=torch.float64, device=device)

cop = tv.Bicop("clayton", parameters=torch.tensor([3.0], device=device))
pdf_gpu = cop.pdf(u_gpu)  # runs entirely on GPU
```

---

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `tv.Bicop` | Bivariate copula (create, fit, evaluate, simulate) |
| `tv.Vinecop` | Vine copula model (select, pdf, simulate, rosenblatt) |
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
Vinecop.from_dimension(d)  |  Vinecop.from_structure(structure=, matrix=)
  .select(data, controls)  .fit(data)
  .pdf(u)  .loglik(u)  .aic(u)  .bic(u)  .simulate(n)
  .rosenblatt(u)  .inverse_rosenblatt(u)  .cdf(u)
  .truncate(level)  .to_json()  .from_json()  .str()
  .structure  .pair_copulas  .dim  .order  .npars
```

### Module-Level Functions

| Function | Description |
|----------|-------------|
| `tv.to_pseudo_obs(data)` | Rank-transform to pseudo-observations |
| `tv.simulate_uniform(n, d)` | Uniform random / quasi-random samples |
| `tv.sobol(n, d)` | Sobol quasi-random sequence |
| `tv.ghalton(n, d)` | Generalized Halton-like sequence |

### Family Convenience Lists

```python
tv.one_par        # [gaussian, clayton, gumbel, frank, joe]
tv.two_par        # [bb1, bb6, bb7, bb8]
tv.three_par      # [tawn]
tv.parametric     # all parametric families
tv.nonparametric  # [indep, tll]
tv.archimedean    # [clayton, gumbel, frank, joe, bb1, bb6, bb7, bb8]
tv.elliptical     # [gaussian]
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
│   ├── optimize.py          # Parameter optimization (MLE)
│   ├── rvine_structure.py   # R-vine / D-vine / C-vine structures
│   ├── stats.py             # Statistical helper functions
│   ├── tll_fit.py           # Transformation local likelihood estimator
│   ├── vine_select.py       # Vine structure & family selection
│   └── vinecop.py           # Vine copula model
├── tests/                   # Unit + cross-check tests (75 tests)
├── benchmarks/              # Speed comparison vs pyvinecopulib
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
| [01 — Getting Started](examples/01_getting_started.ipynb) | Imports, copula basics, simulation, fitting, vine copulas |
| [02 — Bivariate Copulas](examples/02_bivariate_copulas.ipynb) | All families, rotations, parameter effects, TLL, model selection |
| [03 — Vine Copulas](examples/03_vine_copulas.ipynb) | Vine fitting, structure, Rosenblatt transform, high-dimensional example |

---

## Benchmarks

Run `python benchmarks/run_benchmarks.py` to generate speed comparison plots against pyvinecopulib.

Benchmarks cover: Bicop pdf/hfunc/simulate, Vinecop fit/pdf/simulate across multiple families and sample sizes.

---

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

- API design follows [vinecopulib](https://github.com/vinecopulib/vinecopulib) / [pyvinecopulib](https://github.com/vinecopulib/pyvinecopulib) by Thomas Nagler and Thibault Vatter.
- Development assisted by [GitHub Copilot](https://github.com/features/copilot).
