# torchvine

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=180&color=0:ee4c2c,100:ff6f00&text=torchvine&fontColor=ffffff&fontSize=60&fontAlignY=35&desc=Pure-PyTorch%20Vine%20Copula%20Modelling&descAlign=50&descAlignY=55" width="100%" alt="torchvine"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/torchvine/"><img src="https://img.shields.io/pypi/v/torchvine?color=ee4c2c&style=for-the-badge" alt="PyPI"/></a>
  <a href="https://pypi.org/project/torchvine/"><img src="https://img.shields.io/pypi/pyversions/torchvine?style=for-the-badge&color=ff6f00" alt="Python"/></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/></a>
</p>

<p align="center">
  GPU-ready, differentiable vine copula modelling in pure PyTorch.<br>
  <b>Drop-in replacement</b> for <a href="https://github.com/vinecopulib/pyvinecopulib">pyvinecopulib</a> — same API, but with autograd and CUDA support.
</p>

---

## ✨ Why torchvine?

| | torchvine | pyvinecopulib |
|---|---|---|
| **Backend** | Pure PyTorch (GPU / CPU) | C++ with Python bindings |
| **Differentiable** | ✅ Autograd-compatible | ❌ |
| **GPU acceleration** | ✅ CUDA tensors | ❌ CPU only |
| **API** | Drop-in replacement | Reference |
| **Copula families** | 13 (full parity) | 13 |

**Zero C/C++ dependencies** — everything is implemented in pure PyTorch, making it easy to install, debug, and extend.

---

## 🚀 Installation

```bash
pip install torchvine
```

From source:

```bash
git clone https://github.com/Bluerrror/torchvine.git
cd torchvine
pip install -e .
```

**Requirements:** Python ≥ 3.9 &nbsp;|&nbsp; PyTorch ≥ 2.0 &nbsp;|&nbsp; matplotlib ≥ 3.5

---

## 📖 Quick Start

### Bivariate Copula

```python
import torch
import torchvine as tv

# Create a Gaussian copula with correlation 0.7
cop = tv.Bicop(tv.BicopFamily.gaussian, parameters=torch.tensor([0.7]))
print(cop.str())               # <torchvine.Bicop> family: gaussian, parameters: [0.7000]
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
vine = tv.Vinecop.from_dimension(5)
vine.select(data, controls=tv.FitControlsVinecop(family_set=tv.parametric))

print(vine.str())
print(f"Log-likelihood: {vine.loglik(data):.2f}")
print(f"AIC: {vine.aic(data):.2f}")

# Simulate and transform
sim = vine.simulate(1000)
pit = vine.rosenblatt(data)      # probability integral transform
```

### Dependence Measures

```python
x = torch.randn(1000, dtype=torch.float64)
y = 0.6 * x + 0.8 * torch.randn(1000, dtype=torch.float64)

print(tv.kendall_tau(x, y))    # Kendall's tau
print(tv.spearman_rho(x, y))   # Spearman's rho
print(tv.pearson_cor(x, y))    # Pearson correlation
print(tv.blomqvist_beta(x, y)) # Blomqvist's beta
print(tv.hoeffding_d(x, y))    # Hoeffding's D
```

### GPU Acceleration

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
u_gpu = torch.rand(10000, 2, dtype=torch.float64, device=device)

cop = tv.Bicop(tv.BicopFamily.clayton, parameters=torch.tensor([3.0], device=device))
pdf_gpu = cop.pdf(u_gpu)  # runs entirely on GPU
```

---

## 📋 Supported Copula Families

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

## 📚 API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `tv.Bicop` | Bivariate copula — create, fit, evaluate, simulate |
| `tv.Vinecop` | Vine copula model — select, pdf, simulate, rosenblatt |
| `tv.Kde1d` | 1-D kernel density estimation — fit, pdf, cdf, quantile |
| `tv.RVineStructure` | R-vine structure matrix |
| `tv.DVineStructure` | D-vine structure (convenience subclass) |
| `tv.CVineStructure` | C-vine structure (convenience subclass) |
| `tv.FitControlsBicop` | Fitting options for bivariate copulas |
| `tv.FitControlsVinecop` | Fitting options for vine copulas |
| `tv.BicopFamily` | Enum of all copula families |

### Dependence Measures

| Function | Description |
|----------|-------------|
| `tv.kendall_tau(x, y)` | Kendall's rank correlation |
| `tv.spearman_rho(x, y)` | Spearman's rank correlation |
| `tv.pearson_cor(x, y)` | Pearson linear correlation |
| `tv.blomqvist_beta(x, y)` | Blomqvist's beta (medial correlation) |
| `tv.hoeffding_d(x, y)` | Hoeffding's D statistic |
| `tv.wdm(x, y, method)` | Unified interface for all measures |

### Utilities

| Function | Description |
|----------|-------------|
| `tv.to_pseudo_obs(data)` | Rank-transform to pseudo-observations |
| `tv.simulate_uniform(n, d)` | Uniform random / quasi-random samples |
| `tv.pairs_copula_data(data)` | Pairs plot with copula density contours |

---

## 📓 Examples

See the [`examples/`](examples/) directory for Jupyter notebooks:

| Notebook | Topics |
|----------|--------|
| [01 — Getting Started](examples/01_getting_started.ipynb) | Imports, copula basics, simulation, fitting |
| [02 — Bivariate Copulas](examples/02_bivariate_copulas.ipynb) | All families, rotations, Student-t, model selection |
| [03 — Vine Copulas](examples/03_vine_copulas.ipynb) | Vine fitting, structure, simulation, Rosenblatt transform |
| [04 — Kde1d & Statistics](examples/04_kde1d_and_stats.ipynb) | KDE, dependence measures, pairs plot visualization |

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m "Add amazing feature"`
4. Push: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- API design follows [vinecopulib](https://github.com/vinecopulib/vinecopulib) / [pyvinecopulib](https://github.com/vinecopulib/pyvinecopulib) by Thomas Nagler and Thibault Vatter.

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:ee4c2c,100:ff6f00&height=100&section=footer"/>
