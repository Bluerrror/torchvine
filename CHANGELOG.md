# Changelog

## [0.2.0] — 2026-02-18

### Improved
- Expanded PyPI keywords for better search discoverability
- Added License and Information Analysis classifiers
- Added Changelog URL to project metadata
- Fixed README examples: `Vinecop(d=5)` → `Vinecop.from_dimension(5)`
- Corrected `cop.str()` inline comment to match actual output format

## [0.1.0] — 2025-06-15

### Added
- Initial release of torchvine
- Pure-PyTorch bivariate copula (`Bicop`) with 13 families
- Vine copula model (`Vinecop`) with structure selection
- 1-D kernel density estimation (`Kde1d`)
- Five dependence measures: Kendall's τ, Spearman's ρ, Pearson r, Blomqvist's β, Hoeffding's D
- GPU acceleration via CUDA tensors
- Drop-in API compatibility with pyvinecopulib
- R-vine, C-vine, and D-vine structure classes
- Pairs copula data visualization
- Four example Jupyter notebooks
