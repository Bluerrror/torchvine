"""Cross-check torchvine Bicop against pyvinecopulib for numerical accuracy.

Tests compare pdf, cdf, hfunc1, hfunc2, hinv1 for all shared parametric families.
"""
import unittest
import numpy as np
import torch

try:
    import pyvinecopulib as pv
    HAS_PV = True
except ImportError:
    HAS_PV = False

import torchvine as tv


def _pv_params(params):
    """Convert flat param list to pyvinecopulib 2D column array."""
    return np.array(params, dtype=np.float64).reshape(-1, 1)


# Families and params shared between torchvine and pyvinecopulib (no student).
FAMILIES = {
    "gaussian":  {"params": [0.6], "rotations": [0]},
    "clayton":   {"params": [2.5], "rotations": [0, 90, 180, 270]},
    "gumbel":    {"params": [2.0], "rotations": [0, 90, 180, 270]},
    "frank":     {"params": [5.0], "rotations": [0]},
    "joe":       {"params": [2.5], "rotations": [0, 90, 180, 270]},
    "bb1":       {"params": [0.5, 1.5], "rotations": [0, 90, 180, 270]},
    "bb6":       {"params": [2.0, 2.0], "rotations": [0, 90, 180, 270]},
    "bb7":       {"params": [2.0, 1.0], "rotations": [0, 90, 180, 270]},
    "bb8":       {"params": [3.0, 0.6], "rotations": [0, 90, 180, 270]},
    "tawn":      {"params": [0.5, 0.5, 2.0], "rotations": [0]},
}

# Shared test data
RNG = np.random.default_rng(42)
U_NP = np.asfortranarray(np.clip(RNG.uniform(size=(500, 2)), 0.02, 0.98))
U_TH = torch.tensor(U_NP, dtype=torch.float64)


@unittest.skipUnless(HAS_PV, "pyvinecopulib not installed")
class TestBicopPdfAccuracy(unittest.TestCase):
    """Cross-check Bicop.pdf() against pyvinecopulib."""

    def test_pdf_all_families(self):
        for fam_name, cfg in FAMILIES.items():
            params = cfg["params"]
            for rot in cfg["rotations"]:
                with self.subTest(family=fam_name, rotation=rot):
                    pv_fam = getattr(pv.BicopFamily, fam_name)
                    pv_c = pv.Bicop(family=pv_fam, rotation=rot,
                                    parameters=_pv_params(params))

                    tv_c = tv.Bicop(fam_name, rotation=rot,
                                    parameters=torch.tensor(params, dtype=torch.float64))

                    pv_pdf = pv_c.pdf(U_NP)
                    tv_pdf = tv_c.pdf(U_TH).detach().cpu().numpy()

                    max_err = np.max(np.abs(pv_pdf - tv_pdf))
                    rel_err = np.max(np.abs(pv_pdf - tv_pdf) / (np.abs(pv_pdf) + 1e-15))
                    self.assertLess(max_err, 1e-4,
                                    f"{fam_name} rot={rot}: max_err={max_err:.2e}")


@unittest.skipUnless(HAS_PV, "pyvinecopulib not installed")
class TestBicopCdfAccuracy(unittest.TestCase):
    """Cross-check Bicop.cdf() against pyvinecopulib."""

    def test_cdf_all_families(self):
        for fam_name, cfg in FAMILIES.items():
            params = cfg["params"]
            for rot in cfg["rotations"]:
                with self.subTest(family=fam_name, rotation=rot):
                    pv_fam = getattr(pv.BicopFamily, fam_name)
                    pv_c = pv.Bicop(family=pv_fam, rotation=rot,
                                    parameters=_pv_params(params))

                    tv_c = tv.Bicop(fam_name, rotation=rot,
                                    parameters=torch.tensor(params, dtype=torch.float64))

                    pv_cdf = pv_c.cdf(U_NP)
                    tv_cdf = tv_c.cdf(U_TH).detach().cpu().numpy()

                    max_err = np.max(np.abs(pv_cdf - tv_cdf))
                    self.assertLess(max_err, 0.02,
                                    f"{fam_name} rot={rot}: max_err={max_err:.2e}")


@unittest.skipUnless(HAS_PV, "pyvinecopulib not installed")
class TestBicopHfuncAccuracy(unittest.TestCase):
    """Cross-check Bicop.hfunc1/hfunc2 against pyvinecopulib."""

    def test_hfunc1_all_families(self):
        for fam_name, cfg in FAMILIES.items():
            params = cfg["params"]
            for rot in cfg["rotations"]:
                with self.subTest(family=fam_name, rotation=rot):
                    pv_fam = getattr(pv.BicopFamily, fam_name)
                    pv_c = pv.Bicop(family=pv_fam, rotation=rot,
                                    parameters=_pv_params(params))
                    tv_c = tv.Bicop(fam_name, rotation=rot,
                                    parameters=torch.tensor(params, dtype=torch.float64))

                    pv_h1 = pv_c.hfunc1(U_NP)
                    tv_h1 = tv_c.hfunc1(U_TH).detach().cpu().numpy()

                    max_err = np.max(np.abs(pv_h1 - tv_h1))
                    self.assertLess(max_err, 1e-4,
                                    f"{fam_name} rot={rot}: max_err={max_err:.2e}")

    def test_hfunc2_all_families(self):
        for fam_name, cfg in FAMILIES.items():
            params = cfg["params"]
            for rot in cfg["rotations"]:
                with self.subTest(family=fam_name, rotation=rot):
                    pv_fam = getattr(pv.BicopFamily, fam_name)
                    pv_c = pv.Bicop(family=pv_fam, rotation=rot,
                                    parameters=_pv_params(params))
                    tv_c = tv.Bicop(fam_name, rotation=rot,
                                    parameters=torch.tensor(params, dtype=torch.float64))

                    pv_h2 = pv_c.hfunc2(U_NP)
                    tv_h2 = tv_c.hfunc2(U_TH).detach().cpu().numpy()

                    max_err = np.max(np.abs(pv_h2 - tv_h2))
                    self.assertLess(max_err, 1e-4,
                                    f"{fam_name} rot={rot}: max_err={max_err:.2e}")


@unittest.skipUnless(HAS_PV, "pyvinecopulib not installed")
class TestBicopHinvAccuracy(unittest.TestCase):
    """Cross-check Bicop.hinv1 against pyvinecopulib."""

    def test_hinv1_all_families(self):
        for fam_name, cfg in FAMILIES.items():
            params = cfg["params"]
            rot = cfg["rotations"][0]  # test default rotation only
            with self.subTest(family=fam_name, rotation=rot):
                pv_fam = getattr(pv.BicopFamily, fam_name)
                pv_c = pv.Bicop(family=pv_fam, rotation=rot,
                                parameters=_pv_params(params))
                tv_c = tv.Bicop(fam_name, rotation=rot,
                                parameters=torch.tensor(params, dtype=torch.float64))

                pv_hinv = pv_c.hinv1(U_NP)
                tv_hinv = tv_c.hinv1(U_TH).detach().cpu().numpy()

                max_err = np.max(np.abs(pv_hinv - tv_hinv))
                self.assertLess(max_err, 5e-3,
                                f"{fam_name} rot={rot}: max_err={max_err:.2e}")


@unittest.skipUnless(HAS_PV, "pyvinecopulib not installed")
class TestBicopParametersToTau(unittest.TestCase):
    """Cross-check parameters_to_tau against pyvinecopulib."""

    # Only families with analytical tau formulas
    TAU_FAMILIES = {k: v for k, v in FAMILIES.items()
                    if k in ("gaussian", "clayton", "gumbel", "frank", "joe")}

    def test_tau_all_families(self):
        for fam_name, cfg in self.TAU_FAMILIES.items():
            params = cfg["params"]
            with self.subTest(family=fam_name):
                pv_fam = getattr(pv.BicopFamily, fam_name)
                pv_c = pv.Bicop(family=pv_fam, rotation=0,
                                parameters=_pv_params(params))
                tv_c = tv.Bicop(fam_name, rotation=0,
                                parameters=torch.tensor(params, dtype=torch.float64))

                pv_tau = pv_c.parameters_to_tau(_pv_params(params))
                tv_tau = tv_c.parameters_to_tau()

                self.assertAlmostEqual(pv_tau, tv_tau, places=3,
                                       msg=f"{fam_name}: pv={pv_tau:.6f} tv={tv_tau:.6f}")


@unittest.skipUnless(HAS_PV, "pyvinecopulib not installed")
class TestBicopFitAccuracy(unittest.TestCase):
    """Cross-check Bicop.fit() (itau) against pyvinecopulib."""

    def test_itau_fit(self):
        rng = np.random.default_rng(123)
        u = np.asfortranarray(np.clip(rng.uniform(size=(600, 2)), 1e-6, 1 - 1e-6))
        u_th = torch.tensor(u, dtype=torch.float64)

        for fam_name in ["gaussian", "clayton", "gumbel", "frank", "joe"]:
            with self.subTest(family=fam_name):
                pv_c = pv.Bicop(family=getattr(pv.BicopFamily, fam_name))
                pv_c.fit(u, controls=pv.FitControlsBicop(parametric_method="itau"))

                tv_c = tv.Bicop(family=fam_name)
                tv_c.fit(u_th, controls=tv.FitControlsBicop(parametric_method="itau"))

                pv_p = np.array(pv_c.parameters).flatten()
                tv_p = tv_c.parameters.detach().cpu().numpy().flatten()

                max_err = np.max(np.abs(pv_p - tv_p))
                self.assertLess(max_err, 5e-3,
                                f"{fam_name}: pv={pv_p} tv={tv_p} err={max_err:.6f}")


@unittest.skipUnless(HAS_PV, "pyvinecopulib not installed")
class TestToPseudoObs(unittest.TestCase):
    """Cross-check to_pseudo_obs against pyvinecopulib."""

    def test_pseudo_obs_match(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal((200, 3))
        x_th = torch.tensor(x, dtype=torch.float64)

        pv_u = pv.to_pseudo_obs(x)
        tv_u = tv.to_pseudo_obs(x_th).numpy()

        max_err = np.max(np.abs(pv_u - tv_u))
        self.assertLess(max_err, 1e-10,
                        f"to_pseudo_obs max_err={max_err:.2e}")


if __name__ == "__main__":
    unittest.main()
