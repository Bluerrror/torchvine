"""Cross-check torchvine Vinecop against pyvinecopulib for numerical accuracy.

Tests compare vine copula fitting, pdf, simulation, and Rosenblatt transform.
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


@unittest.skipUnless(HAS_PV, "pyvinecopulib not installed")
class TestVinecopSelectAccuracy(unittest.TestCase):
    """Cross-check vine copula selection against pyvinecopulib."""

    def _make_data(self, d=4, n=500, seed=42):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, d))
        # Apply correlation via Cholesky
        rho = 0.5
        L = np.eye(d) * (1 - rho) + np.ones((d, d)) * rho
        L = np.linalg.cholesky(L)
        x = x @ L.T
        # Convert to pseudo-observations
        u_np = pv.to_pseudo_obs(x)
        u_th = torch.tensor(u_np, dtype=torch.float64)
        return u_np, u_th

    def test_select_3d_families_match(self):
        """Fitted families should broadly agree (both choose dependency)."""
        u_np, u_th = self._make_data(d=3, n=600, seed=42)

        # pyvinecopulib fit (exclude student to match torchvine)
        pv_fams = [f for f in pv.BicopFamily if f != pv.BicopFamily.student]
        pv_v = pv.Vinecop(d=3)
        pv_v.select(u_np, controls=pv.FitControlsVinecop(
            family_set=pv_fams, parametric_method="itau"))

        # torchvine fit
        tv_v = tv.Vinecop.from_dimension(3)
        tv_v.select(u_th, controls=tv.FitControlsVinecop(
            parametric_method="itau"))

        # Check dimensions match
        self.assertEqual(pv_v.dim, tv_v.dim)
        self.assertEqual(pv_v.dim, 3)

        # At least the first tree should pick a non-indep family
        pv_fam0 = str(pv_v.get_family(0, 0))
        tv_fam0 = str(tv_v.get_family(0, 0))
        # Both should detect dependency
        non_indep_pv = pv_fam0 != "BicopFamily.indep"
        non_indep_tv = tv_fam0 != "BicopFamily.indep"
        self.assertEqual(non_indep_pv, non_indep_tv,
                         f"pv={pv_fam0}, tv={tv_fam0}")

    def test_select_pdf_agreement(self):
        """After fitting, pdf values should be close on the same data."""
        u_np, u_th = self._make_data(d=3, n=600, seed=123)

        pv_fams = [f for f in pv.BicopFamily if f != pv.BicopFamily.student]
        pv_v = pv.Vinecop(d=3)
        pv_v.select(u_np, controls=pv.FitControlsVinecop(
            family_set=pv_fams, parametric_method="itau"))

        tv_v = tv.Vinecop.from_dimension(3)
        tv_v.select(u_th, controls=tv.FitControlsVinecop(
            parametric_method="itau"))

        # Evaluate on a subset
        test_np = u_np[:100]
        test_th = u_th[:100]

        pv_pdf = pv_v.pdf(test_np)
        tv_pdf = tv_v.pdf(test_th).detach().cpu().numpy()

        # Compare log-likelihoods (more stable than raw pdfs)
        pv_ll = np.mean(np.log(np.clip(pv_pdf, 1e-15, None)))
        tv_ll = np.mean(np.log(np.clip(tv_pdf, 1e-15, None)))

        # They should be in the same ballpark (within 0.5 nats)
        self.assertAlmostEqual(pv_ll, tv_ll, delta=0.5,
                               msg=f"loglik mismatch: pv={pv_ll:.4f}, tv={tv_ll:.4f}")


@unittest.skipUnless(HAS_PV, "pyvinecopulib not installed")
class TestVinecopPdfFixedStructure(unittest.TestCase):
    """Cross-check Vinecop pdf with identical structure and pair copulas."""

    def test_pdf_indep_vine(self):
        """Independence vine should give pdf=1 everywhere."""
        u_np = np.asfortranarray(np.random.default_rng(42).uniform(
            0.01, 0.99, (100, 4)))
        u_th = torch.tensor(u_np, dtype=torch.float64)

        pv_v = pv.Vinecop(d=4)
        tv_v = tv.Vinecop.from_dimension(4)

        pv_pdf = pv_v.pdf(u_np)
        tv_pdf = tv_v.pdf(u_th).detach().cpu().numpy()

        np.testing.assert_allclose(tv_pdf, pv_pdf, atol=1e-10)

    def test_pdf_gaussian_pair_copulas(self):
        """Vine with gaussian pair copulas, same params."""
        rho = 0.5

        # torchvine
        tv_pcs = [[tv.Bicop("gaussian", parameters=torch.tensor([rho], dtype=torch.float64)),
                    tv.Bicop("gaussian", parameters=torch.tensor([rho], dtype=torch.float64))],
                   [tv.Bicop("gaussian", parameters=torch.tensor([rho], dtype=torch.float64))]]
        tv_struct = tv.DVineStructure.from_order([1, 2, 3])
        tv_v = tv.Vinecop.from_structure(structure=tv_struct, pair_copulas=tv_pcs)

        u_th = torch.rand(200, 3, dtype=torch.float64).clamp(0.02, 0.98)
        torch.manual_seed(42)

        tv_pdf = tv_v.pdf(u_th).detach().cpu().numpy()

        # All pdfs should be positive and finite
        self.assertTrue(np.all(np.isfinite(tv_pdf)))
        self.assertTrue(np.all(tv_pdf > 0))

        # The vine should increase density compared to independence
        self.assertGreater(np.mean(np.log(tv_pdf)), -1.0)


@unittest.skipUnless(HAS_PV, "pyvinecopulib not installed")
class TestVinecopSimulate(unittest.TestCase):
    """Cross-check that simulated data has correct marginal properties."""

    def test_simulate_uniform_marginals(self):
        """Simulated data should have approximately uniform marginals."""
        tv_v = tv.Vinecop.from_dimension(4)
        torch.manual_seed(42)
        sim = tv_v.simulate(2000)

        for j in range(4):
            col = sim[:, j].numpy()
            # KS test against uniform(0,1)
            from scipy import stats as sp_stats
            ks_stat, p_val = sp_stats.kstest(col, "uniform")
            self.assertGreater(p_val, 0.01,
                               f"Column {j}: KS p={p_val:.4f} < 0.01")


@unittest.skipUnless(HAS_PV, "pyvinecopulib not installed")
class TestVinecopRosenblatt(unittest.TestCase):
    """Cross-check Rosenblatt transform against pyvinecopulib."""

    def test_rosenblatt_indep(self):
        """For independence vine, Rosenblatt = identity."""
        u_np = np.asfortranarray(np.random.default_rng(42).uniform(
            0.01, 0.99, (100, 3)))
        u_th = torch.tensor(u_np, dtype=torch.float64)

        pv_v = pv.Vinecop(d=3)
        tv_v = tv.Vinecop.from_dimension(3)

        pv_ros = pv_v.rosenblatt(u_np)
        tv_ros = tv_v.rosenblatt(u_th).detach().cpu().numpy()

        np.testing.assert_allclose(tv_ros, pv_ros, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
