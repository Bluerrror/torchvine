"""Unit tests for torchvine core functionality."""
import unittest
import torch
import torchvine as tv


class TestBicopFamily(unittest.TestCase):
    """Test BicopFamily enum and convenience lists."""

    def test_no_student(self):
        self.assertTrue(hasattr(tv.BicopFamily, "student"))

    def test_all_families_present(self):
        expected = {"indep", "gaussian", "student", "clayton", "gumbel", "frank", "joe",
                    "bb1", "bb6", "bb7", "bb8", "tawn", "tll"}
        actual = {f.value for f in tv.BicopFamily}
        self.assertEqual(actual, expected)

    def test_family_shortcut_names(self):
        for name in ["indep", "gaussian", "clayton", "gumbel", "frank", "joe",
                      "bb1", "bb6", "bb7", "bb8", "tawn", "tll"]:
            self.assertIs(getattr(tv, name), tv.BicopFamily(name))

    def test_convenience_lists(self):
        self.assertEqual(len(tv.one_par), 5)
        self.assertEqual(len(tv.two_par), 5)
        self.assertEqual(len(tv.three_par), 1)
        self.assertEqual(set(tv.parametric), set(tv.one_par + tv.two_par + tv.three_par))
        self.assertIn(tv.BicopFamily.tll, tv.nonparametric)
        self.assertIn(tv.BicopFamily.gaussian, tv.elliptical)
        self.assertIn(tv.BicopFamily.gumbel, tv.archimedean)
        self.assertIn(tv.BicopFamily.tawn, tv.extreme_value)
        self.assertEqual(len(tv.all), 13)


# Fixed parameter sets for each family
_FAMILY_PARAMS = {
    "indep": [],
    "gaussian": [0.5],
    "clayton": [2.0],
    "gumbel": [2.0],
    "frank": [5.0],
    "joe": [2.0],
    "bb1": [0.5, 1.5],
    "bb6": [2.0, 2.0],
    "bb7": [2.0, 1.0],
    "bb8": [3.0, 0.6],
    "tawn": [0.5, 0.5, 2.0],
}


class TestBicopAllFamilies(unittest.TestCase):
    """Test pdf/cdf/hfunc/hinv for every parametric family."""

    def _make_bicop(self, fam, rotation=0):
        params = _FAMILY_PARAMS[fam]
        p = torch.tensor(params, dtype=torch.float64) if params else None
        return tv.Bicop(fam, rotation=rotation, parameters=p)

    def test_pdf_positive_finite(self):
        u = torch.rand(200, 2, dtype=torch.float64).clamp(0.02, 0.98)
        for fam in _FAMILY_PARAMS:
            with self.subTest(family=fam):
                c = self._make_bicop(fam)
                pdf = c.pdf(u)
                self.assertEqual(pdf.shape, (200,))
                self.assertTrue(torch.isfinite(pdf).all(), f"{fam} pdf has non-finite values")
                self.assertTrue((pdf >= 0).all(), f"{fam} pdf has negative values")

    def test_cdf_in_unit_interval(self):
        u = torch.rand(200, 2, dtype=torch.float64).clamp(0.02, 0.98)
        for fam in _FAMILY_PARAMS:
            with self.subTest(family=fam):
                c = self._make_bicop(fam)
                cdf = c.cdf(u)
                self.assertEqual(cdf.shape, (200,))
                self.assertTrue((cdf >= -1e-6).all() and (cdf <= 1 + 1e-6).all(),
                                f"{fam} cdf out of [0,1]")

    def test_hfunc_in_unit_interval(self):
        u = torch.rand(200, 2, dtype=torch.float64).clamp(0.02, 0.98)
        for fam in _FAMILY_PARAMS:
            with self.subTest(family=fam):
                c = self._make_bicop(fam)
                h1 = c.hfunc1(u)
                h2 = c.hfunc2(u)
                self.assertTrue((h1 >= -1e-6).all() and (h1 <= 1 + 1e-6).all())
                self.assertTrue((h2 >= -1e-6).all() and (h2 <= 1 + 1e-6).all())

    def test_hinv_roundtrip(self):
        u = torch.rand(200, 2, dtype=torch.float64).clamp(0.05, 0.95)
        for fam in _FAMILY_PARAMS:
            with self.subTest(family=fam):
                c = self._make_bicop(fam)
                h1 = c.hfunc1(u)
                uw = torch.stack([u[:, 0], h1], dim=1)
                u2_rec = c.hinv1(uw)
                err = (u2_rec - u[:, 1]).abs().max().item()
                self.assertLess(err, 1e-3, f"{fam} hinv1 roundtrip error={err:.6f}")

    def test_simulate_shape_and_range(self):
        for fam in _FAMILY_PARAMS:
            with self.subTest(family=fam):
                c = self._make_bicop(fam)
                sim = c.simulate(300)
                self.assertEqual(sim.shape, (300, 2))
                self.assertTrue(((sim > 0) & (sim < 1)).all())

    def test_rotated_families(self):
        rotatable = ["clayton", "gumbel", "joe", "bb1", "bb6", "bb7", "bb8"]
        u = torch.rand(100, 2, dtype=torch.float64).clamp(0.05, 0.95)
        for fam in rotatable:
            for rot in [0, 90, 180, 270]:
                with self.subTest(family=fam, rotation=rot):
                    c = self._make_bicop(fam, rotation=rot)
                    pdf = c.pdf(u)
                    self.assertTrue(torch.isfinite(pdf).all())
                    self.assertTrue((pdf >= 0).all())


class TestBicopMethods(unittest.TestCase):
    """Test Bicop utility methods."""

    def setUp(self):
        self.c = tv.Bicop("gaussian", parameters=torch.tensor([0.5], dtype=torch.float64))

    def test_loglik(self):
        u = torch.rand(100, 2, dtype=torch.float64).clamp(0.01, 0.99)
        ll = self.c.loglik(u)
        self.assertIsInstance(ll, float)
        self.assertTrue(ll > -1e10)

    def test_aic_bic(self):
        u = torch.rand(100, 2, dtype=torch.float64).clamp(0.01, 0.99)
        aic = self.c.aic(u)
        bic = self.c.bic(u)
        self.assertIsInstance(aic, float)
        self.assertIsInstance(bic, float)

    def test_npars(self):
        self.assertEqual(self.c.npars, 1.0)
        c2 = tv.Bicop("bb1", parameters=torch.tensor([0.5, 1.5], dtype=torch.float64))
        self.assertEqual(c2.npars, 2.0)

    def test_parameters_to_tau(self):
        tau = self.c.parameters_to_tau()
        self.assertIsInstance(tau, float)
        self.assertGreater(tau, 0)
        self.assertLess(tau, 1)

    def test_tau_to_parameters(self):
        p = self.c.tau_to_parameters(0.3)
        self.assertEqual(p.numel(), 1)

    def test_parameter_bounds(self):
        lb = self.c.parameters_lower_bounds
        ub = self.c.parameters_upper_bounds
        self.assertTrue((lb < ub).all())

    def test_str(self):
        s = self.c.str()
        self.assertIn("gaussian", s)
        self.assertIn("torchvine.Bicop", s)

    def test_to_json_from_json(self):
        j = self.c.to_json()
        c2 = tv.Bicop.from_json(j)
        self.assertEqual(c2.family, self.c.family)
        self.assertEqual(c2.rotation, self.c.rotation)

    def test_tau_property(self):
        t = self.c.tau
        self.assertIsInstance(t, float)


class TestBicopFitSelect(unittest.TestCase):
    """Test Bicop fit and select methods."""

    def test_fit_gaussian_itau(self):
        torch.manual_seed(42)
        c_true = tv.Bicop("gaussian", parameters=torch.tensor([0.6], dtype=torch.float64))
        u = c_true.simulate(500)
        c = tv.Bicop("gaussian")
        c.fit(u, controls=tv.FitControlsBicop(parametric_method="itau"))
        rho = c.parameters[0].item()
        self.assertAlmostEqual(rho, 0.6, delta=0.15)

    def test_fit_clayton_mle(self):
        torch.manual_seed(42)
        c_true = tv.Bicop("clayton", parameters=torch.tensor([3.0], dtype=torch.float64))
        u = c_true.simulate(500)
        c = tv.Bicop("clayton")
        c.fit(u, controls=tv.FitControlsBicop(parametric_method="mle"))
        theta = c.parameters[0].item()
        self.assertAlmostEqual(theta, 3.0, delta=1.0)

    def test_select(self):
        torch.manual_seed(42)
        c_true = tv.Bicop("frank", parameters=torch.tensor([5.0], dtype=torch.float64))
        u = c_true.simulate(500)
        c = tv.Bicop()
        c.select(u)
        self.assertIn(c.family.value, ["frank", "gaussian", "joe", "gumbel", "clayton"])


class TestVinecop(unittest.TestCase):
    """Test Vinecop class."""

    def test_from_dimension(self):
        v = tv.Vinecop.from_dimension(5)
        self.assertEqual(v.dim, 5)

    def test_pdf_shape(self):
        v = tv.Vinecop.from_dimension(4)
        u = torch.rand(100, 4, dtype=torch.float64).clamp(0.01, 0.99)
        pdf = v.pdf(u)
        self.assertEqual(pdf.shape, (100,))

    def test_simulate_shape_and_range(self):
        v = tv.Vinecop.from_dimension(4)
        sim = v.simulate(200)
        self.assertEqual(sim.shape, (200, 4))
        self.assertTrue(((sim > 0) & (sim < 1)).all())

    def test_rosenblatt_inverse_roundtrip(self):
        torch.manual_seed(42)
        v = tv.Vinecop.from_dimension(3)
        u = torch.rand(100, 3, dtype=torch.float64).clamp(0.05, 0.95)
        w = v.rosenblatt(u)
        u_rec = v.inverse_rosenblatt(w)
        err = (u - u_rec).abs().max().item()
        self.assertLess(err, 1e-3)

    def test_loglik_aic_bic(self):
        v = tv.Vinecop.from_dimension(3)
        u = torch.rand(100, 3, dtype=torch.float64).clamp(0.01, 0.99)
        ll = v.loglik(u)
        aic = v.aic(u)
        bic = v.bic(u)
        self.assertIsInstance(ll, float)
        self.assertIsInstance(aic, float)
        self.assertIsInstance(bic, float)

    def test_str(self):
        v = tv.Vinecop.from_dimension(3)
        s = v.str()
        self.assertIn("torchvine.Vinecop", s)

    def test_to_json_from_json(self):
        v = tv.Vinecop.from_dimension(3)
        j = v.to_json()
        v2 = tv.Vinecop.from_json(j)
        self.assertEqual(v2.dim, v.dim)

    def test_get_family_tau_params(self):
        v = tv.Vinecop.from_dimension(3)
        fam = v.get_family(0, 0)
        self.assertIsInstance(fam, tv.BicopFamily)
        tau = v.get_tau(0, 0)
        self.assertIsInstance(tau, float)

    def test_pair_copulas_property(self):
        v = tv.Vinecop.from_dimension(4)
        pc = v.pair_copulas
        self.assertIsInstance(pc, list)
        self.assertEqual(len(pc), 3)  # d-1 trees

    def test_structure_property(self):
        v = tv.Vinecop.from_dimension(4)
        s = v.structure
        self.assertIsInstance(s, tv.RVineStructure)

    def test_var_types_property(self):
        v = tv.Vinecop.from_dimension(3)
        vt = v.var_types
        self.assertIsInstance(vt, list)
        self.assertEqual(len(vt), 3)


class TestVinecopFitSelect(unittest.TestCase):
    """Test Vinecop fit and select."""

    def test_select_small(self):
        torch.manual_seed(42)
        u = torch.rand(200, 3, dtype=torch.float64).clamp(0.01, 0.99)
        v = tv.Vinecop.from_dimension(3)
        v.select(u)
        self.assertEqual(v.dim, 3)
        # After selection, families should be assigned
        fam = v.get_family(0, 0)
        self.assertIsInstance(fam, tv.BicopFamily)

    def test_truncate(self):
        v = tv.Vinecop.from_dimension(5)
        v.truncate(2)
        self.assertEqual(v.trunc_lvl, 2)


class TestRVineStructure(unittest.TestCase):
    """Test RVineStructure."""

    def test_from_dimension(self):
        s = tv.RVineStructure.from_dimension(5)
        self.assertEqual(s.dim, 5)
        self.assertEqual(len(s.order), 5)

    def test_from_matrix(self):
        # Use a valid 3x3 D-vine matrix
        s = tv.RVineStructure.from_dimension(3)
        M = s.matrix
        s2 = tv.RVineStructure.from_matrix(M)
        self.assertEqual(s2.dim, 3)

    def test_matrix_roundtrip(self):
        s = tv.RVineStructure.from_dimension(4)
        M = s.matrix
        s2 = tv.RVineStructure.from_matrix(M)
        self.assertEqual(s2.order, s.order)

    def test_simulate(self):
        s = tv.RVineStructure.simulate(5)
        self.assertEqual(s.dim, 5)

    def test_truncate(self):
        s = tv.RVineStructure.from_dimension(5)
        s2 = s.truncate(2)
        self.assertEqual(s2.trunc_lvl, 2)

    def test_str(self):
        s = tv.RVineStructure.from_dimension(3)
        self.assertIn("RVineStructure", s.str())

    def test_json_roundtrip(self):
        s = tv.RVineStructure.from_dimension(4)
        j = s.to_json()
        s2 = tv.RVineStructure.from_json(j)
        self.assertEqual(s2.order, s.order)
        self.assertEqual(s2.dim, s.dim)

    def test_dvine_structure(self):
        s = tv.DVineStructure.from_order([1, 2, 3, 4])
        self.assertEqual(s.dim, 4)

    def test_cvine_structure(self):
        s = tv.CVineStructure.from_order([1, 2, 3, 4])
        self.assertEqual(s.dim, 4)


class TestFitControls(unittest.TestCase):
    """Test FitControls classes."""

    def test_bicop_defaults(self):
        fc = tv.FitControlsBicop()
        self.assertEqual(fc.parametric_method, "mle")
        self.assertEqual(fc.selection_criterion, "bic")
        self.assertIsNone(fc.weights)

    def test_bicop_str(self):
        fc = tv.FitControlsBicop()
        s = fc.str()
        self.assertIn("Parametric method", s)
        self.assertIn("Selection criterion", s)

    def test_vinecop_defaults(self):
        fcv = tv.FitControlsVinecop()
        self.assertEqual(fcv.tree_criterion, "tau")
        self.assertEqual(fcv.threshold, 0.0)

    def test_vinecop_str(self):
        fcv = tv.FitControlsVinecop()
        s = fcv.str()
        self.assertIn("Tree criterion", s)
        self.assertIn("Tree algorithm", s)

    def test_mst_algorithm_alias(self):
        fcv = tv.FitControlsVinecop()
        self.assertEqual(fcv.mst_algorithm, fcv.tree_algorithm)


class TestModuleFunctions(unittest.TestCase):
    """Test module-level functions."""

    def test_simulate_uniform(self):
        u = tv.simulate_uniform(100, 5)
        self.assertEqual(u.shape, (100, 5))
        self.assertTrue(((u >= 0) & (u <= 1)).all())

    def test_simulate_uniform_qrng(self):
        u = tv.simulate_uniform(100, 3, qrng=True)
        self.assertEqual(u.shape, (100, 3))

    def test_sobol(self):
        s = tv.sobol(50, 4)
        self.assertEqual(s.shape, (50, 4))
        self.assertTrue(((s >= 0) & (s <= 1)).all())

    def test_ghalton(self):
        g = tv.ghalton(50, 4)
        self.assertEqual(g.shape, (50, 4))
        self.assertTrue(((g >= 0) & (g <= 1)).all())

    def test_to_pseudo_obs(self):
        x = torch.randn(100, 3, dtype=torch.float64)
        u = tv.to_pseudo_obs(x)
        self.assertEqual(u.shape, (100, 3))
        self.assertTrue(((u > 0) & (u < 1)).all())


if __name__ == "__main__":
    unittest.main()
