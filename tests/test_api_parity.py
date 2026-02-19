"""Test that torchvine has API parity with pyvinecopulib (excluding student copula)."""
import unittest

try:
    import pyvinecopulib as pv
    HAS_PV = True
except ImportError:
    HAS_PV = False

import torchvine as tv


@unittest.skipUnless(HAS_PV, "pyvinecopulib not installed")
class TestBicopAPIParity(unittest.TestCase):
    """Every public method on pv.Bicop should exist on tv.Bicop."""

    SKIP = {"plot"}  # plot may differ in implementation

    def test_all_methods_exist(self):
        pv_methods = {m for m in dir(pv.Bicop) if not m.startswith("_")} - self.SKIP
        tv_methods = {m for m in dir(tv.Bicop) if not m.startswith("_")}
        missing = pv_methods - tv_methods
        self.assertEqual(missing, set(),
                         f"Missing Bicop methods: {missing}")


@unittest.skipUnless(HAS_PV, "pyvinecopulib not installed")
class TestVinecopAPIParity(unittest.TestCase):
    """Every public method on pv.Vinecop should exist on tv.Vinecop."""

    SKIP = {"plot"}

    def test_all_methods_exist(self):
        pv_methods = {m for m in dir(pv.Vinecop) if not m.startswith("_")} - self.SKIP
        tv_inst = tv.Vinecop.from_dimension(3)
        tv_methods = {m for m in dir(tv_inst) if not m.startswith("_")}
        missing = pv_methods - tv_methods
        self.assertEqual(missing, set(),
                         f"Missing Vinecop methods: {missing}")


@unittest.skipUnless(HAS_PV, "pyvinecopulib not installed")
class TestRVineStructureAPIParity(unittest.TestCase):
    """Every public method on pv.RVineStructure should exist on tv.RVineStructure."""

    def test_all_methods_exist(self):
        pv_methods = {m for m in dir(pv.RVineStructure) if not m.startswith("_")}
        tv_inst = tv.RVineStructure.from_dimension(4)
        tv_methods = {m for m in dir(tv_inst) if not m.startswith("_")}
        missing = pv_methods - tv_methods
        self.assertEqual(missing, set(),
                         f"Missing RVineStructure methods: {missing}")


@unittest.skipUnless(HAS_PV, "pyvinecopulib not installed")
class TestFitControlsAPIParity(unittest.TestCase):
    """FitControls classes should have matching attributes."""

    SKIP_BICOP = {"nonparametric_grid_size"}
    SKIP_VINECOP = {"nonparametric_grid_size"}

    def test_bicop_controls(self):
        pv_attrs = {m for m in dir(pv.FitControlsBicop) if not m.startswith("_")} - self.SKIP_BICOP
        tv_inst = tv.FitControlsBicop()
        tv_attrs = {m for m in dir(tv_inst) if not m.startswith("_")}
        missing = pv_attrs - tv_attrs
        self.assertEqual(missing, set(),
                         f"Missing FitControlsBicop attrs: {missing}")

    def test_vinecop_controls(self):
        pv_attrs = {m for m in dir(pv.FitControlsVinecop) if not m.startswith("_")} - self.SKIP_VINECOP
        tv_inst = tv.FitControlsVinecop()
        tv_attrs = {m for m in dir(tv_inst) if not m.startswith("_")}
        missing = pv_attrs - tv_attrs
        self.assertEqual(missing, set(),
                         f"Missing FitControlsVinecop attrs: {missing}")


@unittest.skipUnless(HAS_PV, "pyvinecopulib not installed")
class TestModuleLevelParity(unittest.TestCase):
    """Module-level names should exist in torchvine."""

    # Names that are intentionally excluded
    SKIP = {"student", "benchmark", "pyvinecopulib_ext"}

    def test_module_names(self):
        pv_names = {m for m in dir(pv)
                    if not m.startswith("_") and m[0].islower()} - self.SKIP
        tv_names = {m for m in dir(tv) if not m.startswith("_") and m[0].islower()}
        missing = pv_names - tv_names
        self.assertEqual(missing, set(),
                         f"Missing module-level names: {missing}")


@unittest.skipUnless(HAS_PV, "pyvinecopulib not installed")
class TestFamilyEnumParity(unittest.TestCase):
    """BicopFamily enum should contain all pyvinecopulib families except student."""

    def test_families_match(self):
        pv_fams = {f.name for f in pv.BicopFamily}
        tv_fams = {f.name for f in tv.BicopFamily}
        self.assertEqual(pv_fams, tv_fams)


if __name__ == "__main__":
    unittest.main()
