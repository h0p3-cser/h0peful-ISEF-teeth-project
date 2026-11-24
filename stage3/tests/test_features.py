import unittest
import numpy as np

from qlf_stage3.assemble import assemble
from qlf_stage3.huber import calibrated_loss
from qlf_stage3.huber_fit import fit
from qlf_stage3.pixels import intensity
from qlf_stage3.texture import texture


class FeatureTests(unittest.TestCase):
    def test_masked_intensity_and_texture(self):
        green = np.array([[64, 64, 255], [64, 64, 255]], dtype=np.uint8)
        crown = green == 64
        self.assertEqual(intensity(green, crown)['mean_intensity'], 64)
        self.assertEqual(tuple(texture(green, crown).values()), (0, 1, 1, 0))

    def test_threshold_excludes_exact_boundary_and_invalid_denominator(self):
        baseline = np.array([100., 100., 100., 10.])
        measured = np.array([74., 75., 110., 8.])
        _, valid, loss = calibrated_loss(np.zeros(4), baseline, measured, 1., 0.)
        np.testing.assert_array_equal(valid, [True, True, True, False])
        np.testing.assert_allclose(loss, [26., 25., -10., np.nan])
        self.assertEqual(int((loss > 25).sum()), 1)

    def test_affine_fit_with_outliers(self):
        x = np.linspace(30, 210, 80)
        y = 0.8 * x + 9 + 0.15 * np.sin(x)
        y[::17] += 45
        slope, intercept = fit(x, y, 'huber')
        self.assertAlmostEqual(slope, 0.8, delta=0.01)
        self.assertAlmostEqual(intercept, 9, delta=1)

    def test_assembly_checks_identity_and_preserves_blanks(self):
        identity = [{'sample_id': 'a', 'qlf_path': 'a.jpg'}]
        observed = {'a.jpg': {'mean_intensity': 50}}
        paired = {'a.jpg': {'huber_delta_f_pct': ''}}
        self.assertEqual(assemble(identity, observed, paired)[0]['huber_delta_f_pct'], '')
        with self.assertRaises(ValueError):
            assemble(identity, observed, {'other.jpg': paired['a.jpg']})
