import unittest

import numpy as np
from skimage.color import lab2rgb, rgb2lab

from qlf_stage3.bstar.color import apply_offset, region_offset
from qlf_stage3.bstar.gray_audit import difference_stats, histogram


class YellowBlueOffsetTests(unittest.TestCase):
    def test_region_votes_and_strict_encoded_limits(self):
        im = np.array([[[150,140,100]]*7 + [[90,100,120],[5,160,170],[90,250,200]]],np.uint8)
        regions = [dict(box=[0,0,7,1]), dict(box=[7,0,10,1]), dict(box=[8,0,10,1])]
        m, detail = region_offset(im,regions)
        expected = rgb2lab(im[0,[0,7]])[:,2].mean()
        self.assertAlmostEqual(m,expected,places=7)
        self.assertEqual([r['valid_pixels'] for r in detail],[7,1,0])
        self.assertIsNone(region_offset(im,regions[2:])[0])
        self.assertIsNone(region_offset(im,[])[0])

    def test_single_quantization_and_axis_sign(self):
        rgb = np.random.default_rng(26).integers(32,220,(102,101,3),dtype=np.uint8)
        for shift in [0,-5.25,4.75]:
            lab = rgb2lab(rgb)
            lab[...,2] -= shift
            expected = np.rint(lab2rgb(lab)*255).astype(np.uint8)
            np.testing.assert_array_equal(apply_offset(rgb,shift),expected)

    def test_native_histogram_percentile_including_interpolation(self):
        diff = np.arange(24,dtype=np.uint8).reshape(4,6)
        mask = diff % 2 == 0
        stats = difference_stats(histogram(diff,mask))
        self.assertEqual(stats['pixels'],mask.sum())
        self.assertAlmostEqual(stats['mean_abs'],diff[mask].mean())
        self.assertAlmostEqual(stats['p95_abs'],np.percentile(diff[mask],95))


if __name__ == '__main__':
    unittest.main()
