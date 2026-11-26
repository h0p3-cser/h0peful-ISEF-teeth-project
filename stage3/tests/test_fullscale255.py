import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from qlf_stage3.fullscale255 import analyze_curves, count_curve, loss_map_255


def frame(tooth, angle, stage, rank, percent, size=100):
    counts = np.zeros(101, dtype=int)
    counts[:3] = np.array([100, percent, percent - 10]) * size // 100
    return dict(tooth_id=tooth, angle_id=angle, state_id=stage, repeat_rank=rank,
                qlf_path=f"{tooth}/{angle}/{stage}/{rank}", status="ok",
                valid_crown_px=size, counts=counts)


class FullScaleTests(unittest.TestCase):
    def test_signed_affine_loss_and_validity(self):
        baseline = np.array([[40., 100., 190.]])
        roi = np.ones_like(baseline, dtype=bool)
        for a, b in ((1., 0.), (.8, 20.), (1.2, -15.)):
            np.testing.assert_allclose(loss_map_255(a * baseline + b, roi, a, b),
                                       100. * (255. - baseline) / 255.)
        green = np.array([[15., 16., 220., 255., np.nan]])
        loss = loss_map_255(green, np.ones_like(green, dtype=bool), .8, 0.)
        self.assertTrue(np.isfinite(loss[0, 1]) and loss[0, 2] < 0)
        self.assertTrue(np.isnan(loss[0, [0, 3, 4]]).all())
        n, counts = count_curve(np.array([[-3., 0., 25., 25., 100., 101., np.nan]]), [0, 25, 100])
        self.assertEqual((n, counts.tolist()), (6, [4, 2, 1]))

    def test_equal_view_weights_lower_tie_and_omitted_teeth(self):
        frames = []
        for tooth in ("T1", "T2", "T3"):
            for i, stage in enumerate(("S0", "S1", "S2")):
                shift = 10 * i
                frames.extend([frame(tooth, "A", stage, 1, 30 + shift),
                               frame(tooth, "A", stage, 2, 50 + shift, size=1000),
                               frame(tooth, "B", stage, 1, 60 + shift)])
            frames.append(frame(tooth, "unmatched", "S2", 1, 95))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            selection = analyze_curves(folder, frames, np.arange(101))
            self.assertEqual(selection["matched_views"], 6)
            self.assertEqual(selection["threshold_pct"], 1)
            self.assertEqual(selection["threshold_when_each_tooth_is_omitted"], dict.fromkeys(("T1", "T2", "T3"), 1))
            with (folder / "stage-curves.csv").open(encoding="utf-8-sig", newline="") as stream:
                medians = [float(r["median"]) for r in csv.DictReader(stream) if r["threshold_pct"] == "1"]
            self.assertEqual(medians, [50., 60., 70.])

    def test_unsupported_threshold_is_empty(self):
        frames = [frame(t, "A", s, 1, p) for t in ("T1", "T2")
                  for s, p in (("S0", 40), ("S1", 60), ("S2", 50))]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for sample in ([], frames[:3], frames):
                self.assertIsNone(analyze_curves(folder, sample, np.arange(101))["threshold_pct"])


if __name__ == "__main__":
    unittest.main()
