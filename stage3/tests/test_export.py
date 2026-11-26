import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from qlf_stage3 import export


class TrainingCSVTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.run = Path(tmp.name)
        (self.run / "maps").mkdir()
        self.source = dict(image_path="QLF/repeat.jpg", s0_image_path="QLF/anchor.jpg",
                           rgb_path="RGB/paired.jpg", tooth_id="T004", angle_id="A000",
                           state_id="S0", repeat_rank="1")
        self.frame = dict(qlf_path="QLF/repeat.jpg", anchor_path="QLF/anchor.jpg", status="ok",
                          root_tiles="18", slope="0.8", intercept="4", reference255="208",
                          fit_source="paired root Huber fit", valid_crown_px="4", mean_green="151.9",
                          mean_loss_pct="27.5", map_path="maps/repeat.npz")
        np.savez(self.run / self.frame["map_path"], loss_pct=[[-10., 30., 30., 60., np.nan]])
        self.threshold = dict(valid_crown_px="4", response_px="1", response_pct="25.0")

    def test_signed_mean_and_strict_boundary_use_the_frame_denominator(self):
        rec = export.cached_measurement(self.run, self.source, self.frame, self.threshold)
        self.assertEqual((rec.mean, rec.area, rec.restored_fields), ("27.5", "25.0", ()))
        rec = export.cached_measurement(self.run, self.source, dict(self.frame, mean_loss_pct=""),
                                        dict(self.threshold, response_px="3", response_pct="75.0"))
        self.assertEqual((float(rec.mean), float(rec.area)), (27.5, 25.0))
        self.assertEqual(rec.restored_fields, ("mean_loss_pct", "response_pct"))

    def test_non_anchor_s0_needs_its_own_fit(self):
        with self.assertRaisesRegex(ValueError, "per-frame paired root fit"):
            export.cached_measurement(self.run, self.source, dict(self.frame, fit_source="reference exposure"), None)
        with self.assertRaisesRegex(ValueError, "wrong full-scale reference"):
            export.cached_measurement(self.run, self.source, dict(self.frame, reference255="255"), None)

    def test_missing_old_s0_row_falls_back_and_old_loss_is_ignored(self):
        path = self.source["image_path"]
        single = dict(self.source, sample_id="repeat", qlf_path=path, s0_reference_path=self.source["s0_image_path"],
                      single_image_status="ok", p10_intensity="70", G_texture_entropy="3.7")
        upstream = ("single-image", {path: single})
        row, name = export.single_features(self.source, [("old", {}), upstream])
        self.assertEqual((row, name), (single, "single-image"))
        old = dict(single, huber_delta_f_pct="", loss_area_gt25_pct="")
        row, name = export.single_features(self.source, [("old", {path: old}), upstream])
        self.assertEqual((row, name), (old, "old"))
        with self.assertRaisesRegex(ValueError, "rgb_path"):
            export.single_features(self.source, [("foreign", {path: dict(single, rgb_path="RGB/wrong.jpg")})])

    def test_only_uncached_frames_reach_the_scorer(self):
        path = self.source["image_path"]
        missing = "QLF/missing.jpg"
        rows = {path: self.source, missing: dict(self.source, image_path=missing, repeat_rank="3")}
        frame = {**self.frame, **{k: self.source[k] for k in export.GROUP}}
        export_file = self.run / "frames.csv"
        with export_file.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=frame)
            writer.writeheader()
            writer.writerow(frame)
        with patch.object(export, "rescore_missing", return_value={missing: {"status": "excluded", "reason": "qc"}}) as scorer:
            accepted, excluded, needs_score, restored, _ = export.measurements(self.run, self.run, {}, rows, self.run)
        self.assertEqual(scorer.call_args.args[0], {missing: "missing 255 frame"})
        self.assertEqual(set(accepted), {path})
        self.assertEqual(excluded, {missing: "qc"})
        self.assertEqual(set(needs_score), {missing})
        self.assertEqual(restored, {path: ("response_pct",)})


if __name__ == "__main__":
    unittest.main()
