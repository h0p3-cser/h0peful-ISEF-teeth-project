"""Regression checks for failure accounting and independent whole-object colour QC."""
from types import SimpleNamespace
import unittest

import numpy as np

from qlf_stage3.files import load
from qlf_stage3.materials.patches import paired
from qlf_stage3.materials.pilot import PROTOCOL, coverage, patch_log
from qlf_stage3.materials.scenes import fit_pairs
from qlf_stage3.rgb.color import FrameIssue


def patch(name,material,role,value):
    return SimpleNamespace(region=name,material=material,role=role,linear=np.full(3,value),srgb=np.full(3,127.),
                           reason="",xx=np.arange(60),n_total=60,valid_fraction=1.,polygon=[[0,0],[8,0],[8,8]])


class TestMaterialPilot(unittest.TestCase):
    def test_heldout_object_cannot_satisfy_fit_gate(self):
        dark=patch("bed_1","bed","fit",.005)
        bright=patch("disc_1","disc","holdout",.6)
        pairs=[(dark,dark),(bright,bright)]
        trace={}
        with self.assertRaises(FrameIssue) as caught:
            fit_pairs(pairs,load(PROTOCOL),trace)
        self.assertEqual(caught.exception.code,"insufficient_matched_materials")
        self.assertEqual(trace["n_fit_materials"],1)
        self.assertAlmostEqual(trace["current_all_visible_range_R"],.595)

    def test_missing_anchor_patch_does_not_get_replaced(self):
        disc=patch("disc_1","disc","holdout",.1)
        rim=patch("rim_1","rim","fit",.6)
        current={disc.region:disc}
        anchor={disc.region:disc,rim.region:rim}
        pairs=paired(current,anchor)
        rows={r["region"]:r for r in patch_log(current,anchor,pairs)}
        self.assertEqual(len(pairs),1)
        self.assertEqual(rows["rim_1"]["reason"],"current_material_not_visible")
        self.assertFalse(rows["rim_1"]["paired"])
        self.assertEqual(rows["disc_1"]["fit_weight"],0.)

    def test_old_failure_and_identities_have_separate_denominators(self):
        row=SimpleNamespace(rgb_image_id="C",anchor_rgb_id="A",status="failed",old_status="ok")
        self.assertEqual(coverage(row),"lost")
        row.status,row.old_status="ok","failed"
        self.assertEqual(coverage(row),"recovered")
        row.rgb_image_id=row.anchor_rgb_id
        self.assertEqual(coverage(row),"anchor")


if __name__=="__main__":
    unittest.main()
