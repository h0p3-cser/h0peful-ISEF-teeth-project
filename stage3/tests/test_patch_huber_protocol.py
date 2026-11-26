"""Regressions between the provisional material pilot and the frozen 1.0 protocol."""
import unittest

import numpy as np

from qlf_stage3.patch_pilot.freeze import combine,paired_blocks
from qlf_stage3.patch_pilot.protocol import split


def block(identity,x):
    return dict(patch_id=identity+str(x),physical_id=identity,polygons=[[[x,0],[x+4,0],[x+4,4],[x,4]]],
                confirmed=True,visibility="present")


class TestOfficialPatchProtocol(unittest.TestCase):
    def test_twenty_five_valid_pixels_and_one_vote_per_physical_block(self):
        raw=np.full((5,15,3),80,np.uint8)
        raw[:,10:]=185
        masks,obs=combine(raw,[block("grey",0),block("grey",5),block("white",10)])
        self.assertEqual(len(masks["grey"]),50)
        self.assertEqual(len(masks["white"]),25)
        pairs=paired_blocks(obs,obs)
        self.assertEqual(split(pairs),(["grey","white"],None,""))
        raw[0,10,0]=250
        _,short=combine(raw,[block("grey",0),block("white",10)])
        self.assertEqual(split(paired_blocks(obs,short))[2],"insufficient_paired_blocks")

    def test_essential_bright_block_cannot_be_held_out(self):
        pairs={"A":([.7]*3,[.65]*3),"B":([.05]*3,[.05]*3),"C":([.05]*3,[.055]*3)}
        self.assertEqual(split(pairs),(["A","C"],"B",""))

    def test_one_short_reference_channel_prevents_an_output(self):
        pairs={"dark":([.12]*3,[.1]*3),"light":([.87]*3,[.89,.9,.11])}
        self.assertEqual(split(pairs),([],None,"insufficient_channel_span"))

    def test_all_three_can_fit_without_a_feasible_holdout(self):
        p={"A":([.2,.2,.9],[.2,.2,.9]),"B":([.2,.9,.2],[.2,.9,.2]),"C":([.9,.2,.2],[.9,.2,.2])}
        self.assertEqual(split(p),(["A","B","C"],None,""))
