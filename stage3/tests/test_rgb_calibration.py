import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image, ImageCms

from qlf_stage3.files import load, sha
from qlf_stage3.rgb.calibration import (DEFAULT_CONFIG, calibrate_photo, prepare_reference)
from qlf_stage3.rgb.color import FrameIssue, decode, fit_channels, save_png
from qlf_stage3.rgb.geometry import orientation_basis, transform
from qlf_stage3.rgb.report import MEASURES, equal_units
from qlf_stage3.rgb.study import Photo, select_anchor
from qlf_stage3.rgb.verify import synthetic


class RgbCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load(DEFAULT_CONFIG)
        self.cfg["annotation_version"] = "synthetic-test"

    def test_direct_colour_and_ciede_reference(self):
        self.assertTrue(synthetic(self.cfg))
        x = np.linspace(.1,.8,50)[:,None]*np.ones((1,3))
        narrow = x.copy()
        narrow[:,2] = .1
        for a,b in ((narrow,x),(x,narrow)):
            with self.assertRaisesRegex(FrameIssue,"insufficient_channel_span"):
                fit_channels(a,b,self.cfg)
        with self.assertRaisesRegex(FrameIssue,"insufficient_fit_tiles"):
            fit_channels(x[:10],x[:10],self.cfg)
        with self.assertRaises(FrameIssue) as invalid:
            fit_channels(x,.9-x,self.cfg)
        self.assertEqual(invalid.exception.code,"invalid_huber_parameters")

    def test_exif_geometry_and_profile(self):
        with TemporaryDirectory() as directory:
            dest = Path(directory)/"oriented.png"
            raw = np.arange(5*8*3,dtype=np.uint8).reshape(5,8,3)
            profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
            for orientation in range(1,9):
                exif = Image.Exif()
                exif[274] = orientation
                Image.fromarray(raw).save(dest,exif=exif,icc_profile=profile)
                decoded,info = decode(dest)
                x,y = transform([[2,3]],orientation_basis(8,5,orientation))[0].astype(int)
                np.testing.assert_array_equal(decoded[y,x],raw[3,2])
                self.assertEqual(info["color_interpretation"],"icc_to_srgb")
            Image.fromarray(raw).save(dest,icc_profile=b"broken")
            with self.assertRaisesRegex(FrameIssue,"cannot|profile|open"):
                decode(dest)
            save_png(dest,raw)
            np.testing.assert_array_equal(decode(dest,corrected=True)[0],raw)

    def test_equal_event_weight_keeps_worsening_frames(self):
        rows = []
        for tooth,group,values in [("A","one",[1]*6),("A","two",[-5]),("B","three",[.5,0,.5])]:
            rows.extend(dict(tooth_id=tooth,group_id=group,**dict.fromkeys(MEASURES,v)) for v in values)
        events = equal_units(pd.DataFrame(rows),"group_id")
        teeth = equal_units(events,"tooth_id",parent="group_id").set_index("tooth_id")
        self.assertEqual(teeth.loc["A","de00_gain"],-2)
        self.assertEqual(teeth.loc["A","n_frames"],7)
        self.assertEqual(teeth.loc["A","improved_frames"],6)
        self.assertEqual(teeth.de00_gain.mean(),-.75)

    def test_calibration_statuses_and_reference_priority(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            (out/"corrected_rgb").mkdir()
            yy,xx = np.mgrid[:280,:480]
            body = (xx>35)&(xx<445)&(yy>38)&(yy<245)
            base = np.where(body,55+((xx*2+yy)//4)%136,0)
            raw = np.stack([base,base*.83,base*.61],axis=-1).astype(np.uint8)
            for name in ("anchor","current"):
                save_png(out/(name+".png"),raw)
            mask = out/"mask.png"
            Image.fromarray((255*body).astype("uint8")).save(mask)
            def photo(name):
                path = out/(name+".png")
                return Photo(name,path,path.name,sha(path),mask,mask.name,name,{},"calibration")
            anchor,cur = photo("anchor"),photo("current")
            annotation = {anchor.image_id:dict(sha256=anchor.digest,root=[[45,50],[330,50],[330,230],[45,230]],
                            crown=[[345,50],[431,50],[431,230],[345,230]])}
            self.assertEqual(select_anchor([anchor],cur,[],annotation)[0],anchor)
            self.assertEqual(select_anchor([],cur,[],annotation)[0],cur)
            self.assertEqual(select_anchor([],None,[(anchor,2),(cur,3)],dict(annotation,current={}))[0],anchor)
            self.assertEqual(select_anchor([],None,[],annotation),(None,"no_anchor"))
            with self.assertRaisesRegex(FrameIssue,"no_anchor"):
                prepare_reference(None,annotation,[480,280],self.cfg)
            with self.assertRaisesRegex(FrameIssue,"root"):
                prepare_reference(anchor,{},[480,280],self.cfg)
            ref = prepare_reference(anchor,annotation,[480,280],self.cfg)
            identity,_,_ = calibrate_photo(anchor,ref,self.cfg,out,"test")
            self.assertEqual(identity["status"],"reference")
            self.assertEqual(identity["source_pixel_sha256"],identity["corrected_pixel_sha256"])
            for reason in ("no_anchor","root_not_visible"):
                failed,_,_ = calibrate_photo(cur,FrameIssue(reason),self.cfg,out,"test")
                self.assertEqual((failed["status"],failed["reason"],failed["corrected_path"]),("failed",reason,""))
            with patch("qlf_stage3.rgb.calibration.align",side_effect=FrameIssue("registration_failed")):
                failed,_,_ = calibrate_photo(cur,ref,self.cfg,out,"test")
                self.assertEqual((failed["reason"],failed["corrected_path"]),("registration_failed",""))
            self.assertFalse((out/"corrected_rgb/current.png").exists())
            aligned = np.eye(3),None,None,dict(method="synthetic_known_transform")
            with patch("qlf_stage3.rgb.calibration.align",return_value=aligned):
                limited = dict(self.cfg,min_linear_span=1.2)
                failed,_,_ = calibrate_photo(cur,ref,limited,out,"test")
                self.assertEqual((failed["reason"],failed["corrected_path"]),("insufficient_channel_span",""))
                marked,qc,tiles = calibrate_photo(cur,ref,dict(self.cfg,min_holdout_tiles=10000),out,"test")
            self.assertEqual((marked["status"],marked["reason"]),("ok_no_holdout","insufficient_holdout_tiles"))
            self.assertTrue((out/marked["corrected_path"]).is_file())
            self.assertNotIn("before_de00",qc)
            self.assertTrue(tiles.role.eq("holdout").any())
