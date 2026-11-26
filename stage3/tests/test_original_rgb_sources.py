import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
from PIL import Image

from qlf_stage3.files import sha
from qlf_stage3.modeling.cohort import FEATURES, save_csv
from qlf_stage3.modeling.original_rgb import bind


class OriginalRgbSourceTest(unittest.TestCase):
    def test_same_id_delivery_copy_only_enters_explicit_received_run(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            incoming = root / 'received'
            incoming.mkdir()
            original = root / 'camera.jpg'
            supplied = incoming / 'example_I0001.jpg'
            Image.new('RGB',(12,12),(110,93,75)).save(original)
            Image.new('RGB',(12,12),(100,98,90)).save(supplied)
            photo = dict(sample_id='a',tooth_id='T005',state_id='S2',angle_id='A000',repeat_rank=1,
                         qlf_path='qlf.jpg',rgb_path='rgb.jpg')
            row = dict(photo, rgb_image_id='I0001',qlf_image_id='I0002',rgb_catalog_tooth='T005',
                       rgb_sha256=sha(supplied),rgb_event='r',qlf_event='q',group_id='g',group_frames=1,
                       test_fold=4,session_id='C04',geometry_version='1',fit_source='f',included=True)
            save_csv(root,'sample_manifest.csv',pd.DataFrame([row]))
            save_csv(root,'groups.csv',pd.DataFrame([dict(group_id='g',member_ids=json.dumps(['a']))]))
            save_csv(root,'features.csv',pd.DataFrame([dict(photo,**dict.fromkeys(FEATURES,1.0))]))
            intake = dict(image_id='I0001',modality='RGB',state_id='S2',angle_id='A000',session_id='C04',tooth_id='T005',
                          sha256=sha(supplied),source_camera_sha256=sha(original),source_storage='delivery_copy',
                          pixel_processing='recorded_white_balance',original_backup_relative_path='received/example_I0001.jpg',
                          duplicate_of='',source_filename='camera.jpg')
            save_csv(root,'intake.csv',pd.DataFrame([intake]))

            def resolve(name,supplements,delivery_folder=None):
                out = root / name
                out.mkdir()
                frame = bind(out,root,root/'features.csv',root,root/'intake.csv',supplements,delivery_folder)
                return frame.iloc[0]

            strict = resolve('strict',[incoming])
            self.assertFalse(strict.included)
            delivery = resolve('delivery',[incoming],incoming)
            self.assertTrue(delivery.included)
            self.assertFalse(delivery.matches_original)
            self.assertEqual(delivery.rgb_sha256,sha(supplied))
            recovered = resolve('recovered',[original],incoming)
            self.assertTrue(recovered.matches_original)
            self.assertEqual(recovered.rgb_sha256,sha(original))
            # A file with the right ID but a third encoding is not the recorded delivery either.
            Image.new('RGB',(12,12),(110,110,90)).save(supplied)
            self.assertFalse(resolve('unrecognized',[incoming],incoming).included)
