import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from qlf_stage3.bundle import check_data, write_checksums
from qlf_stage3.files import dataset_info, dump, outside_inputs
from qlf_stage3.modeling.cohort import FEATURES
from qlf_stage3.modeling.training import group_features


class PortableTests(unittest.TestCase):
    def test_moved_dataset_and_changed_input(self):
        with TemporaryDirectory() as temp:
            root = Path(temp) / "first"
            dump(root / "dataset.json", {"schema": 1, "kind": "synthetic"})
            item = "冠区 mask.bin"
            (root / item).write_bytes(b"reviewed")
            write_checksums(root)
            root = root.rename(Path(temp) / "new location")
            self.assertEqual(check_data(root), {"kind": "synthetic", "verified_files": 2})
            (root / item).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                check_data(root)

    def test_output_cannot_nest_inside_inputs(self):
        with TemporaryDirectory() as temp:
            root = Path(temp) / "input"
            with self.assertRaisesRegex(ValueError, "outside"):
                outside_inputs(root / "outputs", root)
            self.assertEqual(outside_inputs(Path(temp) / "run", root), Path(temp) / "run")
            with self.assertRaisesRegex(ValueError, "Missing"):
                dataset_info(root)

    def test_both_modalities_use_the_same_acquisition_members(self):
        rows = []
        for group, name, value in [("second", "z", 20), ("first", "b", 8), ("first", "a", 2)]:
            rows.append(dict(tooth_id="t", state_id="S0", session_id="s", angle_id="a", geometry_version="g",
                             rgb_event=group, qlf_event=group, sample_id=name, group_id=group, rgb_image_id=name,
                             rgb_resolved=name+".png", **dict.fromkeys(FEATURES, value)))
        vectors = np.array([[20, 20], [8, 8], [2, 2]], dtype=np.float32)
        groups, white, qlf = group_features(pd.DataFrame(rows), vectors)
        self.assertEqual(groups.group_frames.tolist(), [2, 1])
        self.assertEqual(groups.representative_id.tolist(), ["a", "z"])
        np.testing.assert_array_equal(white[:, 0], [5, 20])
        np.testing.assert_array_equal(qlf[:, 0], [5, 20])
