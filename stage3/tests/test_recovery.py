"""A lost non-reference S0 frame can be recovered and consumed by the training cohort."""
import csv
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from qlf_stage3.files import FEATURE_CSV, config, load, sha
from qlf_stage3.modeling.cohort import build_cohort


class RecoveryTests(unittest.TestCase):
    def test_missing_frame_and_numeric_map_recover_once(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            data, run = root / "data", root / "measurements"

            def cli(*args):
                result = subprocess.run([sys.executable, "-X", "utf8", "-m", "qlf_stage3", *map(str, args)],
                                        cwd=root, capture_output=True, text=True, encoding="utf-8")
                self.assertEqual(result.returncode, 0, result.stderr)

            cli("demo-data", "--out", data)
            cli("measure", "--data", data, "--out", run)
            cli("export", "--data", data, "--measurements", run)
            golden = sha(run / FEATURE_CSV)
            with (run / "frames.csv").open(encoding="utf-8-sig", newline="") as stream:
                frames = list(csv.DictReader(stream))
            lost = next(r for r in frames if r["state_id"] == "S0" and r["fit_source"] == "paired root Huber fit")
            (run / lost["map_path"]).unlink()
            with (run / "frames.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(lost))
                writer.writeheader()
                writer.writerows(r for r in frames if r["qlf_path"] != lost["qlf_path"])

            cli("export", "--data", data, "--measurements", run)
            audit = run / "training-features-255-thr30-verification.json"
            self.assertEqual(list(load(audit)["rescored_images"]), [lost["qlf_path"]])
            self.assertEqual(sha(run / FEATURE_CSV), golden)
            self.assertTrue((run / lost["map_path"]).is_file())
            manifest, _, _ = build_cohort(config(data, run / FEATURE_CSV))
            self.assertEqual(int(manifest.included.sum()), 54)
            cli("export", "--data", data, "--measurements", run)
            self.assertEqual(load(audit)["rescored_images"], {})
            self.assertEqual(load(audit)["restored_from_numeric_maps"], {})
            self.assertEqual(sha(run / FEATURE_CSV), golden)
