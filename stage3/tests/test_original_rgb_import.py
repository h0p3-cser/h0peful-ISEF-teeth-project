import csv
from hashlib import sha256
import importlib.util
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

script = Path(__file__).resolve().parents[1]/"experiments/import_t005_original_rgb.py"
spec = importlib.util.spec_from_file_location("project_import",script)
project_import = importlib.util.module_from_spec(spec)
spec.loader.exec_module(project_import)


def write_table(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def jpeg(color):
    output = BytesIO()
    Image.new("RGB",(12,9),color).save(output,format="JPEG")
    return output.getvalue()


class ProjectOriginalImportTests(unittest.TestCase):
    def test_different_bytes_replace_every_copy_without_changing_references(self):
        with TemporaryDirectory() as temp:
            project = Path(temp).resolve()
            formal = project/"data/formal_t002_t010"
            work = project/"data/stage3_author_workspace"
            portable = project/"stage3/data/study"
            incoming = project/"received"
            before,after = jpeg((30,55,80)),jpeg((100,80,54))
            original_digest = sha256(before).hexdigest()
            rows,bindings = [],[]
            for angle in ("A000","A045","A090","A135","A180"):
                for rank in range(1,8):
                    image_id = f"I{len(rows)+1:04d}"
                    name = f"photo_{image_id}.jpg"
                    indexed = Path("01_images/T005/S2/C04")/angle/"RGB"/name
                    record = dict(image_id=image_id,tooth_id="T005",state_id="S2",session_id="C04",angle_id=angle,
                                  modality="RGB",repeat_rank=rank,width=12,height=9,normalized_relative_path=indexed.as_posix(),
                                  sha256=original_digest,source_camera_sha256=original_digest,pixel_processing="previous")
                    rows.append(record)
                    targets = [formal/indexed]
                    for prefix in ("Data","Data_rgb_corrected"):
                        alias = (Path(prefix)/"S2/T005"/angle/"RGB"/name).as_posix()
                        bindings.append(dict(image_id=image_id,workspace_path=alias,modality="RGB",sha256=original_digest))
                        targets += [work/alias,portable/"workspace"/alias]
                    for p in targets:
                        p.parent.mkdir(parents=True,exist_ok=True)
                        p.write_bytes(before)
                    supplied = incoming/angle/"RGB"/name
                    supplied.parent.mkdir(parents=True,exist_ok=True)
                    supplied.write_bytes(after)
            for path in (formal/"images.csv",portable/"catalog/images.csv"):
                write_table(path,rows)
            for path in (work/"local-photo-bindings.csv",portable/"workspace/local-photo-bindings.csv"):
                write_table(path,bindings)
            pair_bytes = b"pair_id,rgb_image_id\r\np1,I0001\r\n"
            for path in (formal/"pairs.csv",portable/"catalog/pairs.csv"):
                path.write_bytes(pair_bytes)
            index = portable/"files.sha256"
            index.write_text("".join(sha256(p.read_bytes()).hexdigest()+"  "+p.relative_to(portable).as_posix()+"\n"
                                     for p in sorted(portable.rglob("*")) if p.is_file()),encoding="utf8")
            receipt = project/"receipt.json"
            project_import.install(project,incoming,receipt)
            result = json.loads(receipt.read_text(encoding="utf8"))
            self.assertEqual(result["unchanged_image_bytes"],0)
            self.assertEqual(result["verified_installed_files"],175)
            for record,previous in zip(project_import.read_rows(formal/"images.csv"),rows):
                self.assertEqual(record["normalized_relative_path"],previous["normalized_relative_path"])
                self.assertEqual(record["source_camera_sha256"],original_digest)
                self.assertEqual(record["pixel_processing"],project_import.AS_SUPPLIED)
                self.assertEqual(record["sha256"],sha256(after).hexdigest())
            self.assertTrue(all((project/p).read_bytes()==after for r in result["records"] for p in r["installed_paths"]))
            self.assertEqual((formal/"pairs.csv").read_bytes(),pair_bytes)
            for line in index.read_text(encoding="utf8").splitlines():
                digest,name = line.split("  ",1)
                self.assertEqual(digest,sha256((portable/name).read_bytes()).hexdigest())
