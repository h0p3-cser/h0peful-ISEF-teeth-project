"""Install the supplied 35 T005/S2 white-light JPEGs in the existing project layout."""
import argparse
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image

AS_SUPPLIED = "original_as_supplied"


def replace_bytes(path, data):
    with NamedTemporaryFile(dir=path.parent, prefix=".t005-", delete=False) as handle:
        pending = Path(handle.name)
        handle.write(data)
    try:
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def update_table(path, digests, *, catalog=False):
    raw = path.read_bytes()
    encoding = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
    rows = read_rows(path)
    covered = set()
    changed = False
    for row in rows:
        if row["image_id"] in digests:
            covered.add(row["image_id"])
            new = {"sha256":digests[row["image_id"]]}
            if catalog:
                new["pixel_processing"] = AS_SUPPLIED
            if any(row[k] != v for k,v in new.items()):
                row.update(new)
                changed = True
    if covered != set(digests):
        raise ValueError(f"Missing target image ID in {path}")
    if not changed:
        return
    dest = io.StringIO(newline="")
    writer = csv.DictWriter(dest, fieldnames=rows[0], lineterminator="\r\n" if b"\r\n" in raw else "\n")
    writer.writeheader()
    writer.writerows(rows)
    data = dest.getvalue().encode(encoding)
    if data != raw:
        replace_bytes(path, data)


def install(project, source, receipt):
    formal = project / "data/formal_t002_t010"
    work = project / "data/stage3_author_workspace"
    portable = project / "stage3/data/study"
    catalog = formal / "images.csv"
    pairs = [formal / "pairs.csv", portable / "catalog/pairs.csv"]
    pair_hashes = [sha256(p.read_bytes()).hexdigest() for p in pairs]
    records = [r for r in read_rows(catalog) if (r["tooth_id"],r["state_id"],r["session_id"],r["modality"])
               == ("T005","S2","C04","RGB")]
    if len(records) != 35 or {r["angle_id"] for r in records} != {"A000","A045","A090","A135","A180"}:
        raise ValueError("Expected 35 cataloged T005 white-light images across the five C04 views")
    bindings = read_rows(work / "local-photo-bindings.csv")
    digests, plans, log = {}, [], []
    portable_files = {"catalog/images.csv","workspace/local-photo-bindings.csv"}
    for r in records:
        key = r["image_id"]
        canonical = formal / r["normalized_relative_path"]
        path = source / r["angle_id"] / "RGB" / canonical.name
        raw = path.read_bytes()
        digest = sha256(raw).hexdigest()
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "JPEG" or image.mode != "RGB" or image.size != (int(r["width"]),int(r["height"])):
                raise ValueError(f"Unexpected image format or size for {key}")
        aliases = [b["workspace_path"] for b in bindings if b["image_id"] == key and b["modality"] == "RGB"]
        if len(set(aliases)) != 2:
            raise ValueError(f"Unexpected workspace entries for {key}")
        destinations = [canonical] + [base / name for base in (work,portable/"workspace") for name in aliases]
        for d in destinations:
            if not d.resolve().is_relative_to(project) or d.name != canonical.name or not d.is_file():
                raise ValueError(f"Invalid existing project image path: {d}")
            if d.is_relative_to(portable):
                portable_files.add(d.relative_to(portable).as_posix())
        digests[key] = digest
        plans.append((raw, destinations))
        log.append(dict(image_id=key,angle_id=r["angle_id"],repeat_rank=r["repeat_rank"],
                        supplied_sha256=digest,previous_project_sha256=sha256(canonical.read_bytes()).hexdigest(),
                        pixel_processing=AS_SUPPLIED,
                        installed_paths=[d.relative_to(project).as_posix() for d in destinations]))
    if len(digests) != len(records) or any(sum(r["angle_id"] == a for r in records) != 7
                                             for a in {r["angle_id"] for r in records}):
        raise ValueError("Expected seven different image IDs at each view")
    index = portable/"files.sha256"
    checks = index.read_text(encoding="utf8").splitlines()
    if not portable_files.issubset({line.split("  ",1)[1] for line in checks}):
        raise ValueError("Project data package does not list all target files")
    for raw, destinations in plans:
        for target in destinations:
            replace_bytes(target, raw)
            if target.read_bytes() != raw:
                raise OSError(f"Installed JPEG differs from incoming bytes: {target}")
    for target in (catalog,portable/"catalog/images.csv"):
        update_table(target,digests,catalog=True)
    for target in (work/"local-photo-bindings.csv",portable/"workspace/local-photo-bindings.csv"):
        update_table(target,digests)
    new_checks = []
    for line in checks:
        _,name = line.split("  ",1)
        if name in portable_files:
            line = sha256((portable/name).read_bytes()).hexdigest()+"  "+name
        new_checks.append(line)
    if new_checks != checks:
        replace_bytes(index, ("\n".join(new_checks)+"\n").encode("utf8"))
    if pair_hashes != [sha256(p.read_bytes()).hexdigest() for p in pairs]:
        raise RuntimeError("Pairing indexes changed during installation")
    receipt.parent.mkdir(exist_ok=True,parents=True)
    result = dict(adoption="provided originals for this project", images=len(log), verified_installed_files=sum(len(d) for _,d in plans),
                  unchanged_image_bytes=sum(r["supplied_sha256"] == r["previous_project_sha256"] for r in log),
                  target_status=AS_SUPPLIED, pair_index_sha256=pair_hashes, records=log)
    replace_bytes(receipt, (json.dumps(result, ensure_ascii=False, indent=2)+"\n").encode("utf8"))
    print(f"Installed {len(log)} supplied JPEGs in {result['verified_installed_files']} existing locations, verified both pairing indexes and each JPEG.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True, help="Legacy workspace containing data/formal_t002_t010 and stage3/data/study")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    install(args.project.resolve(),args.source.resolve(),args.receipt.resolve())
