"""Export one paired QLF/RGB row per QC-passing frame at a fixed 30% pixel loss."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import BadZipFile

import numpy as np

from .files import FEATURE_CSV as CSV_NAME, SOURCE_TABLE, dump

STATES = ("S0", "S1", "S2")
THRESHOLD = 30
GROUP = ("tooth_id", "state_id", "angle_id", "repeat_rank")
SINGLE = ("p10_intensity", "G_texture_entropy")
FEATURES = ("huber_delta_f_pct", "loss_area_gt30_pct", *SINGLE)
COLUMNS = ("sample_id", *GROUP, "qlf_path", "rgb_path", *FEATURES)
REPORT_NAME = "training-features-255-thr30-verification.json"


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def indexed(rows, key, name):
    result = {}
    for row in rows:
        value = row[key]
        if not value or value in result:
            raise ValueError(f"{name}: empty or repeated {key}: {value!r}")
        result[value] = row
    return result


def same_fields(row, expected, keys):
    for key in keys:
        if row[key] != expected[key]:
            raise ValueError(f"{expected['image_path']}: inconsistent {key}: {row[key]!r} / {expected[key]!r}")


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def close(value, expected):
    return finite(value) and math.isclose(float(value), expected, rel_tol=1e-12, abs_tol=1e-10)


def map_values(path):
    with np.load(path, allow_pickle=False) as saved:
        loss = saved["loss_pct"]
    if loss.ndim != 2 or np.isinf(loss).any():
        raise ValueError(f"Invalid signed loss map: {path.name}")
    valid = loss[np.isfinite(loss)]
    if valid.size == 0:
        raise ValueError(f"Empty valid crown: {path.name}")
    count = int(np.count_nonzero(valid > THRESHOLD))
    return int(valid.size), count, float(valid.mean(dtype=np.float64)), 100.0 * count / valid.size


@dataclass(frozen=True)
class Measurement:
    mean: str
    area: str
    map_file: Path
    fit_source: str
    restored_fields: tuple


def cached_measurement(folder, source, frame, threshold):
    path = source["image_path"]
    if frame["status"] != "ok" or frame["anchor_path"] != source["s0_image_path"]:
        raise ValueError("missing accepted 255 fit or wrong S0 reference")
    a, b, reference, green = (float(frame[k]) for k in ("slope", "intercept", "reference255", "mean_green"))
    if not np.isfinite([a, b, reference, green]).all() or a <= 0 or not 16 <= green < 255:
        raise ValueError("invalid exposure fit or valid-crown mean green")
    if path == frame["anchor_path"]:
        if source["state_id"] != "S0" or (a, b) != (1, 0) or frame["fit_source"] != "reference exposure":
            raise ValueError("S0 reference is not the identity exposure")
    elif frame["fit_source"] != "paired root Huber fit" or int(frame["root_tiles"]) < 12:
        raise ValueError("missing per-frame paired root fit")
    if not close(reference, 255*a + b):
        raise ValueError("wrong full-scale reference")
    relative = Path("maps") / (Path(path).stem + ".npz")
    if Path(frame["map_path"]) != relative:
        raise ValueError("loss map belongs to a different image or is missing")
    map_file = folder / relative
    n, count, mean, area = map_values(map_file)
    if n != int(frame["valid_crown_px"]) or not close(mean, 100.0 * (reference - green) / (255*a)):
        raise ValueError("numeric map disagrees with the per-frame 255 formula or denominator")

    restored = []
    mean_text = frame.get("mean_loss_pct")
    if not close(mean_text, mean):
        mean_text = repr(mean)
        restored.append("mean_loss_pct")
    if (threshold and finite(threshold.get("valid_crown_px")) and float(threshold["valid_crown_px"]) == n
            and finite(threshold.get("response_px")) and float(threshold["response_px"]) == count
            and close(threshold.get("response_pct"), area) and 0 <= float(threshold["response_pct"]) <= 100):
        area_text = threshold["response_pct"]
    else:
        area_text = repr(area)
        restored.append("response_pct")
    return Measurement(str(mean_text), str(area_text), map_file, frame["fit_source"], tuple(restored))


def rescore_missing(paths, sources, root, status, scratch):
    """Prepare only affected views and invoke the development scorer only for the requested frames."""
    from . import fullscale255
    from .crown import geometry
    from .dataset import DATA, cv2, inventory
    from .direct_features import indexes

    if DATA.resolve() != (root / "Data").resolve():
        raise ValueError("Active Stage 3 data root differs from the 255 run")
    (scratch / "maps").mkdir()
    group_reader = None
    if status.get("author_transforms_sha256"):
        bundle = root / "stage3-author-transforms.zip"
        if hashlib.sha256(bundle.read_bytes()).hexdigest() != status["author_transforms_sha256"]:
            raise ValueError("Reviewed transforms differ from the 255 run")
        group_reader, _ = fullscale255.reviewed_registration_group(bundle, sources, scratch)
    groups, configs = inventory(), geometry()
    legacy, index, held = indexes()
    views = sorted({(sources[p]["tooth_id"], sources[p]["angle_id"]) for p in paths})

    cv2.setNumThreads(1)
    records = []
    for view in views:
        records.extend(fullscale255.measure_view(view, groups, legacy, index, held, configs, sources, scratch,
                                                scratch, (THRESHOLD,), registration_group=group_reader, wanted=paths))
    recovered = indexed(records, "qlf_path", "targeted 255 rescore")
    if recovered.keys() != paths.keys():
        raise ValueError("Targeted 255 rescore did not cover the requested image IDs")
    return recovered


def measurements(run, root, status, sources, scratch):
    frame_file, threshold_file = run / "frames.csv", run / "frame-thresholds.csv"
    frames = indexed(csv_rows(frame_file), "qlf_path", frame_file) if frame_file.is_file() else {}
    selected = (r for r in csv_rows(threshold_file)
                if finite(r.get("threshold_pct")) and float(r["threshold_pct"]) == THRESHOLD)
    thresholds = indexed(selected, "qlf_path", threshold_file) if threshold_file.is_file() else {}
    if (frames.keys() | thresholds.keys()) - sources.keys():
        raise ValueError("255 tables contain images outside the reviewed inventory")
    accepted, excluded, needs_score, restored = {}, {}, {}, {}
    for path, source in sources.items():
        frame, threshold = frames.get(path), thresholds.get(path)
        if threshold is not None:
            same_fields(threshold, source, GROUP)
        if frame is None:
            needs_score[path] = "missing 255 frame"
            continue
        same_fields(frame, source, GROUP)
        if (frame.get("status") == "excluded" and frame.get("reason") not in (None, "", "baseline")
                and frame.get("anchor_path") == source["s0_image_path"]
                and {"reference255", "mean_loss_pct", "fit_source"} <= frame.keys()):
            excluded[path] = frame["reason"]
            continue
        try:
            measure = cached_measurement(run, source, frame, threshold)
        except (KeyError, TypeError, ValueError, OSError, EOFError, BadZipFile) as exc:
            needs_score[path] = str(exc)
        else:
            accepted[path] = measure
            if measure.restored_fields:
                restored[path] = measure.restored_fields
    if needs_score:
        repaired = rescore_missing(needs_score, sources, root, status, scratch)
        frames.update(repaired)
        for path, frame in repaired.items():
            if frame["status"] == "excluded":
                excluded[path] = frame["reason"]
            else:
                accepted[path] = cached_measurement(scratch, sources[path], frame, None)
    return accepted, excluded, needs_score, restored, frames


def publish_repairs(run, scratch, frames, measures, status):
    """Refresh the generated measurement tables from verified maps without re-fitting intact frames."""
    from .fullscale255 import FRAME_COLUMNS, analyze_curves, count_curve, csv_table

    thresholds = np.arange(101)
    records = []
    for path, frame in frames.items():
        record = {col: frame.get(col, "") for col in FRAME_COLUMNS}
        if path in measures:
            measure = measures[path]
            with np.load(measure.map_file, allow_pickle=False) as archive:
                n, counts = count_curve(archive["loss_pct"], thresholds)
            record.update(valid_crown_px=n, counts=counts, mean_loss_pct=measure.mean)
        records.append(record)
    csv_table(scratch / "frames.csv", ({k:r[k] for k in FRAME_COLUMNS} for r in records), FRAME_COLUMNS)
    selection = analyze_curves(scratch, records, thresholds)
    dump(scratch / "status.json", dict(status, selection=selection, scanned_thresholds_pct=thresholds.tolist(),
         frame_counts=dict(Counter(r["state_id"] + "/" + r["status"] for r in records)),
         exclusions=dict(Counter(r["reason"] for r in records if r["status"] != "ok"))))
    (run / "maps").mkdir(exist_ok=True)
    for repaired_map in (scratch / "maps").glob("*"):
        repaired_map.replace(run / "maps" / repaired_map.name)
    for name in ("frames.csv", "frame-thresholds.csv", "tooth-curves.csv", "stage-curves.csv",
                 "threshold-response.png", "status.json"):
        item = scratch / name
        if item.is_file():
            item.replace(run / name)
        elif name in ("tooth-curves.csv", "stage-curves.csv", "threshold-response.png"):
            (run / name).unlink(missing_ok=True)


def single_features(source, tables):
    path = source["image_path"]
    for table_name, table in tables:
        row = table.get(path)
        if row is None:
            continue
        same_fields(row, source, (*GROUP, "rgb_path"))
        if row["sample_id"] != Path(path).stem or row["s0_reference_path"] != source["s0_image_path"]:
            raise ValueError(f"Stage 3 sample ID or reference mismatch: {path}")
        if row.get("single_image_status", "ok") == "ok" and all(finite(row.get(k)) for k in SINGLE):
            return row, table_name
    return None, None


def verify_written(path, expected, measures):
    rows = list(csv_rows(path))
    by_path = indexed(rows, "qlf_path", "exported QLF")
    indexed(rows, "sample_id", "exported IDs")
    if by_path.keys() != expected.keys():
        raise ValueError("CSV membership differs from eligible images")
    for row in rows:
        if row != expected[row["qlf_path"]] or not all(finite(row[k]) for k in FEATURES):
            raise ValueError(f"Invalid exported values: {row['sample_id']}")
        if not 0 <= float(row["loss_area_gt30_pct"]) <= 100:
            raise ValueError(f"Invalid area percentage: {row['sample_id']}")
    checks = {}
    for stage in STATES:
        candidates = [r for r in rows if r["state_id"] == stage]
        if not candidates:
            continue
        # Prefer a non-anchor S0 repeat so its root fit is exercised in the acceptance example.
        row = next((r for r in candidates if measures[r["qlf_path"]].fit_source == "paired root Huber fit"), candidates[0])
        measure = measures[row["qlf_path"]]
        n, count, mean, area = map_values(measure.map_file)
        if not close(row["huber_delta_f_pct"], mean) or not close(row["loss_area_gt30_pct"], area):
            raise ValueError(f"Stored CSV and signed NPZ disagree: {row['sample_id']}")
        checks[stage] = dict(sample_id=row["sample_id"], qlf_path=row["qlf_path"], valid_crown_px=n,
                             loss_gt30_px=count, map_mean_loss_pct=mean, csv_mean_loss_pct=float(row["huber_delta_f_pct"]),
                             map_area_gt30_pct=area, csv_area_gt30_pct=float(row["loss_area_gt30_pct"]))
    return checks


def native_singles(root, sources):
    from .dataset import DATA
    from .feature_export import single_image

    if DATA != root / 'Data':
        raise ValueError('Single-image extractor loaded a different data root')
    for path, row in sources.items():
        values = single_image(row)
        yield dict(sample_id=Path(path).stem, qlf_path=path, s0_reference_path=row['s0_image_path'],
                   **{k: row[k] for k in (*GROUP, 'rgb_path')},
                   **{k: str(values[k]) for k in SINGLE}, single_image_status=values['single_image_status'])


def export(run, root, feature_file=None, singles_file=None):
    run = run.resolve()
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    root = Path(root).resolve()
    if status.get("status") != "complete":
        raise ValueError("Complete the 255 measurement stage before exporting features")
    source_file = root / SOURCE_TABLE
    if status.get("feature_table_sha256") and hashlib.sha256(source_file.read_bytes()).hexdigest() != status["feature_table_sha256"]:
        raise ValueError("Reviewed inventory differs from the one used in the 255 run")
    sources = indexed(csv_rows(source_file), "image_path", source_file)
    if len({Path(path).stem for path in sources}) != len(sources):
        raise ValueError("Reviewed inventory contains duplicate sample IDs")
    tables = [(file.name, indexed(csv_rows(file), "qlf_path", file)) for file in (feature_file, singles_file) if file]
    missing = {path: row for path, row in sources.items() if single_features(row, tables)[0] is None}
    if missing:
        native = indexed(native_singles(root, missing), "qlf_path", "native green P10 and texture entropy")
        tables.append(("native reviewed crown measurements", native))
    anchors = {}
    for path, src in sources.items():
        anchor = sources[src["s0_image_path"]]
        view = (src["tooth_id"], src["angle_id"])
        if (src["state_id"] not in STATES or anchor["state_id"] != "S0" or (anchor["tooth_id"], anchor["angle_id"]) != view
                or anchors.setdefault(view, src["s0_image_path"]) != src["s0_image_path"]):
            raise ValueError(f"Inconsistent S0 reference: {path}")

    with TemporaryDirectory(dir=run, prefix=".thr30-") as tmp:
        scratch = Path(tmp)
        measures, excluded, rescored, restored, frames = measurements(run, root, status, sources, scratch)
        expected, feature_sources = {}, Counter()
        stages_by_view = defaultdict(set)
        fit_counts = {s: Counter() for s in STATES}
        for path in sorted(measures, key=lambda p: (sources[p]["state_id"], sources[p]["tooth_id"],
                                                   sources[p]["angle_id"], int(sources[p]["repeat_rank"]), p)):
            source, measure = sources[path], measures[path]
            state = source["state_id"]
            fit_counts[state][measure.fit_source] += 1
            stages_by_view[source["tooth_id"], source["angle_id"]].add(state)
            row, feature_source = single_features(source, tables)
            if row is None:
                excluded[path] = "missing_complete_p10_or_texture_entropy"
                continue
            if not all((root / "Data" / row[k]).is_file() for k in ("qlf_path", "rgb_path")):
                excluded[path] = "missing_paired_photo"
                continue
            feature_sources[str(feature_source)] += 1
            expected[path] = {k: row[k] for k in COLUMNS if k not in ("huber_delta_f_pct", "loss_area_gt30_pct")}
            expected[path].update(huber_delta_f_pct=measure.mean, loss_area_gt30_pct=measure.area)
        if (expected.keys() | excluded.keys()) != sources.keys() or expected.keys() & excluded.keys():
            raise ValueError("Incomplete exported/excluded partition")
        output = scratch / CSV_NAME
        with output.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(expected.values())
        checks = verify_written(output, expected, measures)
        matched = {view for view, states in stages_by_view.items() if states == set(STATES)}
        matched_count = sum((sources[p]["tooth_id"], sources[p]["angle_id"]) in matched for p in measures)
        by_stage = Counter(r["state_id"] for r in expected.values())
        reasons = {s: Counter() for s in STATES}
        for p, reason in excluded.items():
            reasons[sources[p]["state_id"]][reason] += 1
        report = dict(csv=CSV_NAME, csv_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                      threshold_pct=THRESHOLD, image_path_root="workspace/Data", source_frames=len(sources),
                      feature_columns=FEATURES, source_table_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
                      qc_passed=len(measures), exported=len(expected), excluded=len(excluded),
                      exported_by_stage={s: by_stage[s] for s in STATES}, exclusion_reasons_by_stage=reasons,
                      excluded_images=excluded, matched_three_stage_views=len(matched), matched_qc_frames=matched_count,
                      fit_sources_by_stage=fit_counts, reused_255_rows=len(measures.keys() - rescored.keys() - restored.keys()),
                      rescored_images=rescored, restored_from_numeric_maps=restored,
                      single_image_feature_sources=feature_sources, spot_checks=checks)
        audit = scratch / REPORT_NAME
        audit.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if rescored or restored:
            publish_repairs(run, scratch, frames, measures, status)
        output.replace(run / CSV_NAME)
        audit.replace(run / REPORT_NAME)
    return report
