"""Signed root-calibrated 255 loss maps and descriptive threshold-response curves."""
import csv
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from .files import SOURCE_TABLE, dump, sha

FRAME_COLUMNS = ("tooth_id", "angle_id", "state_id", "repeat_rank", "qlf_path", "anchor_path",
                 "status", "reason", "map_path", "slope", "intercept", "reference255", "root_tiles",
                 "root_min", "root_max", "valid_crown_px", "mean_green", "mean_loss_pct", "fit_source")


def loss_map_255(green, valid, slope, intercept):
    import numpy as np
    green = np.asarray(green, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if green.ndim != 2 or green.shape != valid.shape:
        raise ValueError("Crown green image and mask must be matching 2-D arrays")
    if not np.isfinite([slope, intercept]).all() or slope <= 0:
        raise ValueError("Invalid tooth-root fit")
    valid = valid & np.isfinite(green) & (green >= 16) & (green < 255)
    reference = 255.0 * slope + intercept
    loss = np.full(green.shape, np.nan, dtype=np.float64)
    loss[valid] = 100.0 * (reference - green[valid]) / (255.0 * slope)
    return loss


def count_curve(loss, thresholds):
    import numpy as np
    values = np.sort(loss[np.isfinite(loss)])
    if not values.size:
        raise ValueError("Empty valid crown")
    counts = values.size - np.searchsorted(values, thresholds, side="right")
    return int(values.size), counts


def save_map(out, key, loss, bounds):
    import numpy as np
    from PIL import Image
    from matplotlib import colormaps
    name = Path(key).stem
    maps = out / "maps"
    # Names include tooth, state, view and unique QLF ID in the active inventory.
    if Path(name).name != name:
        raise ValueError("Invalid QLF sample stem")
    quantitative = maps / (name + ".npz")
    if quantitative.exists():
        raise ValueError(f"Duplicate output image ID: {name}")
    np.savez_compressed(quantitative, loss_pct=loss, reference_xyxy=np.asarray(bounds, dtype=np.int32))
    valid = np.isfinite(loss)
    display = np.where(valid, np.clip(loss, 0, 100) / 100., 0.)
    rgba = colormaps["viridis"](display, bytes=True)
    rgba[..., 3] = np.uint8(valid) * 255
    Image.fromarray(rgba).save(maps / (name + ".png"))
    return quantitative.relative_to(out).as_posix()


def score_frame(row, anchor, base, root0, bounds, config, index, held, table, *, out, thresholds):
    import numpy as np
    from .area25 import image_path, with_crown
    from .direct_features import frame_native, in_reference, measure_pair
    from .huber import calibrated_loss, fit, paired_root_tiles

    path = image_path(row)
    item = table[path]
    stage = row["state_id"]
    rec = dict(tooth_id=row["tooth_id"], angle_id=row["angle_id"], state_id=stage,
               repeat_rank=row["repeat_rank"], qlf_path=path, anchor_path=image_path(anchor),
               status="excluded", reason="", map_path="", slope="", intercept="",
               reference255="", root_tiles="", root_min="", root_max="", valid_crown_px="",
               mean_green="", mean_loss_pct="", fit_source="", counts=None)

    def exclude(reason):
        rec["reason"] = reason
        return rec

    if not item.get("crown_semantic_mask_path"):
        return exclude(item.get("crown_mask_status") or "missing crown mask")
    if stage != "S0" and item.get("loss_status") != "automatic_qc_pass":
        return exclude(item.get("loss_status") or "missing registration QC")
    meta = index[row["path"]]
    frame = frame_native(row, meta, held)
    body = in_reference(frame, meta, anchor["path"], bounds)
    if body is None:
        return exclude("no verified alignment")
    crown = with_crown(frame, meta, table, anchor, bounds)
    if crown is None or base is None:
        return exclude("no aligned crown")
    raw, status, _ = measure_pair(base, crown)
    if status != "automatic_qc_pass":
        return exclude(status)
    try:
        tiles = paired_root_tiles(root0, base, crown, body, bounds, config)
    except ValueError as exc:
        return exclude(str(exc))
    if len(tiles) == 0:
        return exclude("no valid paired root tiles")
    rec.update(root_tiles=len(tiles), root_min=float(tiles[:, 2].min()), root_max=float(tiles[:, 2].max()))
    if path == image_path(anchor):
        a, b = 1., 0.
        rec["fit_source"] = "reference exposure"
    else:
        if np.ptp(tiles[:, 2]) <= 0:
            return exclude("constant reference-root signal")
        try:
            a, b = fit(tiles[:, 2], tiles[:, 3], "huber")
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            return exclude(f"root Huber fit failed: {exc}")
        rec["fit_source"] = "paired root Huber fit"
    if not np.isfinite([a, b]).all() or a <= 0:
        return exclude("no positive finite root fit")
    # Only the inherited per-pixel eligibility screen is retained from the old loss.
    # The old per-pixel S0 prediction is never used as the response denominator.
    _, eligible, _ = calibrated_loss(raw, base.green, crown.green, a, b)
    loss = loss_map_255(crown.green, eligible, a, b)
    if not np.isfinite(loss).any():
        return exclude("no eligible crown pixels")
    n, counts = count_curve(loss, thresholds)
    rec.update(status="ok", slope=float(a), intercept=float(b), reference255=float(255*a+b),
               mean_green=float(np.mean(crown.green, where=np.isfinite(loss), dtype=np.float64)),
               valid_crown_px=n, mean_loss_pct=float(np.nanmean(loss)), counts=counts,
               map_path=save_map(out, path, loss, bounds))
    return rec


def csv_table(path, rows, columns):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def analyze_curves(out, records, thresholds):
    from collections import defaultdict
    import numpy as np
    import matplotlib.pyplot as plt

    states = ("S0", "S1", "S2")
    by_view = defaultdict(list)
    # Every denominator belongs to one frame. Repeats and angles are then averaged.
    frame_cols = ["tooth_id", "angle_id", "state_id", "repeat_rank", "qlf_path",
                  "threshold_pct", "response_px", "valid_crown_px", "response_pct"]
    with (out / "frame-thresholds.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, frame_cols)
        writer.writeheader()
        for r in records:
            if r["status"] != "ok":
                continue
            n = r["valid_crown_px"]
            percents = 100.*r["counts"]/n
            tooth, angle, state = (r[k] for k in ("tooth_id", "angle_id", "state_id"))
            by_view[tooth, angle, state].append(percents)
            ident = {k:r[k] for k in frame_cols[:5]}
            for t, count, pct in zip(thresholds, r["counts"], percents, strict=True):
                writer.writerow(dict(**ident, threshold_pct=int(t), response_px=int(count),
                                     valid_crown_px=int(n), response_pct=float(pct)))
    matched = sorted({(t, a) for t, a, _ in by_view
                      if all((t, a, s) in by_view for s in states)})
    teeth = sorted({t for t, _ in matched})
    angles_by_tooth = {t: [a for tooth, a in matched if tooth == t] for t in teeth}
    selection = dict(matched_views=len(matched), teeth=teeth, threshold_pct=None,
                     matched_angles_by_tooth=angles_by_tooth,
                     rule="Maximize the smaller median within-tooth increase S1-S0 or S2-S1; "
                          "require both positive; ties use the lower threshold.",
                     interpretation="Exploratory selection. The published classifier uses a fixed 30% feature.")
    if not matched:
        selection["reason"] = "No matched tooth/view with measurements at all three stages."
        return selection

    tooth_curves = []
    for tooth in teeth:
        angles = angles_by_tooth[tooth]
        tooth_curves.append([np.mean([np.mean(by_view[tooth, a, s], axis=0) for a in angles], axis=0)
                             for s in states])
    curves = np.asarray(tooth_curves)
    rows = (dict(tooth_id=t, state_id=s, threshold_pct=int(th), response_pct=float(p))
            for ti, t in enumerate(teeth) for si, s in enumerate(states)
            for th, p in zip(thresholds, curves[ti, si], strict=True))
    csv_table(out/"tooth-curves.csv", rows, ["tooth_id","state_id","threshold_pct","response_pct"])

    def score(data):
        gap01 = np.median(data[:, 1] - data[:, 0], axis=0)
        gap12 = np.median(data[:, 2] - data[:, 1], axis=0)
        minima = np.minimum(gap01, gap12)
        idx = int(np.argmax(minima))
        return gap01, gap12, idx if minima[idx] > 0 else None

    early, later, best = score(curves)
    quantiles = np.percentile(curves, (25, 50, 75), axis=0)
    if len(teeth) < 2:
        best = None
        selection["reason"] = "Only one paired tooth; curves are descriptive."
    if best is not None:
        selection.update(threshold_pct=int(thresholds[best]), median_s1_minus_s0_pp=float(early[best]),
                         median_s2_minus_s1_pp=float(later[best]))
    elif "reason" not in selection:
        selection["reason"] = "The curves do not support two positive adjacent median increases."
    if len(teeth) > 1:
        choices = {}
        for i, tooth in enumerate(teeth):
            _, _, j = score(np.delete(curves, i, axis=0))
            choices[tooth] = int(thresholds[j]) if j is not None else None
        selection["threshold_when_each_tooth_is_omitted"] = choices

    rows = (dict(state_id=s, threshold_pct=int(t), q25=float(quantiles[0,j,k]),
                 median=float(quantiles[1,j,k]), q75=float(quantiles[2,j,k]),
                 paired_s1_minus_s0_pp=float(early[k]), paired_s2_minus_s1_pp=float(later[k]))
            for j, s in enumerate(states) for k, t in enumerate(thresholds))
    csv_table(out/"stage-curves.csv", rows, ["state_id","threshold_pct","q25","median","q75",
                                            "paired_s1_minus_s0_pp","paired_s2_minus_s1_pp"])
    fig, ax = plt.subplots(figsize=(8.4, 5.2), layout="constrained")
    for i, (stage, color) in enumerate(zip(states, ("#315783", "#C25C19", "#2D7C68"), strict=True)):
        lo, mid, hi = quantiles[:, i, :]
        ax.plot(thresholds, mid, lw=2.3, color=color, label=stage)
        ax.fill_between(thresholds, lo, hi, color=color, alpha=.12)
    if best is not None:
        threshold = int(thresholds[best])
        ax.axvline(threshold, color="#30353A", ls="--", lw=1, label=f"Candidate: {threshold}%")
    ax.set(xlim=(0,100), ylim=(0,100), xlabel="Pixel loss threshold (%)",
           ylabel="Valid crown area above threshold (%)",
           title=f"255 reference | {len(teeth)} teeth, {len(matched)} matched views")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.22)
    ax.legend()
    fig.savefig(out/"threshold-response.png", dpi=180)
    plt.close(fig)
    return selection


def reviewed_registration_group(bundle_path, table, scratch, audit_dir=None):
    from .dataset import DATA
    from .reviewed import group_reader, load_registrations, verify_photos

    with ZipFile(bundle_path) as bundle:
        runtime = json.loads(bundle.read("runtime.json"))
        records = load_registrations(bundle, table, scratch, runtime["published_masks_with_pixel_differences"])
        audit = verify_photos(bundle, table, DATA, audit_dir)
    audit.update(registration_source="reviewed workspace indices and saved T004/T005 transforms",
                 author_transforms_sha256=sha(bundle_path), reviewed_registrations=len(records))
    return group_reader(records), audit


def measure_view(view, groups, legacy, index, held, configs, table, scratch, out, thresholds,
                 registration_group=None, wanted=None):
    """Prepare one S0 grid and measure either the complete view or selected missing frames."""
    import numpy as np
    from .area25 import image_path, with_crown
    from .build_masks import process_group
    from .crown import roi_labels
    from .dataset import get_anchor
    from .direct_features import S0_SOURCE_IDS, confirmed_rows, frame_native, reference_crop

    tooth, angle = view
    rows = confirmed_rows(groups, tooth, angle)
    anchor = get_anchor(rows, legacy)
    if tooth in S0_SOURCE_IDS:
        register = registration_group or process_group
        updates = [item for state in ("S1", "S2")
                   for item in register(rows, legacy, None, set(), state=state, label_dir=scratch)]
        index = index | {item["path"]: item for item in updates}
        held = held - {(tooth, "S1"), (tooth, "S2")}
    config = configs[anchor.get("source_tooth_id", tooth), angle]
    anchor_meta = index[anchor["path"]]
    source = frame_native(anchor, anchor_meta, held)
    bounds = reference_crop(source.labels)
    base = with_crown(source, anchor_meta, table, anchor, bounds)
    x, y, x1, y1 = bounds
    start, end = np.array(config["crown_boundary"])
    root_side = dict(crown_boundary=[start, end], crown_side_point=2*start - np.array(config["crown_side_point"]))
    root = roi_labels(source.labels, root_side, np.eye(3))[y:y1, x:x1] == 1
    return [score_frame(row, anchor, base, root, bounds, config, index, held, table,
                        out=out, thresholds=thresholds)
            for row in rows if wanted is None or image_path(row) in wanted]


def measure(workspace, out, kind):
    """Replay the reviewed per-view alignment and measure all registered frames."""
    import numpy as np
    from .crown import geometry
    from .dataset import DATA, cv2, inventory, read_csv
    from .direct_features import indexes

    if DATA != workspace / "Data":
        raise ValueError("The selected data root differs from the loaded measurement workspace")
    cv2.setNumThreads(1)
    out.mkdir(parents=True, exist_ok=False)
    (out / "maps").mkdir()
    thresholds = np.arange(101, dtype=int)
    meta = dict(scope=kind, feature_table_sha256=sha(workspace / SOURCE_TABLE),
                scanned_thresholds_pct=thresholds.tolist(), runner_sha256=sha(Path(__file__)),
                formula="100 * (255*a + b - G) / (255*a)", denominator="all valid crown pixels",
                map_coordinates="crop in the S0 reference grid; reference_xyxy in NPZ",
                registration_source="registered input indices")
    dump(out / "status.json", dict(meta, status="running"))
    try:
        features = read_csv(workspace / SOURCE_TABLE)
        table = {r["image_path"]: r for r in features}
        if len(features) != len(table) or not table:
            raise ValueError("Empty or duplicate reviewed QLF inventory")
        groups, configs = inventory(), geometry()
        legacy, index, held = indexes()
        views = sorted((tooth, angle) for tooth, angle, modality in groups if modality == "QLF")
        records = []
        with TemporaryDirectory(prefix="qlf-registration-") as temp:
            scratch = Path(temp)
            reader = None
            if kind == "study":
                reader, audit = reviewed_registration_group(workspace / "stage3-author-transforms.zip",
                                                           table, scratch, audit_dir=out)
                meta.update(audit)
            for i, view in enumerate(views, 1):
                rows = measure_view(view, groups, legacy, index, held, configs, table, scratch, out, thresholds,
                                    registration_group=reader)
                records.extend(rows)
                print(f"{i}/{len(views)} {view}: {sum(r['status']=='ok' for r in rows)}/{len(rows)}", flush=True)
        found = [r["qlf_path"] for r in records]
        if len(set(found)) != len(found) or set(found) != set(table):
            raise ValueError("Output frames differ from the reviewed inventory")
        csv_table(out / "frames.csv", ({k:r[k] for k in FRAME_COLUMNS} for r in records), FRAME_COLUMNS)
        meta["selection"] = analyze_curves(out, records, thresholds)
        meta["frame_counts"] = dict(Counter(r["state_id"] + "/" + r["status"] for r in records))
        meta["exclusions"] = dict(Counter(r["reason"] for r in records if r["status"] != "ok"))
        dump(out / "status.json", dict(meta, status="complete"))
    except Exception as exc:
        dump(out / "status.json", dict(meta, status="failed", error=str(exc)))
        raise
