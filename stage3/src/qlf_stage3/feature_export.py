"""Export one row per active QLF JPEG for terminal-three modeling."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
from .area25 import process_view
from .dataset import DATA, cv2, inventory, np, read_csv, read_label, read_rgb, write_csv
from .crown import geometry
from .direct_features import FEATURE_DIR, FEATURE_FILE, indexes
from .pixels import INTENSITY, intensity
from .texture import texture

OUT = Path("outputs/training-features.csv")
LABEL = {"S0": 0, "S1": 1, "S2": 2}
TEXTURE = ("G_texture_contrast", "G_texture_homogeneity", "G_texture_energy", "G_texture_entropy")


def identity(row, rgb_index):
    rgb = rgb_index[row["rgb_path"]]
    photo = Path(row["image_path"])
    raw_rgb = Path(rgb["path"])
    if row["state_id"] == "S0" and row["source_tooth_id"] != row["tooth_id"]:
        raw_rgb = Path(raw_rgb.parts[0]) / row["tooth_id"] / Path(*raw_rgb.parts[2:])
    for path in (photo, raw_rgb, row["rgb_path"]):
        if not (DATA / path).is_file():
            raise FileNotFoundError(DATA / path)
    return dict(sample_id=photo.stem, tooth_id=row["tooth_id"], source_tooth_id=row["source_tooth_id"],
                state_id=row["state_id"], label=LABEL[row["state_id"]], dose_hours=row["dose_hours"],
                batch_id=row["batch_id"], angle_id=row["angle_id"], repeat_rank=row["repeat_rank"],
                qlf_path=photo.as_posix(), rgb_path=row["rgb_path"], rgb_raw_path=raw_rgb.as_posix(),
                s0_reference_path=row["s0_image_path"], crown_mask_path=row["crown_semantic_mask_path"])


def single_image(row):
    result = dict.fromkeys((*INTENSITY, *TEXTURE), "")
    result.update(single_image_status=row["crown_mask_status"], crown_feature_pixels="", texture_neighbor_pairs="")
    if not row["crown_semantic_mask_path"]:
        return result
    green = read_rgb({"path": row["image_path"]})[..., 1]
    roi = read_label(DATA / row["crown_semantic_mask_path"]) == 1
    if green.shape != roi.shape or int(roi.sum()) != int(row["roi_pixels"]):
        raise ValueError(f"Invalid native crown mask: {row['image_path']}")
    values = intensity(green, roi)
    for key, value in values.items():
        if not np.isclose(value, float(row[key]), rtol=1e-10, atol=1e-10):
            raise ValueError(f"Native {key} differs from the existing feature table: {row['image_path']}")
    neighbors = int((roi[:, :-1] & roi[:, 1:]).sum() + (roi[:-1, :] & roi[1:, :]).sum())
    result.update(values, crown_feature_pixels=int(roi.sum()), texture_neighbor_pairs=neighbors)
    if neighbors:
        result.update(texture(green, roi), single_image_status="ok")
    else:
        result.update(single_image_status="no_interior_neighbors")
    return result


def longitudinal(row):
    return dict(huber_delta_f_pct=row["huber_delta_f_pct"], loss_area_gt25_px=row["area_px"],
                loss_area_gt25_pct=row["area_pct"], huber_status=row["status"], huber_reason=row["reason"],
                loss_threshold_pct=row["threshold_pct"], huber_valid_crown_px=row["valid_crown_px"],
                huber_root_tiles=row["root_tiles"], huber_slope=row["slope"], huber_intercept=row["intercept"],
                huber_extrapolated_crown_pct=row["extrapolated_pct"])


def extract(workers):
    cv2.setNumThreads(1)
    features = read_csv(FEATURE_DIR / FEATURE_FILE)
    table = {r["image_path"]: r for r in features}
    if len(features) != len(table):
        raise ValueError("The source feature table repeats an image path")
    groups = inventory()
    rgb_index = {r["file_path"]: r for key, group in groups.items() if key[2] == "RGB" for r in group}
    identities = [identity(r, rgb_index) for r in features]
    keys = [(t, a) for t, a, modality in sorted(groups) if modality == "QLF"]
    legacy, index, held = indexes()
    configs = geometry()
    with TemporaryDirectory(prefix="qlf-training-root-") as tmp, ThreadPoolExecutor(max_workers=workers) as pool:
        pairs = pool.map(lambda key: process_view(key, groups, legacy, index, held, configs, table,
                                                  Path(tmp), dest=None), keys)
        longitudinal_by_path = {r["image_path"]: longitudinal(r) for view in pairs for r in view}
        print(f"Huber: {len(longitudinal_by_path)} frames", flush=True)
        values = pool.map(single_image, features)
        singles = {r["image_path"]: value for r, value in zip(features, values, strict=True)}
        print(f"Intensity and texture: {len(singles)} frames", flush=True)
    return identities, singles, longitudinal_by_path


def main():
    from .assemble import assemble

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    rows = assemble(*extract(args.workers))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, rows)
    print(Counter((r["state_id"], r["single_image_status"], r["huber_status"]) for r in rows), flush=True)


if __name__ == "__main__":
    main()
