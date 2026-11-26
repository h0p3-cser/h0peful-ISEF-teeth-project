"""Extract unstained crown QLF features and direct between-stage loss."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image

from .build_masks import process_group
from .dataset import (DATA, OUT, PROJECT, SEM, STATES, cv2, get_anchor, inventory, np,
                         read_csv, read_index, read_label, read_rgb, write_csv)
from .crown import geometry, native_crown, publish_crown
from .pixels import INTENSITY, LOSSES, direct_loss, intensity
from .settings import LOCAL_VIEWS

TAU = 5.0
ERODE = np.ones((3,3),np.uint8)
FEATURE_DIR = PROJECT / "QLF_GPTadvice"
FEATURE_FILE = "QLF_GPTadvice_S0_S1_S2_C04.csv"
# S0 folders were swapped after the user confirmed the identity correction.
S0_SOURCE_IDS = {} if LOCAL_VIEWS.is_file() else {"T004": "T005", "T005": "T004"}
CORRECTED_LABELS = Path("编号复核后") / "标签"
CORRECTED_CROWNS = Path("编号复核后") / "牙冠标签"


@dataclass
class Frame:
    row: dict
    photo: np.ndarray
    green: np.ndarray
    labels: np.ndarray
    exif: tuple
    status: str


def frame_native(row, label_meta, held):
    image = read_rgb(row)
    labels = read_label(SEM/label_meta["label_path"])
    if image.shape[:2] != labels.shape:
        raise ValueError(row["path"])
    status = label_meta["status"]
    if (row["tooth_id"],row["state_id"]) in held:
        labels[labels>0] = 3
        status = "identity_review_required"
    with Image.open(DATA / row.get("physical_path", row["path"])) as im:
        exif = im.getexif().get_ifd(34665)
        settings = tuple(float(exif[k]) if k in exif else None for k in (33434,34855,33437))
    return Frame(row,image,image[...,1],labels,settings,status)


def reference_crop(labels):
    x,y,w,h = cv2.boundingRect(np.uint8(labels>0))
    x0,y0 = max(0,x-25),max(0,y-25)
    return x0,y0,min(labels.shape[1],x+w+25),min(labels.shape[0],y+h+25)


def in_reference(frame, meta, anchor, bounds, display=False):
    path = frame.row["path"]
    x,y,x1,y1 = bounds
    sl = np.s_[y:y1,x:x1]
    if path == anchor:
        return Frame(frame.row,frame.photo[sl] if display else None,frame.green[sl],
                     frame.labels[sl],frame.exif,frame.status)
    if (frame.status in {"registration_review_required","identity_review_required"} or
            meta["s0_anchor"] != anchor or meta["transform"] is None):
        return None
    # matrix locates target pixels; interpolation changes only the coordinate grid.
    origin_shift = np.array([[1.,0,x],[0,1.,y],[0,0,1.]])
    matrix = np.asarray(meta["transform"],dtype=float) @ origin_shift
    size = x1-x,y1-y
    def warp(arr, flags):
        return cv2.warpAffine(arr,matrix,size,flags=flags|cv2.WARP_INVERSE_MAP)
    green = warp(frame.green.astype(np.float32),cv2.INTER_LINEAR)
    saturated = warp(np.float32(frame.green==255),cv2.INTER_LINEAR)
    green[saturated>0] = np.nan
    labels = warp(frame.labels,cv2.INTER_NEAREST)
    photo = warp(frame.photo,cv2.INTER_LINEAR) if display else None
    return Frame(frame.row,photo,green,labels,frame.exif,frame.status)


def clean_common(frames):
    return cv2.erode(np.uint8(np.logical_and.reduce([v.labels==1 for v in frames])),ERODE)>0


def measure_pair(base, current, clean=None):
    empty = dict(common_roi_pixels=0,common_body_fraction=None,valid_loss_roi_pixels=0,
                 valid_loss_roi_fraction=None,**dict.fromkeys(LOSSES))
    if base is None or current is None:
        return None, "no_verified_alignment", empty
    shared_body = (base.labels>0) & (current.labels>0)
    target_px = np.count_nonzero(current.labels>0)
    overlap = float(shared_body.sum()/target_px) if target_px else 0.
    roi = clean_common((base,current)) if clean is None else clean
    area = int(roi.sum())
    result = dict(empty,common_roi_pixels=area,common_body_fraction=overlap)
    if overlap < .85 or area<500:
        return None, "insufficient_common_roi", result
    loss, stats, n = direct_loss(base.green,current.green,roi,TAU)
    result.update(valid_loss_roi_pixels=n,valid_loss_roi_fraction=n/area)
    if n/area<.90:
        return None, "insufficient_signal_roi", result
    return loss,"automatic_qc_pass",dict(result,**stats)


def exif_status(a,b):
    if any(v is None for v in (*a.exif,*b.exif)):
        return "missing"
    return "same" if a.exif==b.exif else "different"


def metadata(row):
    return dict(**{k:row[k] for k in ("tooth_id","state_id","class_id","dose_hours","batch_id",
                                      "angle_id","repeat_rank","capture_datetime")},
                image_path=row.get("physical_path", row["path"]),body_mask_path=row["mask_path"],
                is_duplicate=int(row["is_duplicate"]),duplicate_of=row["duplicate_of"])


def existing_crown(frame, record):
    status = record["crown_mask_status"]
    if not record["crown_semantic_mask_path"]:
        return None, status
    relative_path = Path("牙冠标签") / Path(frame.row["path"]).with_suffix(".png")
    labels = read_label(SEM / relative_path)
    if (labels.shape != frame.labels.shape or not np.isin(labels, (0, 1, 2)).all()
            or np.any((labels > 0) & (labels != frame.labels))
            or np.count_nonzero(labels == 1) != int(record["roi_pixels"])):
        raise ValueError(f"Crown mask differs from the reviewed ROI: {relative_path}")
    return replace(frame, labels=labels), status


def group_records(rows, legacy, index, held, rgb_paths, crown_config, reviewed):
    tooth = rows[0]["tooth_id"]
    corrected = ({r["path"] for r in rows if r["state_id"] != "S0"}
                 if tooth in S0_SOURCE_IDS else set())
    if corrected:
        index = index | {
            record["path"]: record for state in ("S1", "S2")
            for record in process_group(rows, legacy, None, set(), state=state, label_dir=CORRECTED_LABELS)
        }
        held = held - {(tooth, state) for state in ("S1", "S2")}
    s0 = get_anchor(rows,legacy)["path"]
    anchor_row = next(r for r in rows if r["path"] == s0)
    anchor_path = anchor_row.get("physical_path", s0)
    anch_native = frame_native(anchor_row,index[s0],held)
    bounds = reference_crop(anch_native.labels)

    def get_crown(frame, item):
        if reviewed is not None and frame.row["path"] not in corrected:
            return existing_crown(frame, reviewed[frame.row["path"]])
        return native_crown(frame, anch_native, item, crown_config)

    anchor_crown,_ = get_crown(anch_native,index[s0])
    if anchor_crown is None:
        raise ValueError(f"Baseline crown has no usable pixels: {s0}")
    anchor = in_reference(anchor_crown,index[s0],s0,bounds)
    qlf, adjacent, aligned = [],[],{}
    for r in rows:
        path = r["path"]
        item = index[path]
        native = anch_native if path==s0 else frame_native(r,item,held)
        f,crown_status = get_crown(native,item)
        crown_path = ""
        if f is not None:
            if path in corrected:
                relative_path = Path(publish_crown(f, CORRECTED_CROWNS))
            elif reviewed is not None:
                relative_path = Path("牙冠标签") / Path(path).with_suffix(".png")
            else:
                relative_path = Path(publish_crown(f))
            crown_path = (SEM.relative_to(DATA) / relative_path).as_posix()
        tooth = native.labels>0
        bbox = reference_crop(native.labels)
        x,y,x1,y1 = bbox
        sl = np.s_[y:y1,x:x1]
        ex = native.exif
        row = dict(**metadata(r),rgb_path=rgb_paths[r.get("source_tooth_id", r["tooth_id"]),
                                                  r["angle_id"],r["state_id"],r["repeat_rank"]],
                   semantic_mask_path=(SEM.relative_to(DATA)/item["label_path"]).as_posix(),
                   crown_semantic_mask_path=crown_path,
                   body_pixels=int(tooth.sum()),crown_pixels=int((f.labels>0).sum()) if f is not None else None,
                   roi_pixels=int((f.labels==1).sum()) if f is not None else None,
                   stain_pixels=int((f.labels==2).sum()) if f is not None else None,
                   uncertain_tooth_pixels=int((native.labels==3).sum()),
                   saturated_crown_pixels=int(((f.green==255)&(f.labels==1)).sum()) if f is not None else None,
                   exif_exposure_s=ex[0],exif_iso=ex[1],exif_aperture=ex[2],
                   mask_status=native.status,crown_mask_status=crown_status,
                   s0_image_path=anchor_path,exif_settings_status=exif_status(anch_native,native),
                   tau_pct=TAU,denominator_min_g=16,
                   source_tooth_id=r.get("source_tooth_id", r["tooth_id"]),
                   **{k:item["quality"].get(k) for k in ("dice","reverse_dice","gray_ncc",
                                                        "reverse_gray_ncc","cycle_fraction")})
        row.update(intensity(f.green[sl],f.labels[sl]==1) if f is not None else dict.fromkeys(INTENSITY))
        if r["state_id"]=="S0":
            row.update(loss_status="baseline",common_roi_pixels=None,common_body_fraction=None,
                       valid_loss_roi_pixels=None,valid_loss_roi_fraction=None,**dict.fromkeys(LOSSES))
        else:
            current = in_reference(f,item,s0,bounds) if f is not None else None
            aligned[r["state_id"],r["repeat_rank"]] = current
            _,status,stats = measure_pair(anchor,current)
            row.update(loss_status=crown_status if current is None else status,**stats)
        qlf.append(row)

    by_key={(r["state_id"],r["repeat_rank"]):r for r in rows}
    for rank in map(str,range(1,8)):
        a,b=(by_key[s,rank] for s in ("S1","S2"))
        fa,fb=(aligned.get((s,rank)) for s in ("S1","S2"))
        _,status,stats = measure_pair(fa,fb)
        adjacent.append(dict(tooth_id=a["tooth_id"],angle_id=a["angle_id"],repeat_rank=rank,
                             from_dose_hours=a["dose_hours"],to_dose_hours=b["dose_hours"],
                             from_image_path=a["path"],to_image_path=b["path"],s0_registration_anchor=anchor_path,
                             from_source_tooth_id=a.get("source_tooth_id", a["tooth_id"]),
                             to_source_tooth_id=b.get("source_tooth_id", b["tooth_id"]),
                             is_duplicate=int(b["is_duplicate"]),duplicate_of=b["duplicate_of"],
                             exif_settings_status=exif_status(fa,fb) if fa is not None and fb is not None else "",
                             tau_pct=TAU,loss_status=status,**stats))
    return qlf,adjacent


def indexes():
    if LOCAL_VIEWS.is_file():
        index = read_index(SEM / "local-index.jsonl")
        return index, index, set()
    legacy=read_index(SEM/"legacy_index.jsonl")
    active=read_index(SEM/"gen2_index.jsonl")
    review = SEM / "identity-hold.csv"
    if not review.is_file():
        review = OUT / "identity-hold.csv"
    held={(r["tooth_id"],r["state_id"]) for r in read_csv(review) if r["hold"]=="True"}
    return legacy,legacy|active,held


def confirmed_rows(groups, tooth, angle):
    rows = groups[tooth, angle, "QLF"]
    if tooth not in S0_SOURCE_IDS:
        return rows
    source_id = S0_SOURCE_IDS[tooth]
    baselines = []
    for row in groups[source_id, angle, "QLF"]:
        if row["state_id"] == "S0":
            path = Path(row["path"])
            physical_path = (Path(path.parts[0]) / tooth / Path(*path.parts[2:])).as_posix()
            if not (DATA / physical_path).is_file():
                raise FileNotFoundError(DATA / physical_path)
            baselines.append(dict(row, tooth_id=tooth, source_tooth_id=source_id, physical_path=physical_path))
    return baselines + [r for r in rows if r["state_id"] != "S0"]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-existing-crowns", action="store_true",
                        help="Use established crown labels and mask status from the current feature table.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/direct-features"))
    args = parser.parse_args()
    cv2.setNumThreads(1)
    groups = inventory()
    legacy,index,held = indexes()
    configs = geometry()
    reviewed = None
    if args.reuse_existing_crowns:
        reviewed = {}
        for r in read_csv(FEATURE_DIR / FEATURE_FILE):
            # Mask folders still use the original filename IDs.
            path = Path(r["image_path"])
            if r["state_id"] == "S0" and r.get("source_tooth_id"):
                path = Path(path.parts[0]) / r["source_tooth_id"] / Path(*path.parts[2:])
            reviewed[path.as_posix()] = r
    rgb_paths = {(r["tooth_id"],r["angle_id"],r["state_id"],r["repeat_rank"]):r["file_path"]
                 for k,rows in groups.items() if k[2]=="RGB" for r in rows}
    qlf, adjacent = [],[]
    sequence = [confirmed_rows(groups, tooth, angle) for tooth, angle, modality in sorted(groups)
                if modality == "QLF"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for fr, ps in pool.map(lambda rr:group_records(rr,legacy,index,held,rgb_paths,
                                      configs[rr[0].get("source_tooth_id", rr[0]["tooth_id"]),
                                              rr[0]["angle_id"]],reviewed),sequence):
            qlf.extend(fr)
            adjacent.extend(ps)
    qlf.sort(key=lambda r:(r["state_id"],r["tooth_id"],r["angle_id"],int(r["repeat_rank"])))
    if len(qlf) != 21 * len(sequence) or len(adjacent) != 7 * len(sequence):
        raise ValueError("Incomplete per-view direct feature export")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / FEATURE_FILE, qlf)
    write_csv(args.output_dir / "QLF_GPTadvice_S1_S2_C04_直接损失.csv",adjacent)
    print(Counter((r["state_id"],r["loss_status"]) for r in qlf),flush=True)
    print(Counter(r["loss_status"] for r in adjacent),flush=True)


if __name__=="__main__":
    main()
