"""Transfer baseline stain labels to generation-2 photographs with registration QA."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil

from .dataset import (DATA, GEN2, OUT, SEM, composition_passes, cv2, get_anchor,
                         inventory, np, packed, read_index, read_label, registered,
                         save_mask, write_csv, write_index)

ANGLES = ("A000", "A045", "A090", "A135", "A180")


def identity_review(groups, legacy, binary):
    """Compare the two archived-label hypotheses using independently repeated angles."""
    rows = []
    for angle in ANGLES:
        candidates = {}
        targets = {}
        for tooth in ("T004", "T005"):
            records = groups[tooth,angle,"QLF"]
            a = get_anchor(records, legacy)
            candidates[tooth] = packed(a,read_label(SEM/legacy[a["path"]]["label_path"]))
            for state in ("S1","S2"):
                r = next(r for r in records if r["state_id"]==state and r["repeat_rank"]=="4")
                path = (SEM/legacy[r["path"]]["label_path"] if state=="S1" else
                        binary["QLF"]/Path(r["path"]).relative_to(GEN2).with_suffix(".png"))
                targets[tooth,state] = packed(r,read_label(path))[0]
        for (tooth,state), target in targets.items():
            for match,(anchor,points) in candidates.items():
                m,_,q = registered(anchor,target,points,False)
                rows.append(dict(target_tooth=tooth,state_id=state,angle_id=angle,
                                 s0_tooth=match,target_path=target.image_id,s0_path=anchor.image_id,
                                 accepted=m is not None,**q))
    suspect = set()
    for tooth in ("T004","T005"):
        for state in ("S1","S2"):
            strong = 0
            for angle in ANGLES:
                rows1 = [r for r in rows if (r["target_tooth"],r["state_id"],r["angle_id"]) == (tooth,state,angle)]
                nominal = next(r for r in rows1 if r["s0_tooth"]==tooth)
                alternate = next(r for r in rows1 if r["s0_tooth"]!=tooth)
                if (alternate["accepted"] and alternate["gray_ncc"] >= .60 and
                        alternate["gray_ncc"] - nominal["gray_ncc"] >= .20):
                    strong += 1
            if strong >= 3:
                suspect.add((tooth,state))
    write_csv(SEM/"identity-review.csv",rows)
    return suspect


def process_group(rows, legacy, binary, suspect, *, state="S2", label_dir=Path("标签")):
    a = get_anchor(rows, legacy)
    meta = legacy[a["path"]]
    baseline = read_label(SEM/meta["label_path"])
    source, corners = packed(a,baseline)
    dark = np.uint8(baseline==2)
    records, pending, completed = {}, [], []
    targets = [r for r in rows if r["state_id"] == state]
    held = (a["tooth_id"], state) in suspect

    def write(r, body, matrix, status, quality):
        inside = body>0
        label = np.uint8(inside)
        if matrix is None:
            label[inside] = 3
        else:
            stain = cv2.warpAffine(dark,matrix[:2],body.shape[::-1],flags=cv2.INTER_NEAREST)>0
            label[inside & stain] = 2
        path = label_dir / Path(r["path"]).with_suffix(".png")
        save_mask(SEM/path,label)
        count = np.bincount(label.ravel(),minlength=4)
        rec = dict(path=r["path"],label_path=path.as_posix(),tooth=r["tooth_id"],angle=r["angle_id"],
                   modality=r["modality"],dose=int(r["dose_hours"]),state=state,rank=int(r["repeat_rank"]),
                   s0_anchor=a["path"],source=meta["source"],status=status,quality=quality,
                   background_pixels=int(count[0]),tooth_pixels=int(count[1]),dark_pixels=int(count[2]),
                   uncertain_pixels=int(count[3]),ignored_pixels=0,
                   transform=None if matrix is None else matrix[:2].tolist(),
                   is_duplicate=False,duplicate_of="")
        records[r["path"]] = rec

    for r in (r for r in targets if not r["is_duplicate"]):
        path = (DATA / r["mask_path"] if binary is None else
                binary[r["modality"]] / Path(r["path"]).relative_to(GEN2).with_suffix(".png"))
        body = read_label(path)
        if held:
            write(r,body,None,"identity_review_required",{})
            continue
        target, points = packed(r,body)
        matrix, status, quality = registered(source,target,corners)
        if matrix is None:
            pending.append((r,body,target,quality))
            continue
        write(r,body,matrix,status,quality)
        completed.append((target,points,matrix,quality["dice"]))

    # Try independent S2 repeats as bridges. Final composition must still pass
    # geometry and texture checks against the same S0.
    bridges = sorted(completed,key=lambda item:-item[3])[:2]
    for row,body,target,first in pending:
        for bridge,points,prior,_ in bridges:
            m, _, q = registered(bridge,target,points,False)
            if m is not None:
                composed = m@prior
                ok,dice = composition_passes(source,target,composed)
                if ok:
                    write(row,body,composed,"registered_via_repeat",
                          dict(q,bridge=bridge.image_id,composed_dice=dice))
                    break
        else:
            write(row,body,None,"registration_review_required",first)

    for r in (r for r in targets if r["is_duplicate"]):
        orig = records[r["duplicate_of"]]
        label = (label_dir / Path(r["path"]).with_suffix(".png")).as_posix()
        shutil.copyfile(SEM/orig["label_path"],SEM/label)
        records[r["path"]] = dict(orig,path=r["path"],label_path=label,rank=int(r["repeat_rank"]),
                                  is_duplicate=True,duplicate_of=r["duplicate_of"])
    return list(records.values())


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--qlf-body",type=Path,required=True)
    parser.add_argument("--rgb-body",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=5)
    options = parser.parse_args()
    cv2.setNumThreads(1)
    groups = inventory()
    legacy = read_index(SEM/"legacy_index.jsonl")
    binary = {"RGB":options.rgb_body,"QLF":options.qlf_body}
    for rr in groups.values():
        for r in rr:
            if r["state_id"]=="S2" and not r["is_duplicate"]:
                path = binary[r["modality"]]/Path(r["path"]).relative_to(GEN2).with_suffix(".png")
                if not path.is_file():
                    raise FileNotFoundError(path)
    suspect = identity_review(groups,legacy,binary)
    print("Identity review:", sorted(suspect),flush=True)
    write_csv(SEM/"identity-hold.csv",[
        dict(tooth_id=tooth,state_id=stage,hold=(tooth,stage) in suspect)
        for tooth in ("T004","T005") for stage in ("S1","S2")])
    records = []
    with ThreadPoolExecutor(max_workers=options.workers) as pool:
        tasks = pool.map(lambda rr:process_group(rr,legacy,binary,suspect),groups.values())
        for i,result in enumerate(tasks,1):
            records.extend(result)
            if i%10==0:
                print(i,len(records),Counter(r["status"] for r in records),flush=True)
    assert len(records)==630 and len({r["path"] for r in records})==630
    write_index(SEM/"gen2_index.jsonl",sorted(records,key=lambda r:r["path"]))
    print(Counter((r["modality"],r["status"]) for r in records),flush=True)


if __name__ == "__main__":
    main()
