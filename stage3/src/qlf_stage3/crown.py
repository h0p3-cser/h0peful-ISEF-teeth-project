"""Reviewed S0 crown split and native-coordinate crown/stain labels."""
from dataclasses import replace
import json
from pathlib import Path

from .dataset import (OUT, SEM, cv2, homogeneous, np, packed,
                         registered, save_mask)
from .settings import LOCAL_VIEWS, PROJECT

MARGIN_PX = 12


def geometry():
    path = (PROJECT if LOCAL_VIEWS.is_file() else OUT) / "crown_rois.json"
    return {tuple(k.split("_")):v for k,v in json.loads(path.read_text(encoding="utf-8")).items()}


def roi_labels(labels, config, matrix):
    # Both endpoints and the crown side marker follow the same geometry.
    xy = np.array([*config["crown_boundary"],config["crown_side_point"]],dtype=float)
    transform = homogeneous(matrix)
    xy = (np.c_[xy,np.ones(3)]@transform.T)[:,:2]
    a,b,c=xy
    direction = b-a
    if np.linalg.norm(direction)<1 or abs(np.linalg.det(np.stack([direction,c-a])))<1e-3:
        raise ValueError("Degenerate crown split")
    xx,yy = np.meshgrid(np.arange(labels.shape[1]),np.arange(labels.shape[0]),sparse=True)
    cross = (direction[0]*(yy-a[1])-direction[1]*(xx-a[0])) / np.linalg.norm(direction)
    sign = np.sign(direction[0]*(c[1]-a[1])-direction[1]*(c[0]-a[0]))
    inside = cross*sign>=MARGIN_PX
    if "outline_guard" in config:
        original = np.asarray(config["outline_guard"])
        aligned = (np.c_[original,np.ones(len(original))]@transform.T)[:,:2]
        guard = cv2.fillPoly(np.zeros(labels.shape,np.uint8),[np.rint(aligned).astype(np.int32)],1)
        inside &= guard>0
    return np.where(inside, labels, 0).astype(np.uint8)


def native_crown(frame, anchor, meta, config):
    status = frame.status
    if frame.row["path"]==anchor.row["path"]:
        matrix=np.eye(3)
    elif status=="automatic_direct_s0":
        source,corners=packed(anchor.row,anchor.labels)
        target,_=packed(frame.row,frame.labels)
        matrix, recovered, _=registered(source,target,corners)
        if matrix is None:
            return None, "crown_registration_review_required"
        status="crown_"+recovered
    else:
        if meta["transform"] is None or meta["s0_anchor"]!=anchor.row["path"] or status=="identity_review_required":
            return None, status
        matrix=meta["transform"]
    lab = roi_labels(frame.labels, config, matrix)
    if np.count_nonzero(lab==1)<500:
        return None, "insufficient_clean_crown"
    return replace(frame,labels=lab), status


def publish_crown(frame, label_dir=Path("牙冠标签")):
    path = label_dir / Path(frame.row["path"]).with_suffix(".png")
    save_mask(SEM/path,frame.labels)
    return path.as_posix()
