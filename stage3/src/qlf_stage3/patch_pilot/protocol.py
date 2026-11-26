"""The prespecified observation, block-level split and v1.1 Huber solver."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..files import PACKAGE, dump, load, sha
from ..rgb.color import CHANNELS, LINEAR, fit_channels

SPEC_HASH="76e0a77fb778a0c11b674d0601929322c2e8034711f2bc5e1056d343f4722918"
PORTABLE_SPEC_HASH="8d06340d4b228a3686215ee3a3a718513a2ff6a2ea1c559088ce9788a316c43c"
SPEC_PATH=PACKAGE/"config/rgb-patch-huber-protocol-v1.md"
SCENES=PACKAGE/"config/rgb_patch_huber_regions_v1.json"
SUCCESS=("ok_evaluable","ok_no_holdout")
STATES=("S0","S1","S2")


def write_csv(path,records):
    df=pd.DataFrame(records)
    df.to_csv(path,index=False,encoding="utf-8-sig",lineterminator="\n",float_format="%.17g")
    return df


def json_lines(path,rows):
    with path.open("w",encoding="utf-8",newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n")


def read_lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def mask_pixels(im,polygons,exclusions=()):
    """Union all spatial windows on ONE physical surface before taking its median."""
    h,w=im.shape[:2]
    if not polygons:
        return np.empty(0,np.int32)
    def checked(poly):
        a=np.asarray(poly)
        if (a.ndim!=2 or a.shape[1]!=2 or len(a)<3 or not np.isfinite(a).all()
                or (a<0).any() or (a>=[w,h]).any() or not np.equal(a,np.rint(a)).all()):
            raise ValueError("Invalid polygon on native oriented photograph")
        return a.astype(np.int32)
    geometries=[checked(p) for p in polygons]
    x,y,bw,bh=cv2.boundingRect(np.vstack(geometries))
    local=np.zeros((bh,bw),np.uint8)
    offset=np.array([x,y],np.int32)
    for poly in geometries:
        cv2.fillPoly(local,[poly-offset],1)
    for poly in exclusions:
        cv2.fillPoly(local,[checked(poly)-offset],0)
    yy,xx=np.nonzero(local)
    yy,xx=yy+y,xx+x
    samples=im[yy,xx]
    valid=((samples>5)&(samples<250)).all(axis=1)
    return (yy[valid]*w+xx[valid]).astype(np.int32)


def medians(im,indices,linear=False):
    if not len(indices):
        return None
    val=im.reshape(-1,3)[indices]
    return np.median(LINEAR[val] if linear else val,axis=0).tolist()


def span(pairs,ids):
    if not ids:
        return [None]*6
    return np.concatenate([np.ptp(np.percentile([pairs[i][side] for i in ids],[10,90],
                    axis=0,method="linear"),axis=0) for side in (0,1)]).tolist()


def eligible(pairs,ids):
    return len(ids)>=2 and all(v>=.02 for v in span(pairs,ids))


def split(pairs):
    """IDs are physical identities; sample eligibility is applied before this function."""
    ids=sorted(pairs)
    if len(ids)>=3:
        for held in ids:
            fit=[i for i in ids if i!=held]
            if eligible(pairs,fit):
                return fit,held,""
    if eligible(pairs,ids):
        return ids,None,""
    if not ids:
        return [],None,"no_confirmed_patch"
    return [],None,"insufficient_paired_blocks" if len(ids)<2 else "insufficient_channel_span"


def solver_cfg(old_config):
    params={k:old_config[k] for k in ("epsilon","alpha","max_iter","tol")}
    return dict(**params,min_fit_tiles=2,min_linear_span=.02)


def solve(pairs,ids,cfg):
    x,y=[np.array([pairs[i][side] for i in ids]) for side in (0,1)]
    return fit_channels(x,y,cfg)


def digest_indices(a):
    return hashlib.sha256(np.asarray(a,dtype="<i4").tobytes()).hexdigest()


def checked_files(root,inventory):
    for name,digest in inventory.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or not p.is_file() or sha(p)!=digest:
            raise ValueError("Frozen input changed: "+str(name))
