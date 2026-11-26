"""Replay the historical, spatially withheld untreated-root pixels as a separate QC tissue."""
import json

import cv2
import numpy as np

from ..rgb.color import CHANNELS, differences
from ..rgb.geometry import Samples, grid, root_region, warp_to_reference, white_body


def root_grid(data,row,raw,info,cfg):
    body,_=white_body(data/row.mask_path,info,raw)
    return grid(raw,body,info,cfg)


def measure(table,ref,root,crown,cur,matrix,old,new):
    """An archived tile count, raw median or old PNG disagreement invalidates the comparison."""
    withheld=table[table.role.eq("holdout")].copy()
    if withheld.empty:
        return []
    if not withheld.side.eq(ref.tile).all():
        raise ValueError("Unexpected historical tile width")
    shape=ref.body.shape
    dy,dx=np.indices((ref.tile,ref.tile))
    yy=withheld.y.to_numpy()[:,None]+dy.ravel()
    xx=withheld.x.to_numpy()[:,None]+dx.ravel()
    cur_good=warp_to_reference(cur.good.astype(np.float32),matrix,shape,cv2.INTER_LINEAR)>=.9999
    valid=(root_region(ref,root,crown)&ref.good&cur_good)[yy,xx]
    if not np.array_equal(valid.sum(axis=1),withheld.valid_pixels):
        raise ValueError("Archived root tile support changed")
    samples=Samples(withheld,yy,xx,valid,{})
    target=samples.medians(ref.rgb)
    def aligned(pixels):
        mapped=warp_to_reference(pixels,matrix,shape,cv2.INTER_LINEAR)
        return samples.medians(mapped)
    colors={"raw":aligned(cur.rgb)}
    for k,im in (("root",old),("material",new)):
        if im is not None:
            small=cv2.resize(im,cur.body.shape[::-1],interpolation=cv2.INTER_AREA)
            colors[k]=aligned(small)
    for prefix,actual in [("reference",target),("raw",colors["raw"]),("corrected",colors.get("root"))]:
        if actual is not None and not np.allclose(actual,withheld[[prefix+"_srgb_"+c for c in CHANNELS]],atol=1e-8,rtol=0):
            raise ValueError("Archived "+prefix+" root-tile colours changed")
    result=[]
    for i,r in enumerate(withheld.itertuples()):
        result.append(dict(material="untreated_root",region=f"root_{r.x}_{r.y}_{r.side}",
                           scope="archived_withheld_root",target_rgb=json.dumps(target[i].tolist()),
                           valid_pixels=int(r.valid_pixels)))
    for name,col in colors.items():
        metrics=differences(col,target)
        for i,(de,dl,ab) in enumerate(metrics):
            result[i].update({name+"_rgb":json.dumps(col[i].tolist()),name+"_de00":de,
                              name+"_abs_dL":dl,name+"_ab_distance":ab})
    return result
