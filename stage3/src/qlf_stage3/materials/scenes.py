"""Photo-bound region review and within-burst geometry; no cross-event image registration."""
from collections import Counter

import cv2
import numpy as np

from ..rgb.color import CHANNELS, FrameIssue
from .patches import huber_map, observe


def edges(raw, size):
    gray = cv2.cvtColor(cv2.resize(raw,tuple(size),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray,(3,3),.7)
    dx,dy = [cv2.Sobel(gray,cv2.CV_32F,*d,ksize=3) for d in ((1,0),(0,1))]
    return np.log1p(cv2.magnitude(dx,dy))


def locate(raw, basis_edges, same_image, cfg):
    if same_image:
        return np.zeros(2),1.
    size = cfg["burst_edge_size"]
    current = edges(raw,size)
    window = cv2.createHanningWindow(tuple(size),cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(basis_edges.copy(),current,window)
    native = np.asarray(shift)*np.array(raw.shape[1::-1])/size
    if (not np.isfinite([*native,response]).all() or response<cfg["min_burst_response"]
            or np.linalg.norm(native)>cfg["max_burst_drift_native_px"]):
        raise FrameIssue("burst_position_unconfirmed",f"native dx/dy={native.tolist()}, response={response:.4f}")
    return native,float(response)


def observations(raw, shift, event, annotation, cfg):
    size = np.array(annotation["drawing_size"])
    if not np.array_equal(size,raw.shape[1::-1]):
        raise ValueError("Reviewed regions require the recorded oriented native dimensions")
    patches = {}
    occupied = []
    for region, polygon in event["regions"].items():
        material = annotation["block_material"][region.split("_")[0]]
        role = annotation["material_roles"][material]
        spec = dict(region=region,material=material,role=role,polygon=(np.asarray(polygon)+shift).tolist())
        p = observe(raw,spec,size,cfg)
        x,y,w,h = cv2.boundingRect(np.asarray(p.polygon,np.int32))
        if any(x<u+uw and x+w>u and y<v+vh and y+h>v for u,v,uw,vh in occupied):
            raise ValueError("Overlapping reviewed regions: "+region)
        occupied.append((x,y,w,h))
        patches[region] = p
    return patches


def fit_pairs(pairs,cfg,record):
    fitting = [(p,q) for p,q in pairs if p.role=="fit"]
    counts = Counter(p.material for p,q in fitting)
    record.update(n_fit_blocks=len(fitting),n_fit_materials=len(counts),
                  n_holdout_blocks=sum(p.role=="holdout" for p,q in pairs))
    # Diagnostic only, including the reserved disc. It never enters the regression or gates.
    if pairs:
        for pos,side in enumerate(("current","anchor")):
            visible = np.array([pair[pos].linear for pair in pairs])
            for c,span in zip(CHANNELS,np.ptp(visible,axis=0)):
                record[side+"_all_visible_range_"+c]=float(span)
    if fitting:
        x = np.array([p.linear for p,q in fitting])
        y = np.array([q.linear for p,q in fitting])
        spans = np.array([np.ptp(np.percentile(z,[10,90],axis=0),axis=0) for z in (x,y)])
        for label,s in zip(("current","anchor"),spans):
            for c,value in zip(CHANNELS,s):
                record[label+"_span_"+c]=float(value)
    if len(counts)<cfg["min_fit_materials"]:
        raise FrameIssue("insufficient_matched_materials",", ".join(sorted(counts)) or "no accepted matched material")
    if len(fitting)<cfg["min_fit_blocks"]:
        raise FrameIssue("insufficient_fit_blocks")
    if np.any(spans<cfg["min_linear_span"]):
        raise FrameIssue("insufficient_channel_span")
    # The absolute normalization is fixed as well as the relative material weights.
    weights = [len(fitting)/(len(counts)*counts[p.material]) for p,q in fitting]
    return huber_map(x,y,weights,cfg)
