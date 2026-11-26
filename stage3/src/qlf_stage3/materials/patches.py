"""Native material observations and per-channel, current-to-anchor Huber mapping."""
from dataclasses import dataclass
import warnings

import cv2
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor

from ..rgb.color import CHANNELS, LINEAR, FrameIssue


@dataclass
class Patch:
    material: str
    region: str
    role: str
    polygon: list
    xx: np.ndarray
    yy: np.ndarray
    valid_fraction: float
    linear: np.ndarray
    srgb: np.ndarray
    reason: str
    n_total: int

    def color(self, image):
        if not len(self.xx):
            return np.full(3,np.nan)
        return np.median(image[self.yy,self.xx],axis=0)


def observe(raw, spec, drawing_size, cfg):
    """A material's location is independently supplied for each photographed scene."""
    height,width = raw.shape[:2]
    pts = np.asarray(spec["polygon"],float)
    if pts.ndim!=2 or pts.shape[1]!=2 or len(pts)<3 or not np.isfinite(pts).all():
        raise ValueError("Invalid material polygon "+spec["region"])
    pts = np.rint((pts+.5)*np.array([width,height])/drawing_size-.5).astype(np.int32)
    if (pts<0).any() or (pts>=[width,height]).any():
        raise ValueError("Material annotation outside oriented source: "+spec["region"])
    x,y,w,h = cv2.boundingRect(pts)
    local = pts - np.array([x,y],dtype=np.int32)
    area = cv2.fillPoly(np.zeros((h,w),np.uint8),[local],1)
    margin = cfg["edge_inset_native_px"]
    if margin:
        area = cv2.erode(area,np.ones((2*margin+1,2*margin+1),np.uint8),
                         borderType=cv2.BORDER_CONSTANT,borderValue=0)
    yy,xx = np.nonzero(area)
    yy,xx = yy+y,xx+x
    samples = raw[yy,xx]
    total = len(samples)
    good = ((samples>cfg["min_code_exclusive"]) & (samples<cfg["max_code_exclusive"])).all(axis=1)
    yy,xx,samples = yy[good],xx[good],samples[good]
    fraction = len(samples)/max(1,total)
    reason = ""
    if len(samples)<cfg["min_pixels"] or fraction<cfg["min_valid_fraction"]:
        reason = "insufficient_valid_material_pixels"
    if len(samples):
        encoded = np.median(samples,axis=0)
        linear = np.median(LINEAR[samples],axis=0)
    else:
        encoded,linear = np.full(3,np.nan),np.full(3,np.nan)
    return Patch(spec["material"],spec["region"],spec["role"],pts.tolist(),xx,yy,
                 fraction,linear,encoded,reason,total)


def paired(current, anchor):
    pairs = []
    for ident in sorted(set(current)&set(anchor)):
        left,right = current[ident],anchor[ident]
        if left.material!=right.material or left.role!=right.role:
            raise ValueError("Conflicting material identity or split: "+ident)
        if not (left.reason or right.reason):
            pairs.append((left,right))
    return pairs


def huber_map(x, y, weights, cfg):
    """Three independent weighted regressions in linear sRGB. Rows are matched physical blocks."""
    x,y,weights = np.asarray(x),np.asarray(y),np.asarray(weights)
    if x.shape!=y.shape or x.ndim!=2 or x.shape[1]!=3 or len(x)<2 or weights.shape!=(len(x),):
        raise FrameIssue("insufficient_fit_blocks")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(weights).all() or (weights<=0).any():
        raise FrameIssue("invalid_fit_observation")
    coeffs = []
    for i,c in enumerate(CHANNELS):
        reg = HuberRegressor(**{k:cfg[k] for k in ("epsilon","alpha","tol","max_iter")})
        with warnings.catch_warnings():
            warnings.simplefilter("error",ConvergenceWarning)
            try:
                reg.fit(x[:,i,None],y[:,i],sample_weight=weights)
            except (ConvergenceWarning,FloatingPointError,ValueError) as ex:
                raise FrameIssue("huber_not_converged",c) from ex
        a,b = float(reg.coef_[0]),float(reg.intercept_)
        if not np.isfinite([a,b]).all() or a<=0:
            raise FrameIssue("invalid_huber_parameters",c)
        coeffs.append((a,b,int(reg.n_iter_),float(np.median(np.abs(y[:,i]-a*x[:,i]-b)))))
    return coeffs
