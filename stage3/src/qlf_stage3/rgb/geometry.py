"""White-light-only anatomy, landmark checks and spatially disjoint paired tiles."""
from dataclasses import dataclass
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree

from .color import CHANNELS, LINEAR, FrameIssue


def homogeneous(matrix):
    return np.vstack([matrix, [0.,0.,1.]]) if np.shape(matrix)==(2,3) else np.asarray(matrix,float)


def transform(points, matrix):
    p = np.asarray(points, float)
    return p @ matrix[:2,:2].T + matrix[:2,2]


def resize_matrix(width, height, size):
    x, y = size[0]/width, size[1]/height
    return np.array([[x,0,(x-1)/2],[0,y,(y-1)/2],[0,0,1.]])


def orientation_basis(w, h, orientation):
    choices = {1:((1,0,0),(0,1,0)), 2:((-1,0,w-1),(0,1,0)), 3:((-1,0,w-1),(0,-1,h-1)),
               4:((1,0,0),(0,-1,h-1)), 5:((0,1,0),(1,0,0)), 6:((0,-1,h-1),(1,0,0)),
               7:((0,-1,h-1),(-1,0,w-1)), 8:((0,1,0),(-1,0,w-1))}
    return homogeneous(choices[orientation])


def read_body(path, info):
    with Image.open(path) as im:
        if im.size != (info["native_width"],info["native_height"]):
            raise FrameIssue("invalid_white_mask", "Body dimensions differ from source photograph")
        orient = info["orientation"]
        method = {2:Image.Transpose.FLIP_LEFT_RIGHT,3:Image.Transpose.ROTATE_180,4:Image.Transpose.FLIP_TOP_BOTTOM,
                  5:Image.Transpose.TRANSPOSE,6:Image.Transpose.ROTATE_270,7:Image.Transpose.TRANSVERSE,
                  8:Image.Transpose.ROTATE_90}.get(orient)
        mask = np.asarray(im.transpose(method) if method is not None else im)
    if mask.ndim != 2:
        raise FrameIssue("invalid_white_mask","Expected a single-channel native RGB body mask")
    return mask > 0


def white_body(path, info, rgb):
    native = read_body(path,info)
    # On a dark carrier a few archived masks outline the carrier itself. Validate
    # the mask against an independently measured central bright foreground before reuse.
    h,w = native.shape
    f = min(1.,672/max(w,h))
    size = (round(w*f),round(h*f))
    gray = cv2.cvtColor(cv2.resize(rgb,size,interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray,(5,5),0)
    _,binary = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    contours,_ = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return native,"native_white_body"
    def central_area(contour):
        x,y,bw,bh = cv2.boundingRect(contour)
        cx,cy = (x+bw/2)/size[0],(y+bh/2)/size[1]
        return cv2.contourArea(contour) if .22<cx<.78 and .2<cy<.80 else 0
    candidate = max(contours,key=central_area)
    frac = central_area(candidate)/gray.size
    if not .015<frac<.5:
        return native,"native_white_body"
    measured = np.zeros_like(gray)
    cv2.drawContours(measured,[candidate],-1,1,cv2.FILLED)
    small = cv2.resize(native.astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)
    agreement = (measured & small).sum()/max(1,(measured | small).sum())
    if agreement>=.15:
        return native,"native_white_body"
    return cv2.resize(measured,(w,h),interpolation=cv2.INTER_NEAREST).astype(bool),"verified_white_foreground_fallback"


@dataclass
class Grid:
    rgb: np.ndarray
    body: np.ndarray
    good: np.ndarray
    resize: np.ndarray
    native_to_work: np.ndarray
    tile: int


def grid(rgb, native_body, info, cfg):
    h,w = rgb.shape[:2]
    f = min(1.,cfg["working_long_edge"]/max(w,h))
    size = round(w*f),round(h*f)
    image = cv2.resize(rgb,size,interpolation=cv2.INTER_AREA)
    body = cv2.resize(native_body.astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)
    tile = max(3,round(cfg["tile_at_1600"]*max(size)/1600))
    margin = max(1,round(tile*cfg["boundary_erosion_tiles"]))
    body = cv2.erode(body,np.ones((2*margin+1,2*margin+1),np.uint8)).astype(bool)
    low,high = rgb.min(axis=2),rgb.max(axis=2)
    luma = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    good = ((low>cfg["minimum_code_exclusive"]) & (high<cfg["maximum_code_exclusive"]) &
            (luma>=cfg["min_luma_code"]) & ~((luma>=cfg["glare_luma_code"]) & (high-low<=cfg["glare_chroma_code"])))
    # Reject a working pixel if its interpolation footprint contained masked native pixels.
    valid = cv2.resize(good.astype(np.float32),size,interpolation=cv2.INTER_AREA) >= .9999
    scale = resize_matrix(w,h,size)
    return Grid(image,body,valid & body,scale,
                scale @ orientation_basis(info["native_width"],info["native_height"],info["orientation"]),tile)


def polygon_mask(shape, points):
    return cv2.fillPoly(np.zeros(shape,np.uint8),[np.rint(points).astype(np.int32)],1).astype(bool)


def polygons(annotation, grid_size, reference):
    h,w = reference.body.shape
    scale = np.array([w,h])/grid_size
    return [np.asarray(annotation[k],float)*scale for k in ("root","crown")]


def landmarks(frame, cfg, anatomical=None):
    h,w = frame.body.shape
    f = min(1.,cfg["landmark_long_edge"]/max(w,h))
    size = round(w*f),round(h*f)
    basis = resize_matrix(w,h,size)
    gray = cv2.cvtColor(cv2.resize(frame.rgb,size,interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2GRAY)
    mask = frame.body if anatomical is None else frame.body & anatomical
    mask = cv2.resize(mask.astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)
    gray = cv2.createCLAHE(2.,(8,8)).apply(gray)
    points, descriptor = cv2.SIFT_create(nfeatures=1600, contrastThreshold=.018).detectAndCompute(gray,mask*255)
    xy = np.array([p.pt for p in points],float).reshape(-1,2)
    return transform(xy,np.linalg.inv(basis)),descriptor


def reciprocal_matches(ref, cur, ratio):
    left, right = ref[1],cur[1]
    if left is None or right is None or min(len(left),len(right))<2:
        return np.empty((0,2)),np.empty((0,2))
    def best(a,b):
        pairs = cv2.BFMatcher().knnMatch(a,b,k=2)
        return {m.queryIdx:m.trainIdx for m,n in pairs if m.distance < ratio*n.distance}
    forward, back = best(left,right),best(right,left)
    # SIFT may emit several orientations for one physical landmark.
    unique = {}
    for i,j in forward.items():
        if back.get(j)==i:
            key = tuple(np.rint(ref[0][i]/3).astype(int))
            unique.setdefault(key,(ref[0][i],cur[0][j]))
    pairs = [unique[k] for k in sorted(unique)]
    if not pairs:
        return np.empty((0,2)),np.empty((0,2))
    return np.array([x for x,_ in pairs]),np.array([y for _,y in pairs])


def warp_to_reference(array, ref_to_cur, ref_shape, interp):
    h,w = ref_shape
    return cv2.warpAffine(array,ref_to_cur[:2],(w,h),flags=interp | cv2.WARP_INVERSE_MAP)


def root_region(frame, root, crown):
    # Exclude the crown and an external buffer even where hand-drawn envelopes touch.
    crown_mask = polygon_mask(frame.body.shape,crown).astype(np.uint8)
    radius = frame.tile
    buffered = cv2.dilate(crown_mask,np.ones((2*radius+1,2*radius+1),np.uint8))
    return polygon_mask(frame.body.shape,root) & frame.body & (buffered==0)


def anatomy_edges(frame, root, crown):
    shape = frame.body.shape
    radius = 2*frame.tile
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    root_band = cv2.dilate(polygon_mask(shape,root).astype(np.uint8),kernel)>0
    neck_band = root_band & (cv2.dilate(polygon_mask(shape,crown).astype(np.uint8),kernel)>0)
    body_edge = cv2.morphologyEx(frame.body.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
    gray = cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.createCLAHE(2.,(8,8)).apply(gray),45,110)>0
    return [(body_edge,root_band),(edges,neck_band & frame.body)]


def check_anatomy(ref,cur,matrix,root,crown):
    # Image and mask edges are independent of the descriptor matches used to estimate geometry.
    reference_edges = anatomy_edges(ref,root,crown)
    current_edges = anatomy_edges(cur,transform(root,matrix),transform(crown,matrix))
    diagnostics, bad, contour_support = {}, [], np.empty((0,2))
    for label,(edge,zone),(target,target_zone) in zip(("root_contour","cervical_edge"),reference_edges,current_edges):
        yy,xx = np.nonzero(edge & zone)
        points = np.column_stack((xx,yy))[::6]
        distance = cv2.distanceTransform((~(target & target_zone)).astype(np.uint8),cv2.DIST_L2,5)
        projected = np.rint(transform(points,matrix)).astype(int)
        inside = ((projected>=0) & (projected<np.array(cur.body.shape[::-1]))).all(axis=1)
        residual = np.full(len(points),2.*ref.tile)
        u,v = projected[inside].T
        scale = np.sqrt(np.linalg.det(matrix[:2,:2]))
        residual[inside] = distance[v,u]/scale
        diagnostics[label+"_n"] = len(points)
        if len(points):
            diagnostics[label+"_median_px"] = float(np.median(residual))
            diagnostics[label+"_p90_px"] = float(np.percentile(residual,90))
            bad.extend(points[residual>=ref.tile])
            if label=="root_contour":
                contour_support = points[residual<.45*ref.tile]
    return diagnostics,np.asarray(bad,float).reshape(-1,2),contour_support


def align(reference, current, roots, ref_points, archived, cfg):
    root, crown = roots
    rp,cp = reciprocal_matches(ref_points, landmarks(current,cfg),cfg["descriptor_ratio"])
    check = np.arange(len(rp)) % 3 == 0
    minimum = cfg["registration_min_checks"]
    if check.sum()<minimum:
        raise FrameIssue("registration_failed",f"Only {check.sum()} independent landmark matches")
    candidates = []
    if archived is not None:
        candidates.append(("native_rgb_record",current.native_to_work @ homogeneous(archived) @
                           np.linalg.inv(reference.native_to_work)))
    if (~check).sum()>=4:
        cv2.setRNGSeed(3127)
        sim,_ = cv2.estimateAffinePartial2D(rp[~check],cp[~check],method=cv2.RANSAC,
                    ransacReprojThreshold=.55*reference.tile, maxIters=2000,confidence=.995)
        if sim is not None:
            candidates.append(("rgb_landmark_similarity",homogeneous(sim)))
    reasons = []
    for name,matrix in candidates:
        singular = np.linalg.svd(matrix[:2,:2],compute_uv=False)
        if np.linalg.det(matrix[:2,:2])<=0 or singular.min()<.55 or singular.max()>1.8:
            reasons.append(name+": scale/orientation")
            continue
        # Residuals and support radii are reported on the reference working grid.
        residual = np.linalg.norm(transform(cp,np.linalg.inv(matrix))-rp,axis=1)
        inlier = residual < reference.tile
        clean = check & inlier
        root_location = cv2.pointPolygonTest
        root_near = np.array([root_location(root.astype(np.float32),tuple(p),True)>-reference.tile for p in rp])
        overlap = warp_to_reference(current.body.astype(np.uint8),matrix,reference.body.shape,cv2.INTER_NEAREST)
        refroot = root_region(reference,root,crown)
        coverage = np.count_nonzero(refroot & (overlap>0))/max(1,refroot.sum())
        if (clean.sum()<minimum or clean.sum()/check.sum()<cfg["registration_min_fraction"] or
                coverage<cfg["min_root_body_overlap"]):
            reasons.append(f"{name}: checks={clean.sum()}/{check.sum()}, root_matches={sum(inlier & root_near)}, overlap={coverage:.3f}")
            continue
        checks = np.column_stack([rp[check],cp[check],residual[check]])
        edge_qc,rejected,contour_support = check_anatomy(reference,current,matrix,root,crown)
        if any(edge_qc[k+"_n"]>=6 and edge_qc[k+"_median_px"]>=reference.tile
               for k in ("root_contour","cervical_edge")):
            reasons.append(name+": cervical or root outline discrepancy reaches a tile width")
            continue
        contour_ok = (len(contour_support)>=8 and np.ptp(contour_support,axis=0).max()>=3*reference.tile and
                      len(contour_support)>=.55*edge_qc["root_contour_n"] and edge_qc["cervical_edge_n"]>=6)
        if np.count_nonzero(inlier & root_near)<2 and not contour_ok:
            reasons.append(name+": insufficient independent support on visible root")
            continue
        support = np.vstack((rp[inlier],contour_support)) if contour_ok else rp[inlier]
        return matrix,support,rejected,dict(**edge_qc,edge_rejections=len(rejected),contour_support=len(contour_support),
                   method=name,check_matches=int(check.sum()),check_inliers=int(clean.sum()),
                   check_median_px=float(np.median(residual[check])),root_overlap=float(coverage),
                   root_matches=int(np.count_nonzero(inlier & root_near)),check_points=json.dumps(checks.round(3).tolist()))
    raise FrameIssue("registration_failed"," | ".join(reasons) or "No plausible white-light transform")


@dataclass
class Samples:
    table: pd.DataFrame
    yy: np.ndarray
    xx: np.ndarray
    valid: np.ndarray
    split: dict

    def medians(self,image,linear=False):
        values = image[self.yy,self.xx]
        values = LINEAR[values] if linear else values.astype(float)
        return np.nanmedian(np.where(self.valid[...,None],values,np.nan),axis=1)


def sample_tiles(ref,cur,matrix,root,crown,points,cfg,rejected=None):
    roi = root_region(ref,root,crown)
    if not roi.any():
        raise FrameIssue("root_not_visible")
    y,x = np.nonzero(roi)
    step = ref.tile
    axis = 0 if np.ptp(x)>=np.ptp(y) else 1
    axis_values = (x,y)[axis]
    lo,hi = axis_values.min(),axis_values.max()+1
    q2,q3 = lo+(hi-lo)*.5,lo+(hi-lo)*.75
    xs,ys = np.meshgrid(np.arange(int(x.min())//step*step, min(int(x.max())+1,roi.shape[1]-step+1),step),
                        np.arange(int(y.min())//step*step, min(int(y.max())+1,roi.shape[0]-step+1),step))
    x0,y0 = xs.ravel(),ys.ravel()
    pos = (x0,y0)[axis]
    hold = (pos>=q2) & (pos+step<=q3)
    fitting = (pos+step<=q2-step) | (pos>=q3+step)
    keep = fitting | hold
    if points is not None:
        coords = np.column_stack([x0+step/2,y0+step/2])
        keep &= cKDTree(points).query(coords)[0] <= cfg["max_landmark_gap_tiles"]*step
    if rejected is not None and len(rejected):
        centers = np.column_stack([x0+step/2,y0+step/2])
        keep &= cKDTree(rejected).query(centers)[0] > 1.75*step
    x0,y0,hold = x0[keep],y0[keep],hold[keep]
    dy,dx=np.indices((step,step))
    yy,xx=y0[:,None]+dy.ravel(),x0[:,None]+dx.ravel()
    cur_good = warp_to_reference(cur.good.astype(np.float32),matrix,roi.shape,cv2.INTER_LINEAR)>=.9999
    valid = (roi & ref.good & cur_good)[yy,xx]
    keep = valid.sum(axis=1) >= np.ceil(step**2*cfg["min_valid_fraction"])
    if not keep.any():
        raise FrameIssue("root_not_visible","No mutually valid root blocks")
    df = pd.DataFrame(dict(x=x0[keep],y=y0[keep],side=step,role=np.where(hold[keep],"holdout","fit"),
                           valid_pixels=valid[keep].sum(axis=1)))
    split=dict(axis="xy"[axis],cut2=float(q2),cut3=float(q3),guard=step)
    return Samples(df,yy[keep],xx[keep],valid[keep],split)


def native_crown(ref_crown,cur,native_body,matrix):
    points = transform(ref_crown,np.linalg.inv(cur.resize) @ matrix)
    envelope = polygon_mask(native_body.shape,points)
    mask = envelope & native_body
    if mask.sum()<500 or mask.sum()/max(1,envelope.sum())<.4:
        return None
    return mask
