"""Reconstruct calibration results from source photos, native masks and the saved lossless outputs."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.linear_model import HuberRegressor
from skimage.color import deltaE_ciede2000, rgb2lab
from threadpoolctl import threadpool_limits

from ..files import dump, load, sha
from ..modeling.vision import GRAY_MODE, white_input
from .calibration import METRICS, prepare_reference, source
from .color import CHANNELS, CORRECTED_MODE, RAW_MODE, apply_rgb, decode, fit_channels, pixel_hash
from .geometry import grid, native_crown, sample_tiles, warp_to_reference
from .report import tiles_for
from .study import pilot_slots, read_study


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual,expected,title,tol=1e-8):
    if not np.allclose(actual,expected,atol=tol,rtol=0,equal_nan=False):
        raise ValueError(f"{title} differs: {actual} versus {expected}")


def independent_linear(codes):
    s = np.asarray(codes,dtype=float)/255
    return np.where(s<=.04045,s/12.92,((s+.055)/1.055)**2.4)


def quantize(light):
    v = np.clip(light,0.,1.)
    return np.rint(255*np.where(v<=.0031308,12.92*v,1.055*v**(1/2.4)-.055)).astype(np.uint8)


def synthetic(cfg):
    z = np.linspace(.07,.64,81)
    x = np.column_stack((z,z[::-1],np.roll(z,11)))
    a,b = np.array([1.09,.81,1.21]),np.array([.025,.062,-.014])
    y = x*a+b
    rng = np.random.default_rng(91)
    y += rng.normal(0,1e-6,y.shape)
    y[[9,28,60]] += .10
    params = fit_channels(x,y,cfg)
    close([p[0] for p in params],a,"Synthetic RGB gains",.0001)
    close([p[1] for p in params],b,"Synthetic RGB intercepts",.0001)
    codes = np.arange(256,dtype=np.uint8).reshape(16,16)
    rgb = np.stack([codes,np.rot90(codes),np.flipud(codes)],axis=-1)
    saved,_,_ = apply_rgb(rgb,a,b,None)
    require(np.array_equal(saved,quantize(independent_linear(rgb)*a+b)),"Synthetic channel order or mapping direction")
    identity,_,_ = apply_rgb(rgb,[1]*3,[0]*3,None)
    require(np.array_equal(rgb,identity),"Identity 8-bit sRGB round trip")
    de = deltaE_ciede2000(np.array([[50,2.6772,-79.7751]]),np.array([[50,0,-82.7485]]))
    close(de,2.0425,"CIEDE2000 reference pair",.00005)
    return True


def check_tile_ledger(out,cal,qc,cfg):
    index = cal.set_index("rgb_image_id")
    count,seen = 0,set()
    accum = {}
    # Each recorded frame is contiguous, but a chunk boundary may bisect its block list.
    tail = None
    def check_one(frame_id,rows):
        r = index.loc[frame_id]
        require(frame_id not in seen,"Non-contiguous duplicate block set "+frame_id)
        seen.add(frame_id)
        require(rows.anchor_rgb_id.eq(r.anchor_rgb_id).all(),"Reference mismatch in block ledger")
        side = int(r.tile_side)
        require((rows.side==side).all() and (rows[["x","y"]]%side==0).all().all() and
                not rows.duplicated(["x","y"]).any(),"Overlapping or off-grid tiles "+frame_id)
        p = rows[r.split_axis]
        val = rows.role.eq("holdout")
        fit = rows.role.eq("fit")
        require((fit|val).all() and (p[val]>=r.split_cut2).all() and (p[val]+side<=r.split_cut3).all(),"Holdout partition "+frame_id)
        require(((p[fit]+side<=r.split_cut2-side)|(p[fit]>=r.split_cut3+side)).all(),"Missing spatial guard "+frame_id)
        require(int(val.sum())==r.holdout_tiles and int(fit.sum())==r.fit_tiles,"Recorded tile counts "+frame_id)
        require(rows.valid_pixels.between(np.ceil(side*side*cfg["min_valid_fraction"]),side*side).all(),"Invalid area "+frame_id)
        if r.status=="ok":
            q = qc.loc[frame_id]
            require(fit.sum()>=cfg["min_fit_tiles"] and val.sum()>=cfg["min_holdout_tiles"],"Too few valid blocks")
            target = rgb2lab(rows.loc[val,["reference_srgb_"+c for c in CHANNELS]].to_numpy()/255)
            for prefix,col in [("before","raw_srgb_"),("after","corrected_srgb_")]:
                lab = rgb2lab(rows.loc[val,[col+c for c in CHANNELS]].to_numpy()/255)
                d = np.column_stack((deltaE_ciede2000(lab,target),np.abs(lab[:,0]-target[:,0]),np.linalg.norm(lab[:,1:]-target[:,1:],axis=1)))
                close(d,rows.loc[val,[prefix+"_"+k for k in METRICS]].to_numpy(),"Per-block colour "+frame_id,1e-9)
                close(np.median(d,axis=0),q[[prefix+"_"+k for k in METRICS]].to_numpy(float),"Frame colour "+frame_id,1e-9)
            for k in METRICS:
                close(q[k+"_gain"],q["before_"+k]-q["after_"+k],"Paired gain "+frame_id)
        accum[frame_id] = (int(fit.sum()),int(val.sum()))
    for chunk in pd.read_csv(out/"root_tiles.csv",chunksize=20000):
        count += len(chunk)
        if tail is not None:
            chunk = pd.concat((tail,chunk),ignore_index=True)
        if chunk.empty:
            continue
        last = chunk.rgb_image_id.iloc[-1]
        tail = chunk.loc[chunk.rgb_image_id.eq(last)]
        for frame_id,rows in chunk.loc[chunk.rgb_image_id.ne(last)].groupby("rgb_image_id",sort=False):
            check_one(frame_id,rows)
    if tail is not None:
        check_one(tail.rgb_image_id.iloc[0],tail)
    tiled = index.get("fit_tiles",pd.Series(index=index.index,dtype=float)).notna()
    require(seen==set(index.index[tiled]),"Missing frame in block ledger")
    fields = ["split_axis","split_cut2","split_cut3","split_guard_px"]
    splits = cal.reindex(columns=["anchor_rgb_id",*fields]).dropna(subset=["split_axis"]).groupby("anchor_rgb_id")[fields].nunique()
    require((splits<=1).all().all(),"Reference partition varies by target frame")
    return {"tiles":count,"frames_with_tiles":len(accum)}


def input_smoke(path,mode,pixels,position=None):
    im = white_input(path,mode)
    small = ImageOps.contain(Image.fromarray(pixels),(224,224),Image.Resampling.BILINEAR)
    canvas = Image.new("RGB",(224,224))
    canvas.paste(small,((224-small.width)//2,(224-small.height)//2))
    arr = np.asarray(im)
    require(im.mode=="RGB" and im.size==(224,224) and np.array_equal(arr,np.asarray(canvas)),"White-light model entry changed field/colour")
    y,x = np.unravel_index(np.ptp(arr,axis=2).argmax(),arr.shape[:2]) if position is None else position
    return {"x":int(x),"y":int(y),"rgb":arr[y,x].tolist()}


def replay_frame(r,q,out,photos,anchors,annotations,cfg):
    photo = photos[r.rgb_image_id]
    known = anchors[(r.tooth_id,r.angle_id)][0]
    ref = prepare_reference(known,annotations["views"],annotations["drawing_grid"],cfg)
    raw,body,info = source(photo)
    cur = grid(raw,body,info,cfg)
    m = np.array(json.loads(r.reference_to_current_work))
    candidates = sample_tiles(ref.grid,cur,m,ref.root,ref.crown,None,cfg)
    table = tiles_for(out,[r.rgb_image_id])
    lookup = {(int(x),int(y)):i for i,(x,y) in enumerate(candidates.table[["x","y"]].values)}
    idx = [lookup[(x,y)] for x,y in table[["x","y"]].itertuples(index=False,name=None)]
    valid = candidates.valid[idx]
    yy,xx = candidates.yy[idx],candidates.xx[idx]
    require(np.array_equal(valid.sum(axis=1),table.valid_pixels),"Changed pre-correction valid pixels")
    def medians(image,linear=False):
        pixels = image[yy,xx]
        values = independent_linear(pixels) if linear else pixels.astype(float)
        return np.nanmedian(np.where(valid[...,None],values,np.nan),axis=1)
    def aligned(image):
        return warp_to_reference(image,m,ref.grid.body.shape,cv2.INTER_LINEAR)
    original = aligned(cur.rgb)
    x,y = medians(original,True),medians(ref.grid.rgb,True)
    fit = table.role.eq("fit").to_numpy()
    val = ~fit
    for name,values in [("current_linear_",x),("reference_linear_",y)]:
        close(values,table[[name+c for c in CHANNELS]].to_numpy(),"Direct root samples")
    params = []
    for i,c in enumerate(CHANNELS):
        reg = HuberRegressor(**{k:cfg[k] for k in ("epsilon","alpha","max_iter","tol")}).fit(x[fit,i,None],y[fit,i])
        close([reg.coef_[0],reg.intercept_],[r["a_"+c],r["b_"+c]],"Refitted Huber "+c,.00002)
        params.append([float(reg.coef_[0]),float(reg.intercept_)])
    dest = out/r.corrected_path
    saved,_ = decode(dest,corrected=True)
    crown = native_crown(ref.crown,cur,body,m)
    out_of_range = np.zeros(raw.shape[:2],bool)
    for i,c in enumerate(CHANNELS):
        # Bound memory by looking up the linear map for each of the 256 possible source codes.
        values = (independent_linear(np.arange(256))*r["a_"+c]+r["b_"+c])[raw[...,i]]
        require(np.array_equal(quantize(values),saved[...,i]),"Saved native pixels differ for "+c)
        close([values.min(),values.max()],[q["preclip_min_"+c],q["preclip_max_"+c]],"Preclip range "+c)
        close([(values<0).sum(),(values>1).sum()],[q["below_zero_"+c],q["above_one_"+c]],"Preclip count "+c)
        out_of_range |= (values<0)|(values>1)
        del values
    close(out_of_range.sum(),q.whole_outside_px,"All-field clipping")
    crown_rate = None
    if crown is not None:
        originally_internal = (raw.min(axis=2)>0)&(raw.max(axis=2)<255)
        new = int((originally_internal & crown & out_of_range).sum())
        crown_rate = float(100*new/crown.sum())
        close([crown.sum(),new,crown_rate],[q.crown_pixels,q.crown_new_clip_px,q.crown_new_clip_pct],"Crown clipping before quantisation")
    small = cv2.resize(saved,cur.body.shape[::-1],interpolation=cv2.INTER_AREA)
    target = rgb2lab(medians(ref.grid.rgb)[val]/255)
    measurements = {}
    for prefix,image,name in [("before",original,"raw_srgb_"),("after",aligned(small),"corrected_srgb_")]:
        block = medians(image)[val]
        close(block,table.loc[val,[name+c for c in CHANNELS]].to_numpy(),"Saved representative colours")
        lab = rgb2lab(block/255,illuminant="D65",observer="2")
        d = float(np.median(deltaE_ciede2000(lab,target,kL=1,kC=1,kH=1)))
        close(d,q[prefix+"_de00"],"Direct held-out delta E",1e-9)
        measurements[prefix+"_de00"] = d
    gray = np.asarray(white_input(photo.path,GRAY_MODE))
    require(np.array_equal(gray[...,0],gray[...,1]) and np.array_equal(gray[...,1],gray[...,2]),"Historical grayscale entry")
    raw_pixel = input_smoke(photo.path,RAW_MODE,raw)
    corrected_pixel = input_smoke(dest,CORRECTED_MODE,saved,(raw_pixel["y"],raw_pixel["x"]))
    return dict(rgb_image_id=r.rgb_image_id,anchor_rgb_id=known.image_id,state_id=r.state_id,channel_fits=params,
                fit_tiles=int(fit.sum()),holdout_tiles=int(val.sum()),crown_new_clip_pct=crown_rate,**measurements,
                raw_input_pixel=raw_pixel,corrected_input_pixel=corrected_pixel)


def compare_with(out,other,cal):
    other = Path(other).resolve()
    require(out!=other,"Choose an independently generated second run")
    for name in ["scope","cache_protocol","input_hashes","protocol","annotation_sha256"]:
        require(load(out/"config.json")[name]==load(other/"config.json")[name],"Different run input or protocol: "+name)
    files = ["anchors.csv","rgb_calibration.csv","root_tiles.csv","qc_by_frame.csv","qc_by_event.csv","qc_by_tooth.csv",
             "qc_coverage.csv","qc_failure_cases.csv","qc_s0_repeats.csv","paired_rgb_inputs.csv","summary.json"]
    for name in files:
        require(sha(out/name)==sha(other/name),"Different replay ledger: "+name)
    saved = cal[cal.corrected_path.notna()]
    require({p.name for p in (other/"corrected_rgb").glob("*.png")}=={Path(p).name for p in saved.corrected_path},"Replay file membership differs")
    for r in saved.itertuples():
        require(sha(other/r.corrected_path)==r.corrected_sha256,"Replay actual PNG differs: "+r.rgb_image_id)
    return {"same_tables":files,"identical_native_pngs":len(saved),"coefficients_status_pixel_hashes":"exact"}


def encoder_smoke(frame,run,data):
    from ..files import PROTOCOL
    from ..modeling.vision import imagenet18, white_vectors
    cfg = load(PROTOCOL)
    checkpoint = data/"weights/resnet18-f37072fd.pth"
    if checkpoint.is_file():
        cfg["checkpoint"] = str(checkpoint)
    net,weights,_ = imagenet18(cfg)
    original = white_vectors([data/frame.source_path],net,RAW_MODE)
    corrected = white_vectors([run/frame.corrected_path],net,CORRECTED_MODE)
    return {"rgb_image_id":frame.rgb_image_id,"weights_sha256":sha(weights),"features_per_input":int(original.shape[1]),
            "distance_between_inputs":float(np.linalg.norm(original-corrected)),"classifier_fitted":False}


def verify(run,data,study_run,compare=None,with_encoder=False):
    out,data,study_run = map(lambda p:Path(p).resolve(),(run,data,study_run))
    require(load(out/"status.json")["status"]=="complete","Run is incomplete")
    doc,annotations = load(out/"config.json"),load(out/"regions.json")
    cfg = doc["protocol"]
    groups,photos,anchors,inputs = read_study(data,study_run,annotations["views"])
    require(inputs==doc["input_hashes"],"Upstream cohort or QLF input changed")
    selected = pilot_slots(groups,cfg) if doc["scope"]=="pilot" else set(groups)
    expected = {m["rgb_image_id"]:m for k in selected for m in groups[k]}
    cal = pd.read_csv(out/"rgb_calibration.csv")
    qc = pd.read_csv(out/"qc_by_frame.csv").set_index("rgb_image_id",verify_integrity=True)
    require(cal.sample_id.is_unique and cal.rgb_image_id.is_unique and
            set(cal.rgb_image_id)==set(expected)==set(qc.index),"Source and output IDs differ")
    native_refs = pd.read_csv(out/"anchors.csv")
    require(not native_refs.duplicated(["tooth_id","angle_id"]).any(),"Duplicate view in reference ledger")
    for a in native_refs.itertuples():
        if pd.notna(a.anchor_rgb_id):
            known = anchors[(a.tooth_id,a.angle_id)][0]
            require(known.image_id==a.anchor_rgb_id and sha(known.path)==a.anchor_sha256,"Reference source identity")
    expected_files = set()
    for r in cal.itertuples():
        meta = expected[r.rgb_image_id]
        require(all(meta[k]==getattr(r,k) for k in ("sample_id","tooth_id","angle_id","state_id","group_id","rgb_event","qlf_event","qlf_image_id")),"Source pairing changed")
        require(r.status==qc.loc[r.rgb_image_id,"status"] and r.status in ("reference","ok","ok_no_holdout","failed"),"QC status mismatch")
        if r.reason not in ("source_missing","source_hash_mismatch"):
            require(sha(data/r.source_path)==r.expected_source_sha256,"Changed actual raw file "+r.rgb_image_id)
        if r.status=="failed":
            require(pd.isna(r.corrected_path) and pd.notna(r.reason),"Failed frame has a PNG or no reason")
            continue
        require(r.corrected_path=="corrected_rgb/"+r.rgb_image_id+".png","Unexpected output location")
        target = out/r.corrected_path
        expected_files.add(target.name)
        require(sha(target)==r.corrected_sha256,"Changed PNG bytes "+r.rgb_image_id)
        pixels,info = decode(target,corrected=True)
        require(pixel_hash(pixels)==r.corrected_pixel_sha256 and info["width"]==r.width and info["height"]==r.height,"Native colour or field changed")
        if r.status=="reference":
            require(r.rgb_image_id==r.anchor_rgb_id and r.source_pixel_sha256==r.corrected_pixel_sha256,"Identity reference changed")
        elif r.status=="ok_no_holdout":
            require(r.holdout_tiles<cfg["min_holdout_tiles"] and pd.isna(qc.loc[r.rgb_image_id].get("before_de00",np.nan)),"Insufficient holdout scored")
    require(expected_files=={p.name for p in (out/"corrected_rgb").glob("*.png")},"Outputs do not match manifest")
    summary = load(out/"summary.json")
    require(sum(summary["statuses"].values())==len(cal) and summary["scope_groups"]==cal.group_id.nunique(),"Coverage does not close")
    pairs = pd.read_csv(out/"paired_rgb_inputs.csv")
    eligible = pairs[pairs.paired_eligible]
    require(pairs.sample_id.is_unique and set(pairs.sample_id)==set(cal.sample_id) and len(eligible)==summary["paired_input_frames"],
            "Shared RGB/QLF membership mismatch")
    require(eligible.status.isin(["ok","ok_no_holdout"]).all() and not eligible.dedicated_reference_or_copy.any(),
            "Technical failure or dedicated reference in paired inputs")
    cv2.setNumThreads(1)
    with threadpool_limits(1,user_api="blas"):
        result = dict(frames=len(cal),native_pngs=len(expected_files),synthetic_mapping=synthetic(cfg),
                      spatial_audit=check_tile_ledger(out,cal,qc,cfg),spot_checks=[])
        for state in ("S0","S1","S2"):
            options = cal[cal.state_id.eq(state) & cal.status.eq("ok")]
            if len(options):
                r = options.iloc[0]
                result["spot_checks"].append(replay_frame(r,qc.loc[r.rgb_image_id],out,photos,anchors,annotations,cfg))
            else:
                result["spot_checks"].append(dict(state_id=state,status="no_evaluable_nonreference_frame"))
    if compare is not None:
        result["independent_replay"] = compare_with(out,compare,cal)
    if with_encoder:
        fitted = cal[cal.status.eq("ok")]
        if len(fitted):
            result["frozen_encoder_smoke"] = encoder_smoke(fitted.iloc[0],out,data)
        else:
            result["frozen_encoder_smoke"] = {"status":"no_evaluable_nonreference_frame"}
    dump(out/"validation.json",result)
    return result
