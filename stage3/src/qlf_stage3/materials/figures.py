"""Native-scale illustrations, coordinates and contrast for the prespecified I0654 case."""
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.color import rgb2lab

from ..files import dump
from ..rgb.color import save_png


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf",size)
    except OSError:
        return ImageFont.load_default(size=size)


def field_overlay(raw,patches,fid,note):
    thumb=Image.fromarray(raw).resize((672,380),Image.Resampling.LANCZOS)
    canvas=Image.new("RGB",(672,425),"white")
    canvas.paste(thumb)
    d=ImageDraw.Draw(canvas)
    colors={"fit":"#ff9340","holdout":"#2fd6d3"}
    for name,p in patches.items():
        pt=np.asarray(p.polygon)*[672/raw.shape[1],380/raw.shape[0]]
        ctr=pt.mean(axis=0)
        d.line([tuple(x) for x in np.concatenate([pt,pt[:1]])],fill=colors[p.role],width=2)
        if name.endswith("_1"):
            d.text((int(np.clip(ctr[0],3,561)),int(np.clip(ctr[1]+3,3,350))),p.material,fill=colors[p.role],font=font(13))
    d.text((6,382),fid+"   "+note,fill="#222222",font=font(13))
    d.text((6,401),"Orange = fit blocks    Turquoise = independently held-out disc",fill="#303030",font=font(13))
    return canvas


def reference_detail(out, sources):
    """Inspect the subtle upper substrate in raw native-coordinate subwindows."""
    windows=[(0,0,450,124),(2442,0,2688,124)]
    canvas=Image.new("RGB",(920,470),"white")
    d=ImageDraw.Draw(canvas)
    for row,(fid,raw,patches) in enumerate(sources):
        y=row*217
        d.text((8,y+3),fid+"   upper substrate / raw sRGB",fill="#17252b",font=font(18))
        for j,(x0,y0,x1,y1) in enumerate(windows):
            crop=Image.fromarray(raw[y0:y1,x0:x1].copy())
            reg=ImageDraw.Draw(crop)
            for name,p in patches.items():
                if p.material=="upper_substrate":
                    polys=np.asarray(p.polygon)-[x0,y0]
                    if (polys[:,0]>=0).all() and (polys[:,0]<crop.width).all():
                        reg.polygon([tuple(q) for q in polys],outline="#ff9340",width=1)
            box=(j*458+8,y+35)
            canvas.paste(crop,box)
            d.text((box[0],y+166),f"Native x [{x0},{x1})    Native y [0,124)",fill="#34474b",font=font(15))
    d.text((8,440),"Orange borders show the sampled interiors. The bright corner glint is outside them.",fill="#17252b",font=font(16))
    canvas.save(out/"figures/I0654-upper-substrate.png")


def case_figure(out,raw,old,new,mask,center,old_clip,new_clip):
    # Coordinates are fixed on the native uncorrected case, never chosen on corrected colours.
    half=208
    x,y=np.rint(center).astype(int)
    x0,y0=x-half,y-half
    tile=lambda a:a[y0:y+half,x0:x+half]
    # Manually reviewed on the raw native crop. Both boxes are included in the coordinate record.
    # The x/y values here are **crop** coordinates.
    rois={"stripe":[30,172,72,209],"adjacent_enamel":[77,270,122,309]}
    colors=[("Raw I0654",raw),("Root Huber",old),("Full-field Huber",new)]
    image=Image.new("RGB",(3*416,2*416+108),"white")
    d=ImageDraw.Draw(image)
    measures={}
    for idx,(title,im) in enumerate(colors):
        left=idx*416
        d.text((left+12,8),title,fill="#17252b",font=font(20))
        if im is None:
            for top in (33,495):
                d.text((left+18,top+160),"Unavailable: no calibrated image",fill="#444444",font=font(15))
            continue
        detail=tile(im).copy()
        png=out/"figures"/("I0654-"+("raw","root","material")[idx]+"-416.png")
        save_png(png,detail)
        image.paste(Image.fromarray(detail),(left,33))
        tinted=detail.copy()
        clip=(None,old_clip,new_clip)[idx]
        if clip is not None:
            keep=tile(clip)&tile(mask)
            tinted[keep]=np.rint(.30*tinted[keep]+.70*np.array([242,62,173])).astype(np.uint8)
        image.paste(Image.fromarray(tinted),(left,496))
        for name,(a,b,c,e) in rois.items():
            d.rectangle((left+a,496+b,left+c-1,496+e-1),outline="#11e6e1" if name=="stripe" else "#ffbb39",width=2)
        vals={}
        for name,(a,b,c,e) in rois.items():
            area=detail[b:e,a:c]
            usable=tile(mask)[b:e,a:c]
            if not usable.all():
                raise ValueError("One of the fixed stripe comparison boxes leaves the native crown ROI")
            lab=rgb2lab(area/255.,illuminant="D65",observer="2")[...,0]
            vals[name+"_median_L"]=float(np.median(lab))
            vals[name+"_median_RGB"]=np.median(area,axis=(0,1)).tolist()
            vals[name+"_L_iqr"]=float(np.ptp(np.percentile(lab,[25,75])))
        vals["enamel_minus_stripe_L"]=vals["adjacent_enamel_median_L"]-vals["stripe_median_L"]
        measures[("raw","root","material")[idx]]=vals
        d.text((left+12,458),f"Enamel minus stripe L*: {vals['enamel_minus_stripe_L']:.2f}",fill="#17252b",font=font(16))
    d.text((12,915),"Native 416 x 416. Magenta = new out-of-range crown pixels. Cyan = stripe. Amber = adjacent enamel.",fill="#17252b",font=font(15))
    path=out/"figures/I0654-native-comparison.png"
    image.save(path)
    result=dict(rgb_image_id="I0654",native_xyxy=[int(x0),int(y0),int(x+half),int(y+half)],
                crop_xyxy_half_open=rois,source_grid="oriented native pixels; 1 display pixel per source pixel in each PNG",
                measured=measures)
    dump(out/"I0654-detail.json",result)
    return result
