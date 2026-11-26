"""Raw-photo identity overlays and a prespecified native crown crop."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..files import dump,load
from ..materials.figures import case_figure,font


COLORS={"notched_disc":"#2fe4d4","outer_bed":"#ffd64a","upper_substrate":"#fc8654","white_backing":"#d18dfb"}


def marked(im,review):
    canvas=Image.fromarray(im).copy()
    draw=ImageDraw.Draw(canvas)
    for region in review["candidates"]:
        c=COLORS.get(region["physical_id"],"#a6d2ff") if region["confirmed"] else "#a1a1a1"
        for p in region["polygons"]:
            draw.line([tuple(q) for q in p+[p[0]]],fill=c,width=3)
    return canvas


def paired_figure(path,left,right,anchor,current):
    canvas=Image.new("RGB",(1000,438),"white")
    draw=ImageDraw.Draw(canvas)
    for i,(im,note,label) in enumerate(((left,anchor,"S0"),(right,current,"Current"))):
        marked_im=marked(im,note)
        x=500*i
        draw.text((x+5,5),f"{label}  {note['rgb_image_id']}",font=font(16),fill="#16292f")
        canvas.paste(marked_im.resize((490,277),Image.Resampling.LANCZOS),(x+5,30))
        for j,box in enumerate([(0,0,245,90),(im.shape[1]-245,0,im.shape[1],90)]):
            canvas.paste(marked_im.crop(box),(x+5+j*245,314))
    draw.text((6,411),"Cyan: notched disc    Yellow: outer bed    Orange: upper strip    Purple: white backing    Grey: identity unconfirmed",
              fill="#203239",font=font(14))
    canvas.save(path,quality=91,subsampling=0)
    return canvas


def visual_review(prepared):
    """Produce evidentiary pairs without loading any corrected image or computing a fit."""
    import pandas as pd
    from .prepare import checked_source,roots
    from .protocol import read_lines
    prepared=Path(prepared).resolve()
    rows=pd.read_csv(prepared/"pilot_manifest.csv",dtype=str,keep_default_na=False).set_index("rgb_image_id",drop=False)
    data=roots(prepared)["data"]
    notes={a["rgb_image_id"]:a for a in read_lines(prepared/"regions.jsonl")}
    out=prepared/"review"
    out.mkdir(exist_ok=True)
    events={}
    targets={}
    indexed={}
    for fid,r in rows.iterrows():
        aid=r.anchor_rgb_id
        if r.input_issue or aid not in rows.index or rows.loc[aid].input_issue:
            continue
        if aid not in targets:
            targets[aid]=checked_source(data,rows.loc[aid])[0]
        raw=checked_source(data,r)[0]
        panel=paired_figure(out/(fid+"-pair.jpg"),targets[aid],raw,notes[aid],notes[fid])
        event=r.rgb_event
        if event not in events:
            events[event]=Image.new("RGB",(3*490,3*220),"white")
        small=panel.resize((490,215),Image.Resampling.LANCZOS)
        rank=int(r.repeat_rank)-1
        events[event].paste(small,((rank%3)*490,(rank//3)*220))
        indexed[fid]=(out/(fid+"-pair.jpg")).relative_to(prepared).as_posix()
    for name,im in events.items():
        im.save(out/(name+"-burst.jpg"),quality=91)
    dump(out/"index.json",indexed)


def crown_case(out,raw,old,new,mask,center,old_clip,new_clip,reason):
    if raw is None:
        canvas=Image.new("RGB",(840,160),"white")
        ImageDraw.Draw(canvas).text((12,55),"I0654 original unavailable: "+reason,fill="black",font=font(20))
        canvas.save(out/"figures/I0654-native-comparison.png")
        return
    can_score=mask is not None
    if center is None:
        center=np.array([1097.,1088.])
    if tuple(np.rint(center).astype(int)-208)!=(889,880):
        raise ValueError("I0654 historical crop drift")
    # A missing native crown mask never contributes a numeric crown measurement.
    if not can_score:
        mask=np.ones(raw.shape[:2],bool)
    case_figure(out,raw,old,new,mask,center,old_clip,new_clip)
    facts=load(out/"I0654-detail.json")
    facts.update(crown_mask_available=can_score,new_unavailable_reason=reason if new is None else "")
    dump(out/"I0654-detail.json",facts)
    if new is None:
        dest=out/"figures/I0654-native-comparison.png"
        with Image.open(dest) as im:
            image=im.copy()
        draw=ImageDraw.Draw(image)
        for y in (230,685):
            draw.text((845,y),reason,fill="#505a60",font=font(15))
        image.save(dest)
