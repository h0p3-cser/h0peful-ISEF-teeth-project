"""Compare the uploaded T005 white-light JPEGs with camera-verified C04 S2 peers."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image, JpegImagePlugin
from skimage.color import rgb2lab

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE/'src'))
from qlf_stage3.files import dump,sha
from qlf_stage3.modeling.cohort import ensure,save_csv
from qlf_stage3.rgb.color import decode
from qlf_stage3.rgb.geometry import white_body
from qlf_stage3.rgb.source_review import report


def color_stats(encoded):
    if not len(encoded):
        return {}
    rgb=encoded.astype(float)
    lab=rgb2lab(rgb/255.0,illuminant='D65',observer='2')
    def summarize(x,y):
        return dict(Lstar=float(np.median(y[:,0])),astar=float(np.median(y[:,1])),
                    bstar=float(np.median(y[:,2])),R_over_G=float(np.median(x[:,0]/x[:,1])),
                    B_over_G=float(np.median(x[:,2]/x[:,1])))
    result=summarize(rgb,lab)
    keep=(lab[:,0]>=45)&(lab[:,0]<=65)
    result['midtone_pixels']=int(keep.sum())
    if keep.any():
        result.update({'midtone_'+k:v for k,v in summarize(rgb[keep],lab[keep]).items()})
    return result


def image_record(row,source,mask_path):
    digest=sha(source)
    if row.tooth_id!='T005':
        ensure(digest==row.source_camera_sha256,'Comparison photograph differs from camera source: '+row.image_id)
    with Image.open(source) as im:
        ensure(im.format=='JPEG' and im.mode=='RGB','Expected a colour JPEG: '+row.image_id)
        exif=im.getexif()
        sub=exif.get_ifd(34665)
        encoded=dict(jpeg_sampling={0:'4:4:4',1:'4:2:2',2:'4:2:0'}.get(JpegImagePlugin.get_sampling(im),'unknown'),
            quant_table_sha256=sha256(json.dumps(im.quantization,sort_keys=True).encode()).hexdigest(),
            exif_camera=exif.get(272),exif_white_balance=sub.get(41987),exif_exposure_s=float(sub.get(33434,np.nan)),
            exif_iso=sub.get(34855),capture_datetime=sub.get(36867),jpeg_bytes=source.stat().st_size)
    raw,info=decode(source)
    mask,method=white_body(mask_path,info,raw)
    # An interior core removes the tooth outline and its immediate dark rim. No chroma is used for region selection.
    core=cv2.erode(mask.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
    pixels=raw[core]
    valid=((pixels>5)&(pixels<250)).all(axis=1)
    ensure(valid.any(),'No interior tooth pixels: '+row.image_id)
    features=color_stats(pixels[valid])
    h,w=raw.shape[:2]
    corners=np.concatenate([raw[round(h*.04):round(h*.12),round(w*x):round(w*(x+.08))].reshape(-1,3)
                              for x in (.04,.88)])
    encoded_corner=np.median(corners,axis=0)
    return dict(rgb_image_id=row.image_id,tooth_id=row.tooth_id,angle_id=row.angle_id,
        camera_sha256=row.source_camera_sha256,observed_sha256=digest,camera_match=digest==row.source_camera_sha256,
        delivered_match=digest==row.sha256,source_path=str(source),mask_method=method,
        core_pixels=int(core.sum()),valid_pixels=int(valid.sum()),core_fraction=core.mean(),
        background_R=encoded_corner[0],background_G=encoded_corner[1],background_B=encoded_corner[2],
        **encoded,**features)


def compare(out,folder,archive,masks):
    out.mkdir(exist_ok=True,parents=True)
    intake=archive/'02_metadata/collection_v18_S2_C04_20261124/image_manifest.csv'
    catalog=pd.read_csv(intake,keep_default_na=False,dtype=str)
    rows=catalog[(catalog.session_id=='C04')&(catalog.state_id=='S2')&(catalog.modality=='RGB')].sort_values(
        ['tooth_id','angle_id','image_id'])
    ensure(rows.tooth_id.nunique()==9 and len(rows)==315,'Unexpected formal C04 RGB cohort')
    records=[]
    cv2.setNumThreads(1)
    for i,row in enumerate(rows.itertuples(index=False),1):
        name=Path(row.normalized_relative_path).name
        source=folder/row.angle_id/'RGB'/name if row.tooth_id=='T005' else archive/row.original_backup_relative_path
        mask_path=masks/row.tooth_id/row.angle_id/'RGB'/(Path(name).stem+'.png')
        records.append(image_record(row,source,mask_path))
        if i%75==0: print(f'Measured {i}/{len(rows)} C04 RGB photographs',flush=True)
    frames=pd.DataFrame(records)
    frames['first_exposure_in_view']=~frames.duplicated(['tooth_id','angle_id','camera_sha256'])
    unique=frames[frames.first_exposure_in_view]
    color_cols=['Lstar','astar','bstar','R_over_G','B_over_G','midtone_Lstar','midtone_astar','midtone_bstar',
                'midtone_R_over_G','midtone_B_over_G','background_R','background_G','background_B']
    angles=unique.groupby(['tooth_id','angle_id'],as_index=False)[color_cols].median()
    angles['unique_exposures']=unique.groupby(['tooth_id','angle_id'],sort=True).size().to_numpy()
    contrasts=[]
    for angle,block in angles.groupby('angle_id'):
        focal=block[block.tooth_id.eq('T005')].iloc[0]
        peers=block[~block.tooth_id.eq('T005')]
        for metric in color_cols:
            vals=peers[metric].dropna()
            if len(vals) and np.isfinite(focal[metric]):
                contrasts.append(dict(angle_id=angle,metric=metric,T005=focal[metric],controls=len(vals),
                    control_median=vals.median(),control_min=vals.min(),control_max=vals.max(),
                    control_teeth_lower=int(vals.lt(focal[metric]).sum()),
                    outside_control_range=bool(focal[metric]<vals.min() or focal[metric]>vals.max())))
    contrast=pd.DataFrame(contrasts)
    save_csv(out,'photo_features.csv',frames)
    save_csv(out,'view_features.csv',angles)
    save_csv(out,'same_view_comparison.csv',contrast)
    dump(out/'design.json',dict(stage='S2',session='C04',source_folder=str(folder),archive_path=str(archive),
        masks_path=str(masks),intake_sha256=sha(intake),
        spots=len(frames),unique_exposures=len(unique),angles=5,control_teeth=8,
        subject='T005 uploaded JPEGs',controls='eight other teeth, each file verified against its camera SHA-256',
        body='existing native white-body mask and its project grayscale foreground validation, eroded by 3 native pixels',
        body_valid='all channels 6..249',midtone='45 <= D65/2 degree L* <= 65',
        background='two native upper-corner rectangles x=4..12% and 88..96%, y=4..12%, raw 8-bit channel medians',
        aggregation='per-frame pixel medians; one vote per distinct camera exposure per tooth/view; median across those exposures; compare T005 against eight teeth at same angle',
        color_scale='CIE L*,a*,b* D65 2deg; R/G and B/G ratios in 8-bit encoded sRGB'))
    print(contrast[contrast.metric.isin(['Lstar','bstar','midtone_bstar'])].to_string(index=False),flush=True)
    report(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True,help='Original acquisition archive with its C04 intake manifest')
    parser.add_argument('--masks',type=Path,default=PACKAGE/'data/study/workspace/Data/数据处理代码/masks/03_较重度脱矿_2代')
    args=parser.parse_args()
    compare(args.out.resolve(),args.folder.resolve(),args.archive.resolve(),args.masks.resolve())
