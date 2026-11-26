"""Matched field-of-view QC from the actual native photographs."""
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..rgb.color import decode

CROP = (889,880,1305,1296)
BOXES = {'stripe':(30,172,72,209), 'enamel':(77,270,122,309)}
COLORS = {'stripe':'#1ad8cf','enamel':'#ffb52e'}


def font(size):
    for name in ('DejaVuSans.ttf','arial.ttf'):
        try:
            return ImageFont.truetype(name,size)
        except OSError:
            pass
    return ImageFont.load_default()


def case_figure(out, row, original_gray, output_gray, mask):
    x0,y0,x1,y1 = CROP
    record = dict(image_id=row.rgb_image_id,status=row.status,gray_scale='Pillow L 0-255',
                  crop_xyxy=list(CROP),boxes_in_crop=BOXES)
    panels = []
    for prefix,path,gray in [('original',row.source_path,original_gray),
                             ('branch',out/row.output_path,output_gray)]:
        raw = decode(path)[0]
        gray_crop = gray[y0:y1,x0:x1]
        for label,(left,top,right,bottom) in BOXES.items():
            if mask is not None and not mask[y0+top:y0+bottom,x0+left:x0+right].all():
                raise ValueError('Fixed I0654 sampling box outside the native crown')
            record[prefix+'_'+label+'_median_gray'] = float(np.median(gray_crop[top:bottom,left:right]))
        record[prefix+'_contrast'] = record[prefix+'_enamel_median_gray'] - record[prefix+'_stripe_median_gray']
        panels.append((prefix,Image.fromarray(raw[y0:y1,x0:x1]),Image.fromarray(gray_crop).convert('RGB')))
    record['contrast_change'] = record['branch_contrast'] - record['original_contrast']
    canvas = Image.new('RGB',(892,968),'#fafafa')
    d = ImageDraw.Draw(canvas)
    d.text((18,8),'I0654  |  same native 416 x 416 px field',font=font(24),fill='#172535')
    for column,(label,rgb,gray) in enumerate(panels):
        x = 20+column*438
        for y,tile,title in [(72,rgb,'Original RGB' if label=='original' else 'Branch RGB'),
                             (530,gray,'Original grayscale' if label=='original' else 'Branch grayscale')]:
            d.text((x,y-28),title,font=font(20),fill='#172535')
            canvas.paste(tile,(x,y))
            for name,(left,top,right,bottom) in BOXES.items():
                d.rectangle((x+left-1,y+top-1,x+right,y+bottom),outline=COLORS[name],width=1)
    d.text((20,946),'Cyan: brown stripe   |   Amber: adjacent enamel   |   Grayscale uses the original 0-255 scale.',font=font(15),fill='#253547')
    canvas.save(out/'I0654_same_position.png')
    return record


def reference_figure(out, frames):
    rows = frames[frames.valid_reference_pixels.gt(0)].sort_values('rgb_image_id')
    if rows.empty:
        return
    # This sheet illustrates the actual sampled material, with native coordinates in the manifest.
    page = Image.new('RGB',(900,92+((len(rows)+3)//4)*148),'#fafafa')
    draw = ImageDraw.Draw(page)
    draw.text((16,12),'Neutral reference on the original photographs',font=font(23),fill='#172535')
    draw.text((16,47),'Native upper-right crop. Orange marks the unprinted matte surface used for b*.',font=font(17),fill='#354451')
    for k,row in enumerate(rows.itertuples(index=False)):
        x,y = 14+(k%4)*222,84+(k//4)*148
        obs = json.loads(row.regions)[0]
        x0,y0,x1,y1 = obs['box']
        width,height = int(row.width),int(row.height)
        left,top = max(0,min(x0-100,width-204)), max(0,min(y0-45,height-108))
        with Image.open(row.source_path) as original:
            from PIL import ImageOps
            patch = ImageOps.exif_transpose(original).convert('RGB').crop((left,top,left+204,top+108))
        page.paste(patch,(x,y))
        draw.rectangle((x+x0-left-1,y+y0-top-1,x+x1-left,y+y1-top),outline='#ffb52e',width=2)
        draw.text((x,y+111),f'{row.rgb_image_id}    m = {row.estimated_bstar:.2f}',font=font(16),fill='#172535')
    page.save(out/'reference_samples.png')
