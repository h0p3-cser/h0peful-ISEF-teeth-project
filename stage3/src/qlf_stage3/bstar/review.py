"""Recreate the anonymous review sheets from original photographs and frozen review codes."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def save_review_sheets(out, manifest):
    dest = out / 'reference_review'
    dest.mkdir(exist_ok=True)
    groups = list(manifest.sort_values('repeat_rank').groupby('review_code',sort=True))
    for number,(code,frames) in enumerate(groups):
        if number % 16 == 0:
            full = Image.new('RGB',(1280,808),'white')
            full_draw = ImageDraw.Draw(full)
        if number % 18 == 0:
            corners = Image.new('RGB',(1090,954),'#565656')
            corner_draw = ImageDraw.Draw(corners)
        pos = number % 16
        x,y = pos % 4 * 320,pos // 4 * 202
        basis = frames.loc[frames.rgb_image_id.eq(frames.basis_rgb_id)].iloc[0]
        if Path(basis.source_path).is_file():
            with Image.open(basis.source_path) as source:
                source.draft('RGB',(640,360))
                photo = ImageOps.exif_transpose(source).convert('RGB')
                full.paste(ImageOps.contain(photo,(320,180)),(x,y))
        full_draw.text((x+4,y+180),code,fill='black')
        top = number % 18 * 53
        corner_draw.text((1,top+10),code,fill='white')
        for j,row in enumerate(frames.itertuples()):
            x = 51+j*145
            if Path(row.source_path).is_file():
                with Image.open(row.source_path) as source:
                    photo = ImageOps.exif_transpose(source).convert('RGB')
                    w,h = photo.size
                    for k,left in enumerate((0,w-300)):
                        corner = photo.crop((left,0,left+300,min(140,h))).resize((71,39))
                        corners.paste(corner,(x+k*71,top))
            corner_draw.text((x,top+38),row.rgb_image_id,fill='white')
        if pos==15 or number==len(groups)-1:
            full.save(dest/f'field_{number//16+1:02}.jpg',quality=88)
        if number%18==17 or number==len(groups)-1:
            corners.save(dest/f'corners_{number//18+1:02}.jpg',quality=93)
