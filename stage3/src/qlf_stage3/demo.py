"""Generate small original synthetic observations with known, identity registration."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from .bundle import write_checksums
from .files import SOURCE_TABLE, dump, sha
from .pixels import intensity


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_demo(out):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.stage3-demo-', dir=out.parent) as temp:
        base = Path(temp) / 'dataset'
        work = base / 'workspace'
        data = work / 'Data'
        sem = Path('数据处理代码/语义掩膜_背景与基线黑斑')
        (data / sem).mkdir(parents=True)
        yy, xx = np.indices((220, 256))
        body = ((xx-128)/119)**2 + ((yy-112)/100)**2 < 1
        crown = body & (yy <= 96)
        Image.fromarray(body.astype('uint8')).save(data / sem / 'body.png')
        Image.fromarray(crown.astype('uint8')).save(data / sem / 'crown.png')
        rng = np.random.default_rng(7582)
        views, geometry, registered, identities, reviewed, bindings, images, pairs = [], {}, [], [], [], [], [], []
        for i in range(9):
            tooth, angle = f'DEMO{i+1:02}', 'A000'
            views.append((tooth, angle))
            geometry[tooth+'_'+angle] = dict(crown_boundary=[[0,108],[255,108]], crown_side_point=[128,20])
            for state_no,state in enumerate(('S0','S1','S2')):
                session = ('C01','C02','C04')[state_no]
                for rank in (1,2):
                    base_id = 12*i + 4*state_no + 2*(rank-1)
                    frame = dict(tooth_id=tooth,state_id=state,angle_id=angle,repeat_rank=rank,batch_id=session)
                    paths = {}
                    for offset,mod in enumerate(('QLF','RGB'), 1):
                        image_id = f'I{base_id+offset:05}'
                        name = f'{tooth}_{state}_{angle}_{mod}_{image_id}.png'
                        relative = Path(tooth) / state / mod / name
                        path = data / relative
                        path.parent.mkdir(exist_ok=True,parents=True)
                        noise = rng.normal(0, 2 + state_no, xx.shape)
                        if mod == 'QLF':
                            signal = 55 + .54*xx + .13*yy + i + noise
                            signal += crown * (45 - 30*state_no + 5*np.sin((xx+i)/9))
                        else:
                            signal = 130 + 28*np.sin((xx+i*3)/(8+2*state_no)) + .18*yy - 18*state_no + noise
                        green = np.uint8(np.clip(np.where(body, signal+rank, 7+i), 0, 254))
                        if mod == 'QLF':
                            photo = np.stack([green//2, green, green//3], axis=2)
                        else:
                            # An actual colour field exercises both white-light decoders in the public demo.
                            red = green.astype(float) + body * (11*np.sin(xx/25+i) + 4*state_no)
                            blue = green.astype(float) + body * (9*np.cos(yy/19+i) - 3*state_no)
                            photo = np.stack([red, green, blue], axis=2).clip(0,255).astype('uint8')
                        Image.fromarray(photo).save(path)
                        paths[mod] = (relative.as_posix(),image_id)
                        identities.append(dict(frame, modality=mod, file_path=relative.as_posix(),is_duplicate=0,
                                               dose_hours=(0, 24, 72)[state_no], mask_path=(sem/'body.png').as_posix()))
                        bindings.append(dict(workspace_path='Data/'+relative.as_posix(),modality=mod,
                                             image_id=image_id,sha256=sha(path)))
                        images.append(dict(image_id=image_id,modality=mod,tooth_id=tooth,previous_formal_tooth_id=tooth,
                                           state_id=state,angle_id=angle,session_id=session,geometry_version='synthetic_v1',
                                           capture_event_id=f'{tooth}-{session}-{angle}-{mod}',normalized_relative_path=relative.as_posix(),
                                           sha256=bindings[-1]['sha256']))
                        if mod == 'QLF':
                            statistics = intensity(green, crown)
                    qpath, qid = paths['QLF']
                    rgb, rid = paths['RGB']
                    if state_no == 0 and rank == 1:
                        anchor = qpath
                    registered.append(dict(path=qpath, s0_anchor=anchor, label_path='body.png',
                                           transform=[[1,0,0],[0,1,0]], status='anchor' if qpath == anchor else 'registered'))
                    reviewed.append(dict(frame, image_path=qpath,rgb_path=rgb,s0_image_path=anchor,
                        source_tooth_id=tooth, crown_semantic_mask_path=(sem/'crown.png').as_posix(),
                        crown_mask_status='registered', roi_pixels=int(crown.sum()),loss_status='automatic_qc_pass',**statistics))
                    pairs.append(dict(pair_id=f'{tooth}-{state}-{rank}',qlf_image_id=qid,rgb_image_id=rid))
        dump(work/'student-views.json',dict(schema='qlf_stage3.views.v1',views=views,repeats_per_view=2))
        dump(work/'crown_rois.json',geometry)
        (data/sem/'local-index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in registered),encoding='utf-8')
        for path, rows in [(data/'manifest.csv',identities),(work/SOURCE_TABLE,reviewed),(work/'local-photo-bindings.csv',bindings),
                           (base/'catalog'/'images.csv',images),(base/'catalog'/'pairs.csv',pairs)]:
            write_rows(path,rows)
        dump(base/'dataset.json',dict(schema=1,kind='synthetic',seed=7582))
        write_checksums(base)
        base.rename(out)
    return out
