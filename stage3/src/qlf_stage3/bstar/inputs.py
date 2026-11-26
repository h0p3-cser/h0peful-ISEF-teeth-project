"""Freeze the historical cohort and produce native-size sRGB inputs shared by both arms."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path, PurePosixPath
import json
import shutil

import pandas as pd
from PIL import Image, ImageOps

from ..files import dump, load, sha
from ..rgb.color import decode, pixel_hash, save_png
from .color import apply_offset, region_offset

REGIONS = Path(__file__).resolve().parents[1] / 'config' / 'rgb_bstar_regions_v1.json'
KEYS = ['sample_id', 'rgb_image_id', 'qlf_image_id', 'tooth_id', 'state_id', 'angle_id',
        'session_id', 'repeat_rank', 'rgb_event', 'qlf_event', 'group_id', 'group_frames',
        'geometry_version', 'test_fold', 'rgb_path', 'qlf_path', 'rgb_resolved', 'rgb_sha256',
        'rgb_pixel_sha256']
CALIBRATED = 'corrected_bstar'
PASS = 'passthrough_no_reference'
MISSING = 'excluded_missing_source'


def timestamp():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def table(out, name, frame):
    path = out / name
    partial = path.with_suffix('.csv.tmp')
    frame.to_csv(partial, encoding='utf-8-sig', index=False)
    partial.replace(path)


def locate(row, rgb_root):
    parts = PurePosixPath(row.rgb_path.replace(chr(92), '/')).parts
    local = rgb_root.joinpath(*parts[parts.index('Data_rgb_corrected') + 1:])
    return local if local.is_file() or not Path(row.rgb_resolved).is_file() else Path(row.rgb_resolved)


def prepare(out, history, rgb_root, config, checkpoint, jobs):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Choose a new output directory for preparation: {out}')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'rgb').mkdir()
    d = pd.read_csv(history / 'sample_manifest.csv')
    frames = d.loc[d.included.eq(True), KEYS].reset_index(drop=True)
    historical = pd.read_csv(history / 'groups.csv').set_index('group_id', verify_integrity=True)
    review = load(REGIONS)
    groups = review['groups']
    if set(groups) != set(frames.group_id) or set(groups) != set(historical.index):
        raise ValueError('Group membership differs from the frozen white-light review')
    if not (frames.sample_id.is_unique and frames.rgb_image_id.is_unique):
        raise ValueError('Duplicate frame identity in the paired manifest')
    for group_id, block in frames.groupby('group_id'):
        record = groups[group_id]
        basis = block[block.rgb_image_id.eq(record['basis_id'])]
        members = json.loads(historical.at[group_id, 'member_ids'])
        if (len(basis) != 1 or basis.iloc[0].rgb_sha256 != record['basis_sha256'] or
                sorted(block.sample_id) != sorted(members)):
            raise ValueError('Historical source or group mismatch: ' + group_id)
    shutil.copyfile(REGIONS, out / 'reference_regions.json')
    shutil.copyfile(config, out / 'training_config.json')
    shutil.copyfile(history / 'folds.csv', out / 'folds.csv')

    def generate(row):
        path = locate(row, rgb_root)
        annotation = groups[row.group_id]
        record = row._asdict()
        record.update(source_path=str(path.resolve()), review_code=annotation['review_code'],
                      reviewed_reason=annotation['reason'], basis_rgb_id=annotation['basis_id'],
                      regions='[]', valid_reference_pixels=0, estimated_bstar=None,
                      applied_delta_bstar=None, output_path='', output_sha256='', output_pixel_sha256='')
        if not path.is_file():
            return dict(record, status=MISSING, reason='missing_original', paired_included=False)
        if sha(path) != row.rgb_sha256:
            raise ValueError('Unexpected version of historical photograph: ' + str(path))
        raw, info = decode(path)
        source_pixels = pixel_hash(raw)
        legacy_pixels = source_pixels
        if info['icc_sha256']:
            import numpy as np
            with Image.open(path) as source:
                legacy_pixels = pixel_hash(np.asarray(ImageOps.exif_transpose(source).convert('RGB')))
        if legacy_pixels != row.rgb_pixel_sha256:
            raise ValueError('Historical decoded-pixel digest mismatch: ' + row.rgb_image_id)
        regions = annotation.get('regions', [])
        if regions and list(raw.shape[1::-1]) != annotation['native_size']:
            raise ValueError('Native reference geometry differs for ' + row.rgb_image_id)
        median, observations = region_offset(raw, regions)
        result = raw if median is None else apply_offset(raw, median)
        relative = Path('rgb') / (row.rgb_image_id + '.png')
        target = out / relative
        temporary = target.with_suffix('.tmp.png')
        save_png(temporary, result)
        temporary.replace(target)
        reason = annotation['reason']
        if regions and median is None:
            reason = 'empty_valid_reference'
        record.update(info)
        record.update(status=PASS if median is None else CALIBRATED, paired_included=True, reason=reason,
                      source_srgb_pixel_sha256=source_pixels, regions=json.dumps(observations,ensure_ascii=False),
                      valid_reference_pixels=sum(x['valid_pixels'] for x in observations),
                      estimated_bstar=median, applied_delta_bstar=0.0 if median is None else -median,
                      output_path=relative.as_posix(), output_sha256=sha(target),
                      output_pixel_sha256=pixel_hash(result) if median is not None else source_pixels)
        return record

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        records = []
        for number, record in enumerate(pool.map(generate, frames.itertuples(index=False)), 1):
            records.append(record)
            if number % 100 == 0:
                print(f'Native inputs: {number}/{len(frames)}', flush=True)
    manifest = pd.DataFrame(records)
    table(out, 'frame_manifest.csv', manifest)
    coverage = manifest.groupby(['state_id','status']).size().unstack(fill_value=0)
    coverage = coverage.reindex(columns=[CALIBRATED, PASS, MISSING], fill_value=0)
    coverage['paired_frames'] = coverage[CALIBRATED] + coverage[PASS]
    table(out, 'coverage.csv', coverage.reset_index())
    from .review import save_review_sheets
    save_review_sheets(out, manifest)
    dump(out / 'prepared.json', dict(completed=timestamp(), history=str(history), input_review_sha256=sha(REGIONS),
         source_manifest_sha256=sha(history / 'sample_manifest.csv'), groups_sha256=sha(history / 'groups.csv'),
         folds_sha256=sha(out / 'folds.csv'), frame_manifest_sha256=sha(out / 'frame_manifest.csv'),
         training_config_sha256=sha(config), checkpoint=str(checkpoint), source_root=str(rgb_root),
         states=sorted(manifest.state_id.unique().tolist()), frames=len(frames), groups=len(historical),
         paired_frames=int(manifest.paired_included.sum()), teeth=int(frames.tooth_id.nunique())))
    print(coverage.to_string(), flush=True)
