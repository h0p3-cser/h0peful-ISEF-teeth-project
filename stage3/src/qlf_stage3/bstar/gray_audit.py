"""Native-coordinate grayscale measurements, opened only after saving both OOF results."""
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
from skimage.color import rgb2lab

from ..files import dump, load, sha
from ..materials.study import historical_crown
from ..modeling.vision import native_gray
from ..rgb.color import decode
from .inputs import CALIBRATED, table, timestamp


def histogram(values, mask=None):
    mask8 = None if mask is None else mask.astype(np.uint8)
    return cv2.calcHist([values], [0], mask8, [256], [0,256]).ravel().astype(np.int64)


def difference_stats(hist):
    n = int(hist.sum())
    if not n:
        return {}
    cumulative = hist.cumsum()
    position = (n - 1) * 0.95
    lo, hi = np.floor(position), np.ceil(position)
    a, b = cumulative.searchsorted([int(lo)+1, int(hi)+1])
    return dict(pixels=n, mean_abs=float(hist @ np.arange(256) / n),
                p95_abs=float(a + (position-lo) * (int(b)-int(a))),
                max_abs=int(np.flatnonzero(hist)[-1]))


class CrownArchive:
    """Use accepted historical registration even when its subsequent color fit failed."""
    def __init__(self, root):
        self.available = all((root / f).is_file() for f in ('rgb_calibration.csv','anchors.csv','qc_by_frame.csv'))
        if self.available:
            self.records = pd.read_csv(root/'rgb_calibration.csv').set_index('rgb_image_id',verify_integrity=True)
            self.anchors = pd.read_csv(root/'anchors.csv').set_index('anchor_rgb_id',verify_integrity=True)
            self.qc = pd.read_csv(root/'qc_by_frame.csv').set_index('rgb_image_id',verify_integrity=True)

    def native_mask(self, row, data):
        idx = row.rgb_image_id
        if not self.available or idx not in self.records.index:
            return None, 'no_archived_white_crown'
        old = self.records.loc[idx]
        if old.expected_source_sha256 != row.rgb_sha256 or old.sample_id != row.sample_id:
            raise ValueError('Historical ROI source differs: ' + idx)
        if not isinstance(old.reference_to_current_work, str):
            return None, 'no_accepted_white_registration'
        if old.anchor_rgb_id not in self.anchors.index:
            return None, 'no_archived_crown_anchor'
        anchor = self.anchors.loc[old.anchor_rgb_id]
        if not (anchor.status == 'ready' and anchor.anchor_sha256 == old.anchor_sha256 and
                anchor.tooth_id == row.tooth_id and anchor.angle_id == row.angle_id):
            raise ValueError('Historical crown anchor differs: ' + idx)
        mask_path = data / old.mask_path
        if not mask_path.is_file():
            return None, 'missing_archived_white_mask'
        if sha(mask_path) != old.mask_sha256:
            raise ValueError('Archived native mask differs: ' + idx)
        rgb, info = decode(row.source_path)
        mask, _ = historical_crown(data, old, anchor, rgb, info)
        if mask is None or not mask.any():
            return None, 'empty_archived_crown'
        recorded = self.qc.at[idx,'crown_pixels']
        if pd.notna(recorded) and int(mask.sum()) != int(recorded):
            raise ValueError('Archived crown pixel count differs: ' + idx)
        return mask, ('verified_historical_pixel_count' if pd.notna(recorded)
                      else 'accepted_historical_geometry')


def reference_measurements(out, frames):
    """Check the intended offset on exactly the sampled pixels of the delivered PNG."""
    records = []
    for row in frames[frames.status.eq(CALIBRATED)].itertuples():
        raw = decode(row.source_path)[0]
        saved = decode(out/row.output_path,corrected=True)[0]
        for region in json.loads(row.regions):
            x0,y0,x1,y1 = region['box']
            before, after = (image[y0:y1,x0:x1].reshape(-1,3) for image in (raw,saved))
            valid = ((before > 5) & (before < 250)).all(axis=1)
            if not valid.any():
                continue
            medians = [np.median(rgb2lab(image[valid],illuminant='D65',observer='2')[:,2])
                       for image in (before,after)]
            records.append(dict(rgb_image_id=row.rgb_image_id, material=region['material'],
                valid_pixels=int(valid.sum()), original_median_bstar=medians[0],
                png_median_bstar=medians[1], applied_delta_bstar=row.applied_delta_bstar))
    return pd.DataFrame(records,columns=['rgb_image_id','material','valid_pixels',
                        'original_median_bstar','png_median_bstar','applied_delta_bstar'])


def audit(out, root, data):
    from .figures import case_figure, reference_figure
    from .qc_report import write_audit
    scored = load(out / 'scored.json')
    for name, key in [('model_scores.md','report_sha256'), ('scores.csv','scores_sha256'),
                      ('oof_groups.csv','prediction_sha256'), ('frame_manifest.csv','inputs_sha256')]:
        if sha(out/name) != scored[key]:
            raise ValueError('Frozen model results or cohort changed: ' + name)
    frames = pd.read_csv(out/'frame_manifest.csv')
    frames = frames.loc[frames.paired_included].reset_index(drop=True)
    archive = CrownArchive(root)
    cv2.setNumThreads(1)
    measures, pooled, case = [], {}, None
    for i, row in enumerate(frames.itertuples(index=False),1):
        source, output = Path(row.source_path), out/row.output_path
        if sha(source) != row.rgb_sha256 or sha(output) != row.output_sha256:
            raise ValueError('Native QC image version changed: ' + row.rgb_image_id)
        original_gray = np.asarray(native_gray(source))
        output_gray = np.asarray(native_gray(output))
        if original_gray.shape != output_gray.shape or output_gray.shape != (row.height,row.width):
            raise ValueError('Native geometry changed: ' + row.rgb_image_id)
        delta = cv2.absdiff(original_gray, output_gray)
        mask, reason = archive.native_mask(row,data)
        record = dict(rgb_image_id=row.rgb_image_id, sample_id=row.sample_id, tooth_id=row.tooth_id,
                      state_id=row.state_id, group_id=row.group_id, status=row.status, crown_status=reason)
        for region, selected in [('whole',None),('crown',mask)]:
            if region == 'crown' and selected is None:
                continue
            counts = histogram(delta, selected)
            stats = difference_stats(counts)
            record.update({f'{region}_{name}': val for name,val in stats.items()})
            key = row.status, region
            if key not in pooled:
                pooled[key] = np.zeros(256,dtype=np.int64)
            pooled[key] += counts
        if row.rgb_image_id == 'I0654':
            case = case_figure(out,row,original_gray,output_gray,mask)
        measures.append(record)
        if i % 150 == 0:
            print(f'Native grayscale QC: {i}/{len(frames)}',flush=True)
    measured = pd.DataFrame(measures)
    table(out,'gray_by_frame.csv',measured)
    summaries = []
    for status, block in measured.groupby('status',sort=True):
        for region in ['whole','crown']:
            key = status,region
            if key not in pooled:
                summaries.append(dict(status=status,region=region,available_frames=0,total_frames=len(block)))
                continue
            keep = block.dropna(subset=[region+'_pixels'])
            stats = difference_stats(pooled[key])
            summaries.append(dict(status=status,region=region,available_frames=len(keep),total_frames=len(block),
                frame_mean_mae=keep[region+'_mean_abs'].mean(), frame_median_mae=keep[region+'_mean_abs'].median(),
                min_frame_mae=keep[region+'_mean_abs'].min(),max_frame_mae=keep[region+'_mean_abs'].max(),
                frame_median_p95=keep[region+'_p95_abs'].median(),min_frame_p95=keep[region+'_p95_abs'].min(),
                max_frame_p95=keep[region+'_p95_abs'].max(),pooled_mean_abs=stats['mean_abs'],
                pooled_pixel_p95=stats['p95_abs'],pooled_max_abs=stats['max_abs'],pixels=stats['pixels']))
    summary = pd.DataFrame(summaries)
    table(out,'gray_summary.csv',summary)
    if case is not None:
        dump(out/'I0654_contrast.json',case)
    reference = reference_measurements(out,frames)
    table(out,'reference_bstar.csv',reference)
    reference_figure(out,frames)
    write_audit(out,measured,summary,case,reference)
    dump(out/'complete.json',dict(completed=timestamp(),scores_saved_at=scored['completed'],
        scores_sha256=sha(out/'scores.csv'),model_report_sha256=sha(out/'model_scores.md'),
        final_report_sha256=sha(out/'report.md'),gray_by_frame_sha256=sha(out/'gray_by_frame.csv'),
        native_frames=len(measured),crown_frames=int(measured.get('crown_pixels',pd.Series(dtype=float)).notna().sum()),
        archive=str(root),mask_data=str(data)))
    print(summary[['status','region','available_frames','frame_mean_mae','frame_median_p95']].to_string(index=False),flush=True)
