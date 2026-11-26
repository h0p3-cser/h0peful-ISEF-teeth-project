"""Bind the locked Stage 3 cohort to its actual photos and paired acquisition events."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from ..files import SOURCE_TABLE, load, sha


STATES = ['S0', 'S1', 'S2']
MODELS = ['M1', 'M2', 'M3']
FEATURES = ['huber_delta_f_pct', 'loss_area_gt30_pct', 'p10_intensity', 'G_texture_entropy']
IDS = ['sample_id', 'tooth_id', 'state_id', 'angle_id', 'repeat_rank', 'qlf_path', 'rgb_path']
PROBS = ['p_' + s for s in STATES]


def table(path):
    return pd.read_csv(path, keep_default_na=False, dtype=str)


def index_by(path, key):
    return table(path).set_index(key, verify_integrity=True).to_dict('index')


def save_csv(out, name, frame):
    frame.to_csv(out / name, index=False, encoding='utf-8-sig', lineterminator='\n')


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def photo_digest(path):
    """Count identical decoded photographs even when JPEG metadata/encodings differ."""
    with Image.open(path) as image:
        pixels = ImageOps.exif_transpose(image).convert('RGB')
    digest = hashlib.sha256(str(pixels.size).encode('ascii'))
    digest.update(pixels.tobytes())
    return digest.hexdigest()


def event_key(meta, digest):
    return (meta['modality'], meta['tooth_id'], meta['session_id'], meta['angle_id'],
            meta['geometry_version'], digest)


def resolve_photo(path, binding, metadata, catalog):
    # Restore a moved image only if its complete bytes equal the recorded local binding.
    errors = []
    for candidate in dict.fromkeys([path, catalog / metadata['normalized_relative_path']]):
        if candidate.is_file():
            if sha(candidate) == binding['sha256']:
                return candidate.resolve()
            errors.append(str(candidate))
    raise OSError('missing or changed photo: ' + str(path) + (f' (hash mismatch: {errors})' if errors else ''))


def build_cohort(c):
    csv_path, workspace, catalog = c['feature_csv'], c['author_workspace'], c['capture_catalog']
    base = workspace / 'Data'
    upstream = workspace / SOURCE_TABLE
    frames_path = csv_path.with_name('frames.csv')
    verification_path = csv_path.with_name('training-features-255-thr30-verification.json')
    required = [csv_path, verification_path, frames_path, upstream, workspace / 'local-photo-bindings.csv',
                catalog / 'images.csv', catalog / 'pairs.csv', c['_path']]
    hashes = {str(p): sha(p) for p in required}
    verification = load(verification_path)
    ensure(hashes[str(csv_path)] == verification['csv_sha256'], 'Feature CSV differs from its verified export')
    ensure(hashes[str(upstream)] == verification['source_table_sha256'], 'Reviewed Stage 3 pairing table changed')
    data = table(csv_path)
    ensure(data.columns.tolist() == IDS + FEATURES and c['feature_columns'] == FEATURES and c['states'] == STATES,
           'Unexpected formal feature schema or stage order')
    ensure(all(data[k].is_unique and data[k].ne('').all() for k in ['sample_id', 'qlf_path', 'rgb_path']),
           'Formal image or pairing ID is empty or repeated')
    ensure(data.state_id.value_counts().to_dict() == verification['exported_by_stage'], 'Stage counts changed')
    ensure(data.tooth_id.nunique() == 9, 'The locked cohort must contain nine teeth')
    data[FEATURES] = data[FEATURES].astype(float)
    ensure(np.isfinite(data[FEATURES]).all().all() and data.loss_area_gt30_pct.between(0, 100).all(),
           'Invalid QLF feature values')
    frames = index_by(frames_path, 'qlf_path')
    reviewed = index_by(upstream, 'image_path')
    bindings = index_by(workspace / 'local-photo-bindings.csv', 'workspace_path')
    images = index_by(catalog / 'images.csv', 'image_id')
    old_pairs = index_by(catalog / 'pairs.csv', 'qlf_image_id')
    events = defaultdict(set)
    for binding in bindings.values():
        meta = images[binding['image_id']]
        if meta['capture_event_id']:
            events[event_key(meta, binding['sha256'])].add(meta['capture_event_id'])
    pixel_hashes, records = {}, []
    for row in data.to_dict('records'):
        source, frame = reviewed[row['qlf_path']], frames[row['qlf_path']]
        ensure(all(row[k] == source[k] for k in ['tooth_id', 'state_id', 'angle_id', 'repeat_rank', 'rgb_path']),
               f"Pair does not agree with reviewed Stage 3: {row['sample_id']}")
        ensure(frame['status'] == 'ok' and Path(row['qlf_path']).stem == row['sample_id'] and
               frame['anchor_path'] == source['s0_image_path'], 'Frame/anchor ID disagrees')
        anchor = frames[frame['anchor_path']]
        ensure((anchor['state_id'], anchor['tooth_id'], anchor['angle_id']) ==
               ('S0', row['tooth_id'], row['angle_id']), 'Cross-tooth or cross-view calibration')
        row.update(session_id=source['batch_id'], anchor_path=frame['anchor_path'], fit_source=frame['fit_source'],
                   included=True, exclusion_reason='')
        problems, geometries = [], set()
        for mod in ('rgb', 'qlf'):
            path = (base / row[mod + '_path']).resolve()
            binding = bindings[path.relative_to(workspace).as_posix()]
            image_id, digest = binding['image_id'], binding['sha256']
            meta = images[image_id]
            ensure(path.stem.rsplit('_', 1)[-1] == image_id and meta['modality'] == mod.upper() and
                   meta['state_id'] == row['state_id'] and meta['angle_id'] == row['angle_id'] and
                   meta['session_id'] == row['session_id'] and
                   row['tooth_id'] in (meta['tooth_id'], meta['previous_formal_tooth_id']),
                   f'Conflicting capture metadata for {image_id}')
            geometries.add(meta['geometry_version'])
            event = meta['capture_event_id']
            origin = 'recorded'
            if not event:
                matches = events[event_key(meta, digest)]
                event = next(iter(matches)) if len(matches) == 1 else ''
                origin = 'same_photo_sha256' if event else 'unresolved'
                if not event:
                    problems.append('unresolved ' + mod + ' capture event')
            row.update({mod + '_image_id': image_id, mod + '_sha256': digest, mod + '_event': event,
                        mod + '_event_source': origin, mod + '_catalog_tooth': meta['tooth_id']})
            try:
                actual = resolve_photo(path, binding, meta, catalog)
                if digest not in pixel_hashes:
                    pixel_hashes[digest] = photo_digest(actual)
                hashes[str(actual)] = digest
                row.update({mod + '_resolved': actual.relative_to(c['dataset']).as_posix(),
                            mod + '_pixel_sha256': pixel_hashes[digest]})
            except (OSError, ValueError) as exc:
                problems.append(str(exc))
        ensure(len(geometries) == 1 and '' not in geometries, 'Missing or mixed capture geometry')
        row['geometry_version'] = geometries.pop()
        old = old_pairs[row['qlf_image_id']]
        row.update(catalog_pair_id=old['pair_id'], catalog_paired_rgb=old['rgb_image_id'],
                   same_catalog_rgb_pair=old['rgb_image_id'] == row['rgb_image_id'])
        row['group_id'] = ''
        if problems:
            row.update(included=False, exclusion_reason=' | '.join(problems))
        else:
            key = [row[k] for k in ('tooth_id', 'session_id', 'angle_id', 'geometry_version', 'rgb_event', 'qlf_event')]
            suffix = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:12]
            row['group_id'] = f"{row['tooth_id']}-{row['angle_id']}-{suffix}"
        records.append(row)
    manifest = pd.DataFrame(records)
    included = manifest.loc[manifest.included]
    all_pixels = pd.concat([included[['tooth_id', m + '_pixel_sha256']].set_axis(['tooth_id', 'pixel'], axis=1)
                            for m in ('rgb', 'qlf')], ignore_index=True)
    cross = all_pixels.groupby('pixel').tooth_id.nunique()
    if (cross > 1).any():
        bad = set(cross[cross > 1].index)
        affected = included.rgb_pixel_sha256.isin(bad) | included.qlf_pixel_sha256.isin(bad)
        manifest.loc[included.index[affected], ['included', 'exclusion_reason']] = [False, 'cross-tooth identical pixels']
    included = manifest.loc[manifest.included]
    ensure(included.tooth_id.nunique() == 9 and len(included), 'Common cohort lost a whole tooth')
    groups = included.groupby('group_id')
    ensure(groups.state_id.nunique().eq(1).all() and groups.tooth_id.nunique().eq(1).all()
           and not included.duplicated(['group_id', 'repeat_rank']).any(), 'Conflicting group labels or repeat IDs')
    manifest['group_frames'] = manifest.group_id.map(groups.size()).fillna(0).astype(int)
    manifest['repeat_rank'] = manifest.repeat_rank.astype(int)
    bundle = workspace / 'stage3-author-transforms.zip'
    if bundle.is_file():
        hashes[str(bundle)] = sha(bundle)
        with ZipFile(bundle) as archive, archive.open('paired-photo-sha256.csv') as stream:
            ledger = pd.read_csv(stream, encoding='utf-8-sig').set_index('image_path', verify_integrity=True)
        expected = ledger.loc[manifest.qlf_path]
        ensure(expected.rgb_path.tolist() == manifest.rgb_path.tolist(), 'Reviewed white-light pairing changed')
        manifest['author_rgb_hash_status'] = np.where(manifest.rgb_sha256 == expected.rgb_sha256.to_numpy(),
                                                       'matched_in_replay', 'local_differs')
    else:
        manifest['author_rgb_hash_status'] = 'unverified'
    return manifest, hashes, verification


def inspect_cohort(c):
    manifest, _, _ = build_cohort(c)
    active = manifest[manifest.included]
    return dict(ready=True, formal_frames=len(manifest), selected_pairs=len(active),
                teeth=sorted(active.tooth_id.unique()), groups=active.group_id.nunique(),
                excluded_pairs=len(manifest) - len(active), states=active.state_id.value_counts().to_dict())
