"""Bind both RGB versions and the four measured QLF features to the frozen capture events."""
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import pandas as pd

from ..files import PACKAGE, dump, load, relative_path, sha
from ..modeling.cohort import FEATURES, IDS, STATES, ensure, save_csv
from ..modeling.evaluation import read_result
from ..modeling.vision import GRAY_MODE, imagenet18, white_vectors
from ..rgb.color import RAW_MODE
from .inputs import CALIBRATED, PASS, locate, timestamp

CONTROL = 'raw_rgb_control'
GRAY_CONTROL = 'same_source_gray_control'
FOLD_KEYS = ['fold','held_tooth','group_id','tooth_id','state_id','role']


def same_folds(actual, expected):
    for table in (actual, expected):
        ensure(not table.duplicated(['fold','group_id']).any(), 'Duplicate group in a fold')
    order = ['fold','group_id']
    a, b = [t[FOLD_KEYS].sort_values(order).reset_index(drop=True) for t in (actual,expected)]
    pd.testing.assert_frame_equal(a,b)


def load_inputs(branch, history, features):
    prepared = load(branch/'prepared.json')
    scores = load(branch/'scored.json')
    config = load(branch/'training_config.json')
    config.setdefault('csv_sha256', load(PACKAGE/'config/endpoint-reference.json')['features_sha256'])
    for path,digest in [(branch/'training_config.json',prepared['training_config_sha256']),
            (branch/'frame_manifest.csv',scores['inputs_sha256']),
            (branch/'reference_regions.json',prepared['input_review_sha256']),
            (branch/'folds.csv',prepared['folds_sha256']), (branch/'oof_groups.csv',scores['prediction_sha256']),
            (history/'sample_manifest.csv',prepared['source_manifest_sha256']),
            (history/'groups.csv',prepared['groups_sha256']), (features,config['csv_sha256'])]:
        ensure(sha(path)==digest, 'Input version differs: ' + str(path))
    ensure(config['states']==STATES and config['feature_columns']==FEATURES, 'Feature or class order differs')
    b = read_result(branch, 'frame_manifest.csv')
    b = b.loc[b.paired_included].copy()
    formal = pd.read_csv(features,float_precision='round_trip')
    ensure(b.sample_id.is_unique and b.rgb_image_id.is_unique and formal.sample_id.is_unique,
           'Duplicate frame ID in calibration or QLF table')
    joined = b.merge(formal[IDS + FEATURES],on=IDS,how='left',validate='one_to_one',indicator=True)
    ensure(joined['_merge'].eq('both').all(), 'A white-light frame lacks an exact paired QLF row')
    joined = joined.drop(columns='_merge')
    ensure(np.isfinite(joined[FEATURES].to_numpy(float)).all() and joined.loss_area_gt30_pct.between(0,100).all(),
           'Invalid QLF measurement')
    orig = read_result(history,'sample_manifest.csv')
    identity = ['sample_id','rgb_image_id','qlf_image_id','group_id','rgb_event','qlf_event','test_fold','rgb_sha256']
    frames = joined.merge(orig.loc[orig.included,identity+['fit_source']],on=identity,how='left',
                          validate='one_to_one',indicator=True)
    ensure(frames['_merge'].eq('both').all(), 'RGB or QLF event differs from the historical pairing')
    frames = frames.drop(columns='_merge').sort_values('sample_id').reset_index(drop=True)
    ensure(frames.status.isin([CALIBRATED,PASS]).all(), 'Unexpected input status in the paired cohort')
    assigned = read_result(branch,'folds.csv')
    same_folds(assigned,read_result(history,'folds.csv'))
    folds = assigned[assigned.group_id.isin(frames.group_id)].copy()
    tooth_map = {tooth:n for n,tooth in enumerate(sorted(frames.tooth_id.unique()),1)}
    ensure(len(tooth_map)==9 and frames.tooth_id.map(tooth_map).eq(frames.test_fold).all(),
           'Frame holdout assignments differ from the nine-tooth protocol')
    group_ids = set(frames.group_id)
    for fold,part in folds.groupby('fold'):
        held = part.held_tooth.unique()
        ensure(len(held)==1 and set(part.group_id)==group_ids and tooth_map[held[0]]==fold and
               (part.role.eq('test') == part.tooth_id.eq(held[0])).all(), 'Invalid historical fold')
    baseline = read_result(branch,'oof_groups.csv').set_index('group_id',verify_integrity=True)
    ensure(set(baseline.index)==group_ids, 'White-only run used different groups')
    for key,block in frames.groupby('group_id'):
        ensure(sorted(block.sample_id)==sorted(json.loads(baseline.at[key,'members'])),
               'White-only run used different members for ' + key)
    return frames, folds, config


def store_variant(directory, frames, raw, processed, config, net):
    directory.mkdir(exist_ok=True)
    policy = config['white_input_policy']
    mode = config['white_light_mode']
    if policy == 'original_srgb':
        paths, pixels = raw, frames.source_srgb_pixel_sha256
    else:
        paths, pixels = processed, frames.output_pixel_sha256
    print(f'{policy}: encoding {len(paths)} full-field frames',flush=True)
    vectors = white_vectors(paths,net,mode)
    keys = [sha256('|'.join([h,mode,config['weights_sha256']]).encode()).hexdigest() for h in pixels]
    manifest = frames.copy()
    manifest['included'] = True
    manifest['rgb_resolved'] = [relative_path(p,directory) for p in paths]
    manifest['raw_rgb_file'] = [relative_path(p,directory) for p in raw]
    manifest['branch_rgb_file'] = [relative_path(p,directory) for p in processed]
    manifest['embedding_cache_key'] = keys
    save_csv(directory,'sample_manifest.csv',manifest)
    np.savez_compressed(directory/'frozen_frame_features.npz',sample_ids=manifest.sample_id.to_numpy(str),
                       cache_keys=np.asarray(keys),white=vectors)
    dump(directory/'run_config.json',config)
    return vectors


def encode(out, branch, history, features, checkpoint, rgb_root):
    out.mkdir(parents=True,exist_ok=True)
    ensure(not any(out.iterdir()), 'Choose an empty output directory')
    frames,folds,locked = load_inputs(branch,history,features)
    net,weights,_ = imagenet18(dict(locked,checkpoint=str(checkpoint)))
    raw = [(locate(r,rgb_root) if rgb_root is not None else Path(r.source_path)) for r in frames.itertuples()]
    processed = [branch/f for f in frames.output_path]
    for path,digest in zip(raw+processed,frames.rgb_sha256.tolist()+frames.output_sha256.tolist()):
        ensure(sha(path)==digest,'White-light image differs: ' + str(path))
    passthrough = frames.status.eq(PASS)
    ensure(frames.loc[passthrough,'output_pixel_sha256'].equals(frames.loc[passthrough,'source_srgb_pixel_sha256']),
           'A recorded passthrough frame changed RGB pixels')
    save_csv(out,'expected_folds.csv',folds)
    checksums = {}
    encoded = []
    for directory,policy in [(out,'bstar_with_passthrough'),(out/CONTROL,'original_srgb')]:
        config = dict(locked, white_light_mode=RAW_MODE, white_input_policy=policy,
                      qlf_workflow='endpoint3_'+policy+'_qlf255_thr30', feature_csv=relative_path(features,directory),
                      source_branch=relative_path(branch,directory), history=relative_path(history,directory),
                      weights_sha256=sha(weights))
        encoded.append(store_variant(directory,frames,raw,processed,config,net))
        for name in ['run_config.json','sample_manifest.csv','frozen_frame_features.npz']:
            path=directory/name
            checksums[path.relative_to(out).as_posix()] = sha(path)
    np.testing.assert_array_equal(encoded[0][passthrough],encoded[1][passthrough])
    checksums['expected_folds.csv'] = sha(out/'expected_folds.csv')
    dump(out/'encoded.json',dict(completed=timestamp(),checksums=checksums,frames=len(frames),
        groups=frames.group_id.nunique(),teeth=frames.tooth_id.nunique(),corrected=int((~passthrough).sum()),
        corrected_groups=frames.loc[~passthrough,'group_id'].nunique(),passthrough=int(passthrough.sum()),
        identical_passthrough_embeddings=True))
    print(f'{len(frames)} paired frames and {frames.group_id.nunique()} groups bound to fixed folds.',flush=True)


def encode_same_source_gray(out, checkpoint):
    base = load(out/'run_config.json')
    directory = out/GRAY_CONTROL
    ready = directory/'inputs.json'
    if ready.is_file():
        saved = load(ready)
        ensure(saved['rgb_inputs_sha256']==sha(out/'encoded.json'),'Sibling-gray cohort differs')
        ensure(all(sha(directory/k)==v for k,v in saved['files'].items()),'Changed sibling-gray input')
        return
    frames = read_result(out,'sample_manifest.csv')
    raw = [(out/p).resolve() for p in frames.raw_rgb_file]
    processed = [(out/p).resolve() for p in frames.branch_rgb_file]
    ensure(all(sha(p)==h for p,h in zip(processed,frames.output_sha256)), 'Changed sibling grayscale source')
    config = dict(base,white_light_mode=GRAY_MODE,white_input_policy='grayscale_of_branch_png',
                  qlf_workflow='endpoint3_bstar_sibling_gray_qlf255_thr30')
    for key in ('feature_csv','history','source_branch'):
        config[key] = relative_path((out/base[key]).resolve(),directory)
    net,_,_ = imagenet18(dict(config,checkpoint=str(checkpoint)))
    store_variant(directory,frames,raw,processed,config,net)
    dump(ready,dict(rgb_inputs_sha256=sha(out/'encoded.json'),files={k:sha(directory/k) for k in
              ('run_config.json','sample_manifest.csv','frozen_frame_features.npz')}))
