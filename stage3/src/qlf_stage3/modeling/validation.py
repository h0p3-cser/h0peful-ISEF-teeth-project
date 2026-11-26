"""Replay published inputs, fitted fold parameters, scores and paired intervals."""
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .cohort import FEATURES, MODELS, PROBS, STATES, ensure
from .evaluation import read_result
from ..files import FEATURE_CSV, PACKAGE, dataset_info, load, sha


def verify_run(out, data=None, features=None, reference_study=False):
    """Check the archived inputs and fitted objects before replaying every held-out prediction."""
    summary = load(out / 'summary.json')
    ensure(summary['status'] == 'complete' and load(out / 'status.json')['status'] == 'complete',
           'A complete training run is required')
    cfg = load(out / 'run_config.json')
    root, descriptor = dataset_info(data or out / cfg['dataset'])
    csv_path = (features or out / cfg['feature_csv']).resolve()
    ensure(descriptor['kind'] == cfg['dataset_kind'], 'Dataset kind differs from this run')
    for base, key in [(root, 'dataset_hashes'), (csv_path.parent, 'measurement_hashes'),
                      (out / 'source_snapshot', 'code_hashes'), (out / 'models', 'model_hashes')]:
        for relative, digest in summary[key].items():
            target = (base / relative).resolve()
            ensure(target.is_relative_to(base.resolve()) and sha(target) == digest,
                   f'Changed {key}: {relative}')
    result = verify_predictions(out, csv_path)
    result['verified_files'] = {k: len(summary[k]) for k in
                                ('dataset_hashes','measurement_hashes','code_hashes','model_hashes')}
    if reference_study:
        ensure(descriptor['kind'] == 'study', 'The published reference is for the original nine study teeth')
        reference = load(PACKAGE / 'config' / 'study-reference.json')
        for key, expected in reference['counts'].items():
            ensure(result[key] == expected, f'Historical study count differs: {key}')
        primary = read_result(out, 'metrics_summary.csv').query("metric == 'primary_macro_f1'").set_index('model')
        for model, expected in reference['primary_macro_f1'].items():
            np.testing.assert_allclose(primary.at[model,'value'], expected, rtol=0, atol=1e-10)
        result['reference_counts_and_scores_match'] = True
        result['reference_bytes_match'] = {
            name: sha(csv_path if name == FEATURE_CSV else out / name) == digest
            for name, digest in reference['sha256'].items()}
    return result


def verify_predictions(out, feature_csv=None):
    config = load(out / 'run_config.json')
    for name, digest in load(out / 'oof_sha256.json').items():
        ensure(sha(out / name) == digest, 'Sealed OOF file changed: ' + name)
    manifest = read_result(out, 'sample_manifest.csv')
    frames = manifest.loc[manifest.included].set_index('sample_id', verify_integrity=True)
    groups = read_result(out, 'groups.csv').set_index('group_id', verify_integrity=True)
    raw = pd.read_csv(feature_csv or out / config['feature_csv'], float_precision='round_trip').set_index('sample_id', verify_integrity=True)
    np.testing.assert_allclose(frames[FEATURES], raw.loc[frames.index,FEATURES], rtol=0, atol=1e-12)
    with np.load(out / 'frozen_frame_features.npz', allow_pickle=False) as cache:
        np.testing.assert_array_equal(cache['sample_ids'], frames.index)
        np.testing.assert_array_equal(cache['cache_keys'], frames.embedding_cache_key)
        white = pd.DataFrame(cache['white'].astype(np.float64), index=frames.index)
    white_groups = white.groupby(frames.group_id).mean().loc[groups.index]
    qlf_groups = raw.loc[frames.index,FEATURES].groupby(frames.group_id).mean().loc[groups.index]
    np.testing.assert_allclose(groups[FEATURES], qlf_groups, rtol=0, atol=1e-12)
    for key, members in frames.groupby('group_id'):
        ensure(sorted(json.loads(groups.at[key,'member_ids'])) == sorted(members.index)
               and len(members) == groups.at[key,'group_frames'], 'Group membership changed')
    fold_map = read_result(out,'folds.csv')
    oof = read_result(out,'oof_groups.csv').set_index(['model','group_id'], verify_integrity=True)
    oof_frames = read_result(out,'oof_frames.csv').set_index(['model','sample_id'], verify_integrity=True)
    seen = fold_map.query("role == 'test'").set_index('group_id',verify_integrity=True)
    ensure(set(seen.index) == set(groups.index) and (seen.held_tooth == groups.loc[seen.index,'tooth_id']).all()
           and seen.held_tooth.nunique() == 9, 'Invalid held-tooth partition')
    ensure(len(oof) == 3 * len(groups) and len(oof_frames) == 3 * len(frames), 'OOF table size mismatch')
    design = read_result(out,'oof_classifier_inputs.csv').set_index('group_id', verify_integrity=True)
    fits = read_result(out,'fit_audit.csv').set_index(['held_tooth','model'],verify_integrity=True)
    ensure(len(fits) == 3*seen.held_tooth.nunique() and fits.converged.all(), 'Missing or unconverged classifier')
    max_error = 0.0
    with threadpool_limits(limits=1, user_api='blas'):
        for held, fold in fold_map.groupby('held_tooth'):
            train, test = [fold.loc[fold.role == role,'group_id'] for role in ('train','test')]
            expected_train = set(groups.index[groups.tooth_id != held])
            ensure(set(train) == expected_train and train.is_unique, 'Invalid training partition')
            bf = joblib.load(out / 'models' / (held + '.joblib'))
            ensure(set(bf['training_teeth']) == set(groups.loc[train,'tooth_id']) and len(bf['training_teeth'])==8,
                   'Saved model training teeth differ from assigned teeth')
            pca, vs, qs = bf['pca'], bf['white_scaler'], bf['qlf_scaler']
            np.testing.assert_allclose(pca.mean_, white_groups.loc[train].mean(), atol=1e-12, rtol=0)
            projected = pca.transform(white_groups.loc[train].to_numpy())
            for scaler, data in [(vs,projected),(qs,qlf_groups.loc[train].to_numpy())]:
                np.testing.assert_allclose(scaler.mean_, data.mean(axis=0), atol=1e-12, rtol=0)
                np.testing.assert_allclose(scaler.var_, data.var(axis=0), atol=1e-10, rtol=0)
            ids = frames.index[frames.tooth_id == held]
            for is_group, keys, wm, qm, saved in [(True, test, white_groups.loc[test], qlf_groups.loc[test], oof),
                    (False, ids, white.loc[ids], raw.loc[ids,FEATURES], oof_frames)]:
                w = vs.transform(pca.transform(wm.to_numpy()))
                q = qs.transform(qm.to_numpy())
                for model, x in [('M1',w),('M2',q),('M3',np.column_stack([w,q]))]:
                    head = bf['heads'][model]
                    if is_group:
                        rec = fits.loc[held,model]
                        ensure(rec.iterations == head.n_iter_.max() and rec.classifier_input_dim == x.shape[1]
                               and rec.train_groups == len(train), 'Classifier fitting audit mismatch')
                    if is_group:
                        cols = [f'{model}_{i+1}' for i in range(head.n_features_in_)]
                        np.testing.assert_allclose(x, design.loc[keys,cols], rtol=0, atol=1e-10)
                    p = head.predict_proba(x)[:, np.argsort(head.classes_)]
                    rows = saved.loc[model].loc[keys]
                    np.testing.assert_array_equal(np.array(STATES)[p.argmax(axis=1)], rows.pred_state)
                    np.testing.assert_array_equal(rows.held_tooth, np.repeat(held,len(rows)))
                    np.testing.assert_allclose(rows[PROBS], p, rtol=0, atol=1e-10)
                    max_error = max(max_error,float(abs(rows[PROBS].to_numpy()-p).max()))
    # Compute primary scores from confusion counts without calling the reporting code.
    flat = oof.reset_index()
    tooth_scores = {}
    for (tooth, model), block in flat.groupby(['held_tooth','model']):
        cm = pd.crosstab(block.true_state,block.pred_state).reindex(index=STATES,columns=STATES,fill_value=0).to_numpy()
        den = cm.sum(axis=1)+cm.sum(axis=0)
        tooth_scores[tooth,model] = float((2*cm.diagonal()/den).mean()) if (cm.sum(axis=1)>0).all() else np.nan
    scores = pd.Series(tooth_scores).unstack().loc[:,MODELS].dropna()
    reported = read_result(out,'metrics_summary.csv').query("metric == 'primary_macro_f1'").set_index('model')
    draws = np.random.default_rng(config['bootstrap_seed']).integers(
        0,len(scores),size=(config['bootstrap_replicates'],len(scores)))
    for model in MODELS:
        values = scores[model].to_numpy()
        np.testing.assert_allclose([values.mean(), *np.percentile(values[draws].mean(axis=1),[2.5,97.5])],
            reported.loc[model,['value','ci_low','ci_high']].astype(float),rtol=0,atol=1e-12)
    gains = read_result(out,'paired_gains.csv').query("level == 'equal_tooth_mean'").set_index('comparator')
    for comparator in ('M1','M2'):
        delta=(scores.M3-scores[comparator]).to_numpy()*100
        np.testing.assert_allclose([delta.mean(),*np.percentile(delta[draws].mean(axis=1),[2.5,97.5])],
            gains.loc[comparator,['gain_pp','ci_low_pp','ci_high_pp']].astype(float),rtol=0,atol=1e-10)
    return dict(valid=True, held_teeth=len(seen.held_tooth.unique()), paired_frames=len(frames), groups=len(groups),
                oof_group_rows=len(oof), oof_frame_rows=len(oof_frames), converged_heads=len(fits),
                primary_scores_and_paired_intervals_recomputed=True, max_saved_model_probability_difference=max_error)
