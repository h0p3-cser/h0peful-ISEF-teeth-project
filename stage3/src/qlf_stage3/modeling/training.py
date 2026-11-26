"""Frozen grayscale encoder and train-only, paired 4/4/8-dimensional classifiers."""
import hashlib
import importlib.metadata
import json
import logging
import platform
import shutil
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .scores import tooth_folds
from .cohort import FEATURES, MODELS, PROBS, STATES, build_cohort, ensure, save_csv, table
from .vision import GRAY_MODE, IMAGENET_MEAN, IMAGENET_STD, gray_input, gray_vectors, imagenet18, white_vectors
from ..files import PACKAGE, PROTOCOL, dump, load, outside_inputs, relative_path, sha


def preview(frames, out, data):
    selected = (frames.sort_values(['tooth_id', 'repeat_rank', 'sample_id'])
                .groupby(['state_id', 'angle_id']).head(1).sort_values(['state_id', 'angle_id']))
    columns = min(5, selected.angle_id.nunique())
    sheet = Image.new('RGB', (columns * 224, ((len(selected) + columns - 1) // columns) * 245), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(selected.itertuples()):
        x, y = i % columns * 224, i // columns * 245
        sheet.paste(gray_input(data / row.rgb_resolved), (x, y))
        draw.text((x + 6, y + 225), row.sample_id, fill='black')
    sheet.save(out / 'white_light_preview.png')
    save_csv(out, 'white_light_preview.csv', selected[['sample_id', 'rgb_image_id', 'rgb_path']])


def group_features(frames, visual):
    """Only the paired and eligible members enter both arithmetic means."""
    groups, white, qlf = [], [], []
    cols = ['group_id', 'tooth_id', 'state_id', 'session_id', 'angle_id', 'geometry_version', 'rgb_event', 'qlf_event']
    for group_id, members in frames.groupby('group_id', sort=True):
        members = members.sort_values('sample_id')
        block = visual[members.index].astype(np.float64)
        center = block.mean(axis=0)
        features = members[FEATURES].to_numpy(float).mean(axis=0)
        representative = members.iloc[np.linalg.norm(block - center, axis=1).argmin()]
        row = members.iloc[0][cols].to_dict()
        row.update(group_frames=len(members), member_ids=json.dumps(members.sample_id.tolist()),
                   representative_id=representative.sample_id, representative_rgb_id=representative.rgb_image_id,
                   representative_rgb_path=representative.rgb_resolved, **dict(zip(FEATURES, features)))
        groups.append(row)
        white.append(center)
        qlf.append(features)
    return pd.DataFrame(groups), np.stack(white), np.stack(qlf)


def matrices(bundle, white=None, qlf=None):
    result = {}
    if white is not None:
        result['M1'] = bundle['white_scaler'].transform(bundle['pca'].transform(white))
    if qlf is not None:
        result['M2'] = bundle['qlf_scaler'].transform(qlf)
    if white is not None and qlf is not None:
        result['M3'] = np.column_stack([result['M1'], result['M2']])
    return result


def fit_models(c, white, qlf, y):
    seed = c.get('training_seed')
    pca = PCA(n_components=c['pca_components'], svd_solver='full', whiten=False, random_state=seed).fit(white)
    bundle = dict(pca=pca, white_scaler=StandardScaler().fit(pca.transform(white)),
                  qlf_scaler=StandardScaler().fit(qlf), heads={}, states=STATES, feature_columns=FEATURES,
                  white_light_mode=c.get('white_light_mode', GRAY_MODE), weights_sha256=c['weights_sha256'], workflow=c['qlf_workflow'])
    classifier = dict(c['classifier'])
    if seed is not None:
        classifier['random_state'] = seed
        bundle['training_seed'] = seed
    xs = matrices(bundle, white, qlf)
    ensure(pca.n_samples_ == len(y) and all(bundle[k].n_samples_seen_ == len(y)
           for k in ('white_scaler', 'qlf_scaler')), 'Fold preprocessing used the wrong number of groups')
    for model, dims in zip(MODELS, (4, 4, 8)):
        head = LogisticRegression(penalty='l2', **classifier)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            head.fit(xs[model], y)
        ensure(head.n_features_in_ == dims and set(head.classes_) == {0, 1, 2},
               'Classifier inputs or label encoding differ from the protocol')
        bundle['heads'][model] = head
    return bundle


def predict_matrix(bundle, model, x):
    head = bundle['heads'][model]
    prob = head.predict_proba(x)[:, [list(head.classes_).index(i) for i in range(3)]]
    ensure(np.isfinite(prob).all() and (prob >= 0).all() and (prob <= 1).all(), 'Invalid probability')
    np.testing.assert_allclose(prob.sum(axis=1), 1, atol=1e-10, rtol=0)
    result = pd.DataFrame(prob, columns=PROBS)
    result.insert(0, 'pred_state', np.asarray(STATES)[prob.argmax(axis=1)])
    result.insert(0, 'model', model)
    return result


def held_predictions(bundle, subset, white, qlf):
    cols = ['fold', 'held_tooth', 'group_id', 'angle_id', 'session_id', 'state_id', 'group_frames']
    cols.extend(k for k in ('sample_id', 'rgb_image_id', 'repeat_rank', 'rgb_path', 'qlf_path',
                            'representative_id') if k in subset)
    identity = subset[cols].reset_index(drop=True).rename(columns={'state_id': 'true_state'})
    xs = matrices(bundle, white, qlf)
    predictions = [pd.concat([identity, predict_matrix(bundle, model, xs[model])], axis=1) for model in MODELS]
    return pd.concat(predictions, ignore_index=True), xs


def fold_predictions(c, out, groups, white, qlf, frames, visual):
    splits = tooth_folds(groups, STATES)
    group_oof, frame_oof, audit, assignments, design = [], [], [], [], []
    y = groups.state_id.map(STATES.index).to_numpy()
    ensure(len(splits) == 9 and len(set(i for _, _, test in splits for i in test)) == len(groups),
           'Expected nine disjoint held-tooth folds')
    models_dir = out / 'models'
    models_dir.mkdir(exist_ok=True)
    for fold, (held, train, test) in enumerate(splits, 1):
        train_teeth = sorted(groups.iloc[train].tooth_id.unique())
        ensure(len(train_teeth) == 8 and held not in train_teeth and groups.iloc[test].tooth_id.eq(held).all(),
               'Train and test tooth membership is invalid')
        bundle = fit_models(c, white[train], qlf[train], y[train])
        bundle['training_teeth'] = train_teeth
        bundle['held_tooth'] = held
        joblib.dump(bundle, models_dir / f'{held}.joblib', compress=3)
        gr = groups.iloc[test].assign(fold=fold, held_tooth=held)
        fr = frames.loc[frames.tooth_id == held].assign(fold=fold, held_tooth=held)
        row, xs = held_predictions(bundle, gr, white[test], qlf[test])
        group_oof.append(row)
        row, _ = held_predictions(bundle, fr, visual[fr.index].astype(float), fr[FEATURES].to_numpy(float))
        frame_oof.append(row)
        raw = gr[['fold', 'held_tooth', 'group_id']].reset_index(drop=True)
        for model, width in zip(MODELS, (4, 4, 8)):
            design_block = pd.DataFrame(xs[model], columns=[f'{model}_{i+1}' for i in range(width)])
            raw = pd.concat([raw, design_block], axis=1)
            head = bundle['heads'][model]
            audit.append(dict(fold=fold, held_tooth=held, model=model, train_teeth='|'.join(train_teeth),
                              train_groups=len(train), test_groups=len(test), classifier_input_dim=head.n_features_in_,
                              iterations=int(head.n_iter_.max()), converged=True, pca_fitted_groups=bundle['pca'].n_samples_,
                              **{f'pca_variance_ratio_{i+1}': v for i, v in enumerate(bundle['pca'].explained_variance_ratio_)}))
        np.testing.assert_array_equal(xs['M1'], xs['M3'][:, :4])
        np.testing.assert_array_equal(xs['M2'], xs['M3'][:, 4:])
        design.append(raw)
        for role, indices in [('train', train), ('test', test)]:
            assignments.append(groups.iloc[indices][['group_id', 'tooth_id', 'state_id']].assign(
                fold=fold, held_tooth=held, role=role))
        logging.getLogger(__name__).info(
            "Fold %d/%d held %s, train/test groups %d/%d, converged M1/M2/M3 iterations %s",
            fold, len(splits), held, len(train), len(test),
            "/".join(str(bundle['heads'][model].n_iter_.max()) for model in MODELS))
    for name, blocks, key, expected in [('groups', group_oof, 'group_id', len(groups)),
                                         ('frames', frame_oof, 'sample_id', len(frames))]:
        data = pd.concat(blocks, ignore_index=True)
        ensure(len(data) == 3 * expected and not data.duplicated(['model', key]).any()
               and data.groupby('model')[key].nunique().eq(expected).all(), 'Incomplete OOF ' + name)
        save_csv(out, f'oof_{name}.csv', data)
    save_csv(out, 'folds.csv', pd.concat(assignments))
    save_csv(out, 'fit_audit.csv', pd.DataFrame(audit))
    save_csv(out, 'oof_classifier_inputs.csv', pd.concat(design))


def run_experiment(c, out):
    from .evaluation import evaluate
    out = outside_inputs(out, c['feature_csv'].parent, c['dataset'])
    out.mkdir(parents=True, exist_ok=False)
    dump(out / 'status.json', {'status': 'running'})
    try:
        net, weights, url = imagenet18(c)
        run_config = dict(**load(PROTOCOL), dataset=relative_path(c['dataset'], out), dataset_kind=c['kind'],
                          feature_csv=relative_path(c['feature_csv'], out), checkpoint_name=weights.name, weight_source=url,
                          mean=IMAGENET_MEAN, std=IMAGENET_STD, device='cpu', torch_threads=4, cpu_blas_threads=1,
                          embedding='frozen 512-D global average pool', scaling='per training group, ddof=0',
                          classifier_objective='multinomial cross-entropy with L2 penalty; one unit per acquisition group',
                          pca='centered full SVD without whitening, then StandardScaler',
                          crop_reason='The available reviewed crown masks use QLF; whole-frame white light is locked.',
                          groups='tooth, session, angle, geometry, RGB event and QLF event; equal group weights',
                          decision='ordered classes S0=0,S1=1,S2=2; argmax with first exact tie',
                          bootstrap_unit='paired teeth sampled with replacement from fixed out-of-fold scores',
                          versions={k: importlib.metadata.version(k) for k in
                                    ['torch','torchvision','scikit-learn','numpy','pandas','Pillow','matplotlib','joblib','scipy']},
                          python=platform.python_version())
        dump(out / 'run_config.json', run_config)
        manifest, hashes, verification = build_cohort(c)
        hashes[str(weights)] = sha(weights)
        frames = manifest.loc[manifest.included].reset_index(drop=True)
        preview(frames, out, c['dataset'])
        print(f'Validated {len(frames)} paired frames; extracting frozen white-light vectors.', flush=True)
        visual = gray_vectors((c['dataset'] / path for path in frames.rgb_resolved), net)
        keys = [hashlib.sha256('|'.join([h, GRAY_MODE, c['weights'], c['weights_sha256']]).encode()).hexdigest()
                for h in frames.rgb_sha256]
        manifest['embedding_cache_key'] = manifest.sample_id.map(dict(zip(frames.sample_id, keys))).fillna('')
        np.savez_compressed(out / 'frozen_frame_features.npz', sample_ids=frames.sample_id.to_numpy(str),
                            cache_keys=np.asarray(keys), white=visual)
        groups, white, qlf = group_features(frames, visual)
        fold_for = {t: i for i, t in enumerate(sorted(groups.tooth_id.unique()), 1)}
        manifest['test_fold'] = manifest.tooth_id.map(fold_for)
        save_csv(out, 'sample_manifest.csv', manifest)
        save_csv(out, 'groups.csv', groups)
        print(f'{len(groups)} acquisition groups. Fitting the nine held-tooth folds.', flush=True)
        with threadpool_limits(limits=1, user_api='blas'):
            fold_predictions(c, out, groups, white, qlf, frames, visual)
        sealed = {n: sha(out / n) for n in ['oof_groups.csv', 'oof_frames.csv']}
        dump(out / 'oof_sha256.json', sealed)
        results = evaluate(c, out)
        from .validation import verify_predictions
        dump(out / 'validation.json', verify_predictions(out))
        from .report import make_report
        make_report(c, out, results, verification)
        with threadpool_limits(limits=1, user_api='blas'):
            final = fit_models(c, white, qlf, groups.state_id.map(STATES.index).to_numpy())
        final['training_teeth'] = sorted(fold_for)
        final['scope'] = 'all-group inference only; evaluation uses sealed held-tooth predictions'
        joblib.dump(final, out / 'models' / 'final_bundle.joblib', compress=3)
        ensure(all(sha(out / name) == digest for name, digest in sealed.items()), 'OOF changed after sealing')
        ensure(all(sha(path) == digest for path, digest in hashes.items()), 'An experiment input changed during run')
        filenames = sorted([*PACKAGE.rglob('*.py'), *PACKAGE.glob('config/*.json'), *PACKAGE.glob('config/*.csv')])
        code_hashes = {}
        for source in filenames:
            name = source.relative_to(PACKAGE).as_posix()
            target = out / 'source_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            code_hashes[name] = sha(target)
        dataset_hashes = {Path(path).relative_to(c['dataset']).as_posix(): digest for path,digest in hashes.items()
                          if Path(path).is_relative_to(c['dataset'])}
        measurement_hashes = {Path(path).name: digest for path,digest in hashes.items()
                              if Path(path).parent == c['feature_csv'].parent}
        model_hashes = {path.name: sha(path) for path in sorted((out / 'models').glob('*.joblib'))}
        summary = dict(status='complete', workflow=c['qlf_workflow'], pairs=len(frames), observations=len(groups),
                       split='tooth', teeth=len(fold_for), metrics=results['summary'].to_dict('records'),
                       dataset_hashes=dataset_hashes, measurement_hashes=measurement_hashes,
                       code_hashes=code_hashes, model_hashes=model_hashes)
        dump(out / 'summary.json', summary)
        dump(out / 'status.json', {'status':'complete', 'oof_sha256':sealed})
        return summary
    except Exception as exc:
        dump(out / 'status.json', dict(status='failed', error=str(exc)))
        raise


def predict_acquisition(c, model_path, input_path, out, arm='all'):
    """Predict unlabeled paired capture(s). One capture_id denotes one real same-view acquisition."""
    bundle = joblib.load(model_path)
    workflows = {c['qlf_workflow']}
    if c['qlf_workflow'] == load(PROTOCOL)['qlf_workflow']:
        from ..rgb.color import RAW_MODE
        mode = bundle.get('white_light_mode', GRAY_MODE)
        if mode in (GRAY_MODE, RAW_MODE):
            branch = 'gray' if mode == GRAY_MODE else 'rgb'
            workflows.add(f'endpoint3_{branch}_255_thr30')
    ensure(bundle.get('workflow') in workflows and bundle['feature_columns'] == FEATURES
           and bundle['weights_sha256'] == c['weights_sha256'], 'Incompatible endpoint 3 model')
    data = table(input_path)
    if 'capture_id' not in data and 'group_id' in data:
        data = data.rename(columns={'group_id':'capture_id'})
    ensure(len(data) and 'capture_id' in data and data.capture_id.str.strip().ne('').all(),
           'Provide a nonempty capture_id for each frame')
    for field in ('tooth_id', 'session_id', 'angle_id', 'geometry_version', 'rgb_event', 'qlf_event'):
        if field in data:
            ensure(data.groupby('capture_id')[field].nunique().le(1).all(), f'Mixed {field} in a capture group')
    use_white, use_qlf = arm != 'M2', arm != 'M1'
    if use_white:
        base = Path(input_path).resolve().parent
        if 'dataset' in c and 'rgb_resolved' in data:
            paths = [c['dataset'] / path for path in data.rgb_resolved]
        else:
            ensure('rgb_path' in data and data.rgb_path.ne('').all(), 'Provide a white-light rgb_path for each frame')
            paths = [(base / path).resolve() for path in data.rgb_path]
        # Both representations use the same locked ImageNet encoder and normalization.
        net, _, _ = imagenet18(dict(c, white_light_mode=GRAY_MODE))
        visual = white_vectors(paths, net, bundle.get('white_light_mode', GRAY_MODE)).astype(float)
    if use_qlf:
        ensure(set(FEATURES).issubset(data), 'Provide all four locked QLF feature columns')
        xq = data[FEATURES].to_numpy(float)
        ensure(np.isfinite(xq).all() and ((xq[:, 1] >= 0) & (xq[:, 1] <= 100)).all(), 'Invalid QLF values')
    rows = []
    for key, block in data.groupby('capture_id', sort=True):
        idx = block.index
        xs = matrices(bundle, visual[idx].mean(axis=0, keepdims=True) if use_white else None,
                      xq[idx].mean(axis=0, keepdims=True) if use_qlf else None)
        for model in (MODELS if arm == 'all' else [arm]):
            rows.append(predict_matrix(bundle, model, xs[model]).assign(capture_id=key, group_frames=len(idx)))
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    predictions = pd.concat(rows, ignore_index=True)
    save_csv(out, 'predictions.csv', predictions)
    return dict(predictions=len(predictions), fit_performed=False, output=str(out / 'predictions.csv'))


def compare_runs(first, second, tolerance):
    from .evaluation import read_result
    differences = {}
    for name, key in [('oof_groups.csv', 'group_id'), ('oof_frames.csv', 'sample_id')]:
        a, b = [read_result(folder, name).set_index(['model', key]).sort_index() for folder in (first, second)]
        pd.testing.assert_frame_equal(a.drop(columns=PROBS), b.drop(columns=PROBS))
        np.testing.assert_allclose(a[PROBS], b[PROBS], rtol=0, atol=tolerance)
        differences[name] = float(np.abs(a[PROBS] - b[PROBS]).to_numpy().max())
    pd.testing.assert_frame_equal(read_result(first, 'folds.csv'), read_result(second, 'folds.csv'))
    return dict(equal_categories=True, max_probability_differences=differences, tolerance=tolerance)
