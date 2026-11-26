"""Frozen whole-frame encoders, event means and paired held-tooth predictions."""
import json
import platform
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ..files import dump, load, sha
from ..modeling.vision import GRAY_MODE, imagenet18, white_vectors
from ..rgb.color import RAW_MODE
from .inputs import CALIBRATED, table, timestamp

ARMS = {'rgb': RAW_MODE, 'gray': GRAY_MODE}


def summarize(oof, states):
    teeth = []
    for tooth, block in oof.groupby('tooth_id', sort=True):
        row = dict(tooth_id=tooth, groups=len(block), s1_groups=int(block.state_id.eq('S1').sum()))
        for arm in ARMS:
            true, pred = block.state_id, block[arm + '_pred']
            row[arm + '_macro_f1'] = f1_score(true, pred, labels=states, average='macro', zero_division=0)
            s1 = true.eq('S1')
            row[arm + '_s1_recall'] = pred[s1].eq('S1').mean()
        row['rgb_minus_gray_macro_f1'] = row['rgb_macro_f1'] - row['gray_macro_f1']
        teeth.append(row)
    teeth = pd.DataFrame(teeth)
    scores, confusions = [], []
    for arm in ARMS:
        cm = confusion_matrix(oof.state_id, oof[arm + '_pred'], labels=states)
        for i, truth in enumerate(states):
            for j, prediction in enumerate(states):
                total = int(cm[i].sum())
                confusions.append(dict(arm=arm, true_state=truth, pred_state=prediction,
                    groups=int(cm[i, j]), true_groups=total,
                    pct_within_true_state=100 * cm[i,j] / total if total else None))
        i = states.index('S1')
        scores.append(dict(arm=arm, tooth_equal_macro_f1=teeth[arm + '_macro_f1'].mean(),
            s1_correct=int(cm[i,i]), s1_groups=int(cm[i].sum()), s1_recall=cm[i,i]/cm[i].sum(),
            tooth_equal_s1_recall=teeth[arm + '_s1_recall'].mean(),
            correct=int(cm.trace()), groups=len(oof), teeth=len(teeth)))
    return pd.DataFrame(scores), teeth, pd.DataFrame(confusions)


def score(out, checkpoint):
    if (out / 'scored.json').exists():
        raise FileExistsError('Scores are already frozen in this run')
    prepared = load(out / 'prepared.json')
    for file, key in [('frame_manifest.csv','frame_manifest_sha256'), ('folds.csv','folds_sha256'),
                      ('training_config.json','training_config_sha256'), ('reference_regions.json','input_review_sha256')]:
        if sha(out / file) != prepared[key]:
            raise ValueError('Changed experiment input: ' + file)
    frames = pd.read_csv(out / 'frame_manifest.csv')
    frames = frames.loc[frames.paired_included].sort_values('sample_id').reset_index(drop=True)
    paths = [out / rel for rel in frames.output_path]
    if any(sha(path) != expected for path, expected in zip(paths, frames.output_sha256)):
        raise ValueError('A prepared white-light PNG has changed')
    config = load(out / 'training_config.json')
    states = config['states']
    config['checkpoint'] = str(checkpoint)
    net, weights, _ = imagenet18(config)
    assignments = pd.read_csv(out / 'folds.csv')
    retained = set(frames.group_id)
    assignments = assignments[assignments.group_id.isin(retained)]
    grouped, indices = [], []
    fixed = ['group_id', 'tooth_id', 'state_id', 'angle_id', 'session_id', 'rgb_event', 'qlf_event', 'test_fold']
    for _, block in frames.groupby('group_id', sort=True):
        if (block[fixed].nunique() != 1).any():
            raise ValueError('Mixed event, outcome, or fold in a capture group')
        row = block.iloc[0][fixed].to_dict()
        grouped.append(dict(row, frames=len(block), corrected_frames=int(block.status.eq(CALIBRATED).sum()),
                            rgb_members=json.dumps(block.rgb_image_id.tolist()), members=json.dumps(block.sample_id.tolist())))
        indices.append(block.index.to_numpy())
    oof = pd.DataFrame(grouped)
    if oof.tooth_id.nunique() != 9 or assignments.fold.nunique() != 9:
        raise ValueError('The specified nine-tooth experiment requires nine held-out teeth')
    fold_plan = []
    for fold, part in assignments.groupby('fold',sort=True):
        held = part.held_tooth.unique()
        test = oof.test_fold.eq(fold).to_numpy()
        reference = part.set_index('group_id', verify_integrity=True).reindex(oof.group_id)
        if (len(held) != 1 or not test.any() or set(oof.loc[test,'tooth_id']) != {held[0]} or
                oof.loc[~test,'tooth_id'].nunique() != 8 or held[0] in set(oof.loc[~test,'tooth_id']) or
                not np.array_equal(reference.role.to_numpy(), np.where(test, 'test', 'train')) or
                not np.array_equal(reference.tooth_id.to_numpy(), oof.tooth_id) or
                not np.array_equal(reference.state_id.to_numpy(), oof.state_id)):
            raise ValueError(f'Historical fold {fold} differs from the paired cohort')
        fold_plan.append((int(fold), held[0], test))
    (out / 'fold_models').mkdir(exist_ok=True)
    audit = []
    y = np.array([states.index(state) for state in oof.state_id])
    for arm, read_mode in ARMS.items():
        print(f'Extracting {arm}: {len(paths)} branch PNGs', flush=True)
        vectors = white_vectors(paths, net, read_mode)
        white = np.stack([vectors[i].astype(np.float64).mean(axis=0) for i in indices])
        oof_prob = np.full((len(oof), len(states)), np.nan)
        with threadpool_limits(limits=1, user_api='blas'):
            for fold, held_tooth, test in fold_plan:
                training = white[~test]
                pca = PCA(n_components=config['pca_components'], svd_solver='full', whiten=False).fit(training)
                projected = pca.transform(training)
                scaler = StandardScaler().fit(projected)
                head = LogisticRegression(penalty='l2', **config['classifier'])
                with warnings.catch_warnings():
                    warnings.simplefilter('error', ConvergenceWarning)
                    head.fit(scaler.transform(projected), y[~test])
                if (pca.n_samples_ != len(training) or scaler.n_samples_seen_ != len(training) or
                        not np.array_equal(head.classes_, np.arange(len(states)))):
                    raise ValueError('Unexpected train-fold transform or label order')
                oof_prob[test] = head.predict_proba(scaler.transform(pca.transform(white[test])))
                joblib.dump(dict(pca=pca, scaler=scaler, head=head, states=states, arm=arm, held_tooth=held_tooth),
                            out / 'fold_models' / f'{arm}_{held_tooth}.joblib')
                audit.append(dict(arm=arm, fold=fold, held_tooth=held_tooth, train_teeth=8,
                    train_groups=pca.n_samples_, test_groups=int(test.sum()), dimensions=head.n_features_in_,
                    training_teeth='|'.join(sorted(oof.loc[~test,'tooth_id'].unique()))))
        if not np.isfinite(oof_prob).all() or not np.allclose(oof_prob.sum(axis=1),1,rtol=0,atol=1e-10):
            raise ValueError('Incomplete out-of-fold probabilities')
        for j, label in enumerate(states):
            oof[f'{arm}_p_{label}'] = oof_prob[:,j]
        oof[arm+'_pred'] = np.asarray(states)[oof_prob.argmax(axis=1)]
    scores, teeth, confusion = summarize(oof, states)
    table(out,'oof_groups.csv',oof)
    table(out,'fit_audit.csv',pd.DataFrame(audit))
    table(out,'scores.csv',scores)
    table(out,'scores_by_tooth.csv',teeth)
    table(out,'confusion.csv',confusion)
    from .report import write_scores
    write_scores(out, scores, teeth, confusion)
    import PIL, sklearn, torch, torchvision
    dump(out / 'scored.json', dict(completed=timestamp(), inputs_sha256=sha(out/'frame_manifest.csv'),
        weights=str(weights), weights_sha256=sha(weights), classes=states, arms=list(ARMS),
        prediction_sha256=sha(out/'oof_groups.csv'), scores_sha256=sha(out/'scores.csv'),
        report_sha256=sha(out/'model_scores.md'), versions=dict(python=platform.python_version(),
        sklearn=sklearn.__version__, pillow=PIL.__version__, torch=torch.__version__,
        torchvision=torchvision.__version__, numpy=np.__version__, pandas=pd.__version__)))
    print(scores.to_string(index=False), flush=True)
