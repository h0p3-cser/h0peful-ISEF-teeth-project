"""All performance measures start from the sealed out-of-fold tables on disk."""
import json
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_curve

from .scores import metrics
from .cohort import FEATURES, MODELS, PROBS, STATES, ensure, save_csv
from ..files import dump, sha


def read_result(out, name):
    return pd.read_csv(out / name, float_precision='round_trip', keep_default_na=False)


def tooth_intervals(c, scores):
    available = scores.dropna(how='any').loc[:, MODELS]
    ensure(len(available) > 0, 'No tooth has all three stages')
    draws = np.random.default_rng(c['bootstrap_seed']).integers(
        len(available), size=(c['bootstrap_replicates'], len(available)))
    def interval(values):
        means = np.asarray(values)[draws].mean(axis=1)
        return np.quantile(means, [.025, .975])
    bounds = {model: interval(available[model]) for model in MODELS}
    gains = []
    for comparator in ['M1', 'M2']:
        differences = 100 * (scores.M3 - scores[comparator])
        for tooth, value in differences.items():
            gains.append(dict(comparator=comparator, tooth_id=tooth, gain_pp=value, level='tooth'))
        matched = differences.loc[available.index]
        lo, hi = interval(matched)
        gains.append(dict(comparator=comparator, tooth_id='ALL', gain_pp=matched.mean(), level='equal_tooth_mean',
                          ci_low_pp=lo, ci_high_pp=hi, n_teeth=len(matched),
                          improved=int((matched > 1e-10).sum()), tied=int((matched.abs() <= 1e-10).sum()),
                          declined=int((matched < -1e-10).sum())))
    return bounds, pd.DataFrame(gains), len(available)


def paired_cases(out, groups, oof):
    paired = groups.copy()
    for model in MODELS:
        block = oof.loc[oof.model == model, ['group_id', 'true_state', 'held_tooth', 'fold', 'pred_state', *PROBS]]
        block = block.rename(columns={k: f'{model}_{k}' for k in ['pred_state', *PROBS]})
        keys = ['group_id', 'true_state', 'held_tooth', 'fold']
        if model == MODELS[0]:
            paired = paired.assign(true_state=paired.state_id, held_tooth=paired.tooth_id)
            keys.remove('fold')
        paired = paired.merge(block, on=keys, validate='one_to_one', how='inner')
    ensure(len(paired) == len(groups), 'OOF alignment between models changed')
    counts, cases = [], []
    for comparator in ['M1', 'M2']:
        correct_single = paired[comparator + '_pred_state'] == paired.state_id
        correct_fusion = paired.M3_pred_state == paired.state_id
        col = 'change_vs_' + comparator
        paired[col] = np.select([correct_single & correct_fusion, ~correct_single & correct_fusion,
                                correct_single & ~correct_fusion], ['both_correct', 'corrected', 'spoiled'],
                               default='both_wrong')
        for state, direction in product(STATES, ['corrected', 'spoiled']):
            subset = paired[(paired.state_id == state) & (paired[col] == direction)]
            counts.append(dict(comparator=comparator, true_state=state, direction=direction,
                               groups=len(subset), true_stage_groups=int((paired.state_id == state).sum())))
        for direction, wrong_model in [('corrected', comparator), ('spoiled', 'M3')]:
            subset = paired.loc[paired[col] == direction].copy()
            if subset.empty:
                continue
            pred = subset[wrong_model + '_pred_state']
            subset['error_distance'] = (pred.map(STATES.index) - subset.state_id.map(STATES.index)).abs()
            subset['wrong_class_p'] = subset[[wrong_model + '_' + k for k in PROBS]].max(axis=1)
            picked = subset.sort_values(['error_distance', 'wrong_class_p', 'group_id'],
                                        ascending=[False, False, True]).iloc[0].to_dict()
            cases.append(dict(comparator=comparator, direction=direction, wrong_model=wrong_model, **picked))
    counts = pd.DataFrame(counts)
    cases = pd.DataFrame(cases, columns=['comparator','direction','wrong_model','error_distance','wrong_class_p',
                                        *[k for k in paired if not k.startswith('change_vs_')]])
    save_csv(out, 'paired_group_predictions.csv', paired)
    save_csv(out, 'changes_by_stage.csv', counts)
    save_csv(out, 'representative_cases.csv', cases)
    errors = oof.loc[oof.pred_state != oof.true_state].copy()
    errors['wrong_class_p'] = errors[PROBS].max(axis=1)
    errors = errors.merge(groups[['group_id', 'representative_id', 'representative_rgb_path', *FEATURES]],
                          on='group_id', validate='many_to_one')
    errors = errors.sort_values(['model', 'wrong_class_p', 'group_id'], ascending=[True, False, True])
    save_csv(out, 'error_cases.csv', errors)
    return paired, counts, cases, errors


def check_actual_inputs(c, out, groups, manifest):
    design = read_result(out, 'oof_classifier_inputs.csv').set_index('group_id')
    for i in range(4):
        np.testing.assert_array_equal(design[f'M1_{i+1}'], design[f'M3_{i+1}'])
        np.testing.assert_array_equal(design[f'M2_{i+1}'], design[f'M3_{i+5}'])
    row = groups.sort_values(['tooth_id', 'state_id', 'angle_id', 'group_id']).iloc[0]
    member_ids = json.loads(row.member_ids)
    formal = pd.read_csv(c['feature_csv'], float_precision='round_trip').set_index('sample_id', verify_integrity=True)
    source = formal.loc[member_ids, FEATURES]
    ensure(set(member_ids) == set(manifest.loc[manifest.group_id == row.group_id, 'sample_id']),
           'The spot-check group has different registered members')
    direct = {k: sum(source[k].tolist()) / len(source) for k in FEATURES}
    np.testing.assert_allclose([direct[k] for k in FEATURES], row[FEATURES].to_numpy(float), rtol=0, atol=1e-12)
    dump(out / 'group_input_spotcheck.json', dict(group_id=row.group_id, held_tooth=row.tooth_id, members=member_ids,
         formal_member_values=source.reset_index().to_dict('records'), formal_column_means=direct,
         stored_group_means=row[FEATURES].to_dict(), actual_classifier_inputs=design.loc[row.group_id].to_dict(),
         all_groups_fusion_columns_equal=True))


def evaluate(c, out):
    oof, frames, groups, manifest = [
        read_result(out, k + '.csv') for k in ['oof_groups', 'oof_frames', 'groups', 'sample_manifest']]
    active = manifest[manifest.included]
    check_actual_inputs(c, out, groups, active)
    tooth_rows, stage_rows, cms, curves, overall = [], [], [], [], []
    for model in MODELS:
        sample = oof[oof.model == model]
        y = sample.true_state.map(STATES.index).to_numpy()
        pred = sample.pred_state.map(STATES.index).to_numpy()
        prob = sample[PROBS].to_numpy()
        np.testing.assert_array_equal(prob.argmax(axis=1), pred)
        result, stages = metrics(y, prob, STATES)
        cm = np.array(json.loads(result['confusion_matrix']))
        ensure(cm.sum() == len(groups), 'Confusion matrix denominator mismatch')
        for i, j in product(range(3), repeat=2):
            cms.append(dict(model=model, true_state=STATES[i], pred_state=STATES[j], groups=int(cm[i,j]),
                            true_stage_groups=int(cm[i].sum()), row_pct=100*cm[i,j]/cm[i].sum() if cm[i].sum() else np.nan))
        overall.append(dict(model=model, groups=len(y), correct=result['correct'], accuracy=result['accuracy'],
                            balanced_accuracy=result['balanced_accuracy'], ordinal_mae=float(np.abs(y-pred).mean()),
                            macro_auroc=result['roc_auc_ovr_macro'], log_loss=result['log_loss'],
                            adjacent_errors=int((np.abs(y-pred)==1).sum()), two_stage_errors=int((np.abs(y-pred)==2).sum())))
        for i, item in enumerate(stages):
            stage_rows.append(dict(model=model, state_id=STATES[i], true_groups=item['support'],
                predicted_groups=int(cm[:,i].sum()), precision=item['precision'], precision_defined=item['precision_defined'],
                recall=item['sensitivity'], f1=item['f1'], auroc=item['roc_auc_ovr'],
                auroc_note='' if item['roc_auc_ovr'] is not None else 'No positives or no negatives in held groups',
                **{f'to_{s}_n': cm[i,j] for j,s in enumerate(STATES)},
                **{f'to_{s}_pct': 100*cm[i,j]/cm[i].sum() if cm[i].sum() else np.nan for j,s in enumerate(STATES)}))
            if item['roc_auc_ovr'] is not None:
                fp, tp, th = roc_curve(y == i, prob[:,i], drop_intermediate=False)
                curves.extend(dict(model=model, state_id=STATES[i], fpr=f, tpr=t, threshold=v)
                              for f,t,v in zip(fp,tp,th))
        for tooth, block in sample.groupby('held_tooth'):
            absent = sorted(set(STATES) - set(block.true_state))
            f1 = f1_score(block.true_state, block.pred_state, labels=STATES, average=None, zero_division=0)
            tooth_rows.append(dict(tooth_id=tooth, model=model, groups=len(block), errors=int((block.true_state!=block.pred_state).sum()),
                missing_stages='|'.join(absent), macro_f1=np.nan if absent else f1.mean(),
                **{f'{s}_f1':f1[j] for j,s in enumerate(STATES)},
                **{f'{s}_true':int((block.true_state==s).sum()) for s in STATES},
                **{f'{s}_predicted':int((block.pred_state==s).sum()) for s in STATES}))
    teeth, stages = pd.DataFrame(tooth_rows), pd.DataFrame(stage_rows)
    scores = teeth.pivot(index='tooth_id', columns='model', values='macro_f1')
    bounds, gains, n = tooth_intervals(c, scores)
    summary = pd.DataFrame(overall).set_index('model')
    summary['primary_macro_f1'] = scores.dropna().mean()
    summary['primary_ci_low'] = [bounds[m][0] for m in summary.index]
    summary['primary_ci_high'] = [bounds[m][1] for m in summary.index]
    summary['evaluable_teeth'] = n
    summary = summary.reset_index()
    metric_rows = []
    for r in summary.to_dict('records'):
        for key, unit in [('primary_macro_f1','fraction'),('accuracy','fraction'),('balanced_accuracy','fraction'),
                          ('ordinal_mae','stage steps'),('macro_auroc','fraction'),('log_loss','nats/group'),
                          ('adjacent_errors','groups'),('two_stage_errors','groups')]:
            primary = key == 'primary_macro_f1'
            metric_rows.append(dict(model=r['model'], metric=key, value=r[key], unit=unit,
                evaluation='equal-weight teeth, group predictions' if primary else 'pooled out-of-fold groups',
                denominator=n if primary else r['groups'], ci_low=r['primary_ci_low'] if primary else None,
                ci_high=r['primary_ci_high'] if primary else None))
    for name, frame in [('metrics_summary', pd.DataFrame(metric_rows)), ('metrics_by_stage', stages),
                        ('metrics_by_tooth', teeth), ('paired_gains', gains), ('confusion_counts', pd.DataFrame(cms)),
                        ('roc_points', pd.DataFrame(curves))]:
        save_csv(out, name + '.csv', frame)
    paired, changes, cases, errors = paired_cases(out, groups, oof)
    features = groups.groupby(['tooth_id','state_id'], as_index=False)[FEATURES].median()
    save_csv(out, 'feature_stage_medians.csv', features)
    coverage = groups.groupby(['tooth_id','state_id'], as_index=False).agg(groups=('group_id','size'), frames=('group_frames','sum'))
    batch = groups.groupby(['session_id','state_id'], as_index=False).agg(groups=('group_id','size'),frames=('group_frames','sum'))
    save_csv(out, 'coverage_by_tooth.csv', coverage)
    save_csv(out, 'coverage_by_batch.csv', batch)
    s0 = frames.loc[frames.true_state == 'S0'].merge(active[['sample_id','qlf_path','fit_source']],
        on=['sample_id','qlf_path'], validate='many_to_one')
    ensure(len(s0) == 3 * int((active.state_id == 'S0').sum()), 'S0 calibration audit lost frames')
    s0['correct'] = s0.pred_state == 'S0'
    anchors = s0.groupby(['model','fit_source'], as_index=False).agg(
        frames=('sample_id','size'), correct=('correct','sum'), s0_recall=('correct','mean'), median_p_S0=('p_S0','median'))
    save_csv(out, 's0_reference_audit.csv', anchors)
    return dict(summary=summary, stages=stages, teeth=teeth, scores=scores, gains=gains,
                cms=pd.DataFrame(cms), curves=pd.DataFrame(curves), paired=paired, changes=changes,
                cases=cases, errors=errors, features=features, coverage=coverage, batch=batch,
                anchors=anchors, groups=groups, manifest=manifest)
