"""Evaluate M1 visual, M2 QLF and M3 fusion on one frozen paired cohort."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..files import dump, load, sha
from ..modeling.cohort import MODELS, PROBS, ensure, save_csv
from ..modeling.evaluation import evaluate, read_result
from ..modeling.report import figures
from ..modeling.training import fold_predictions, group_features
from ..modeling.validation import verify_predictions
from ..modeling.vision import white_input
from ..rgb.color import RAW_MODE
from .fusion_inputs import CONTROL, GRAY_CONTROL, encode_same_source_gray, same_folds
from .inputs import CALIBRATED, timestamp

POLICIES = ('branch_rgb','original_rgb','sibling_gray')


def train_variant(directory, folds):
    config = load(directory/'run_config.json')
    config['feature_csv'] = (directory/config['feature_csv']).resolve()
    ensure(sha(config['feature_csv'])==config['csv_sha256'],'QLF source differs')
    if (directory/'validation.json').is_file():
        same_folds(read_result(directory,'folds.csv'),folds)
        verify_predictions(directory,config['feature_csv'])
        return evaluate(config,directory)
    frames = read_result(directory,'sample_manifest.csv')
    with np.load(directory/'frozen_frame_features.npz',allow_pickle=False) as cached:
        np.testing.assert_array_equal(cached['sample_ids'],frames.sample_id)
        np.testing.assert_array_equal(cached['cache_keys'],frames.embedding_cache_key)
        visual = cached['white']
    groups, white, qlf = group_features(frames,visual)
    status = frames.groupby('group_id').status
    groups['corrected_frames_in_branch'] = groups.group_id.map(status.agg(lambda x: x.eq(CALIBRATED).sum()))
    save_csv(directory,'groups.csv',groups)
    with threadpool_limits(limits=1,user_api='blas'):
        fold_predictions(config,directory,groups,white,qlf,frames,visual)
    same_folds(read_result(directory,'folds.csv'),folds)
    dump(directory/'oof_sha256.json',{n:sha(directory/n) for n in ('oof_groups.csv','oof_frames.csv')})
    results = evaluate(config,directory)
    dump(directory/'validation.json',verify_predictions(directory,config['feature_csv']))
    return results


def ordered_prediction(table, columns):
    return table.set_index('group_id',verify_integrity=True).sort_index()[columns]


def probability_check(left,right,description):
    pd.testing.assert_index_equal(left.index,right.index)
    difference = float(np.abs(left.to_numpy()-right.to_numpy()).max())
    np.testing.assert_allclose(left.to_numpy(),right.to_numpy(),rtol=0,atol=1e-8,err_msg=description)
    return difference


def replay_anchors(results, branch, history):
    prior_white = read_result(branch,'oof_groups.csv')
    historical = read_result(history,'oof_groups.csv')
    current = results['branch_rgb']['paired']
    agreement = {}
    for policy, prefix in [('branch_rgb','rgb'),('sibling_gray','gray')]:
        previous = ordered_prediction(prior_white,[f'{prefix}_{col}' for col in PROBS])
        repeated = ordered_prediction(results[policy]['paired'],['M1_'+c for c in PROBS])
        agreement[policy+'_M1_vs_white_only'] = probability_check(repeated,previous,policy)
    q = historical[historical.model.eq('M2')]
    expected = ordered_prediction(q,PROBS)
    for policy,r in results.items():
        after = ordered_prediction(r['paired'],['M2_'+c for c in PROBS])
        agreement[policy+'_M2_vs_historical_QLF'] = probability_check(after,expected,policy)
        pd.testing.assert_frame_equal(ordered_prediction(r['paired'],['state_id','tooth_id','fold','member_ids']),
                                      ordered_prediction(current,['state_id','tooth_id','fold','member_ids']))
    return agreement


def side_by_side(out, results):
    summaries, paired = [], None
    for policy,r in results.items():
        stages = r['stages'].set_index(['model','state_id'])
        for s in r['summary'].itertuples():
            s1 = stages.loc[s.model,'S1']
            summaries.append(dict(white_policy=policy,model=s.model,groups=s.groups,correct=s.correct,
                teeth=s.evaluable_teeth,tooth_equal_macro_f1=s.primary_macro_f1,
                tooth_bootstrap_ci_low=s.primary_ci_low,tooth_bootstrap_ci_high=s.primary_ci_high,
                s1_correct=int(s1.to_S1_n),s1_groups=int(s1.true_groups),s1_recall=s1.recall,
                s1_to_s0=int(s1.to_S0_n),s1_to_s2=int(s1.to_S2_n),s0_s2_errors=s.two_stage_errors))
        src = r['paired'].set_index('group_id',verify_integrity=True).sort_index()
        if paired is None:
            paired = src[['tooth_id','state_id','angle_id','session_id','fold','group_frames',
                          'corrected_frames_in_branch','member_ids']].copy()
        for model in MODELS:
            for col in ['pred_state',*PROBS]:
                paired[policy+'_'+model+'_'+col] = src[model+'_'+col]
    save_csv(out,'all_policy_group_predictions.csv',paired.reset_index())
    summary = pd.DataFrame(summaries)
    save_csv(out,'model_comparison.csv',summary)
    return summary


def paired_differences(out,results):
    comparisons = [('M3_vs_RGB','branch_rgb','M3','branch_rgb','M1'),
                   ('M3_vs_QLF','branch_rgb','M3','branch_rgb','M2'),
                   ('correction_visual','branch_rgb','M1','original_rgb','M1'),
                   ('correction_fusion','branch_rgb','M3','original_rgb','M3'),
                   ('color_visual','branch_rgb','M1','sibling_gray','M1'),
                   ('color_fusion','branch_rgb','M3','sibling_gray','M3')]
    summaries, by_tooth = [], []
    for name,left,m1,right,m2 in comparisons:
        a = results[left]['paired'].set_index('group_id').sort_index()
        b = results[right]['paired'].set_index('group_id').sort_index()
        pd.testing.assert_index_equal(a.index,b.index)
        change = results[left]['scores'][m1] - results[right]['scores'][m2]
        prob = np.abs(a[[m1+'_'+k for k in PROBS]].to_numpy() -
                      b[[m2+'_'+k for k in PROBS]].to_numpy())
        a_pred,b_pred = a[m1+'_pred_state'],b[m2+'_pred_state']
        a_correct,b_correct = a_pred.eq(a.state_id),b_pred.eq(b.state_id)
        s1 = a.state_id.eq('S1')
        summaries.append(dict(comparison=name,left=left+'_'+m1,right=right+'_'+m2,
            delta_tooth_equal_macro_f1=change.mean(),improved_teeth=int((change>1e-10).sum()),
            tied_teeth=int((change.abs()<=1e-10).sum()),declined_teeth=int((change< -1e-10).sum()),
            changed_group_predictions=int(a_pred.ne(b_pred).sum()),newly_correct=int((a_correct & ~b_correct).sum()),
            newly_wrong=int((~a_correct & b_correct).sum()),delta_s1_recall=a_correct[s1].mean()-b_correct[s1].mean(),
            mean_absolute_probability_delta=prob.mean(),max_absolute_probability_delta=prob.max()))
        by_tooth.extend(dict(comparison=name,tooth_id=tooth,delta_macro_f1=v) for tooth,v in change.items())
    summary = pd.DataFrame(summaries)
    save_csv(out,'paired_differences.csv',summary)
    save_csv(out,'paired_differences_by_tooth.csv',pd.DataFrame(by_tooth))
    return summary


def compare(out,checkpoint):
    ensure(not (out/'complete.json').exists(),'This comparison is already complete')
    encoded = load(out/'encoded.json')
    for name,digest in encoded['checksums'].items():
        ensure(sha(out/name)==digest,'Changed RGB comparison input: '+name)
    settings = load(out/'run_config.json')
    branch = (out/settings['source_branch']).resolve()
    history = (out/settings['history']).resolve()
    folds = read_result(out,'expected_folds.csv')
    encode_same_source_gray(out,checkpoint)
    directories = [out,out/CONTROL,out/GRAY_CONTROL]
    results = {}
    for policy,directory in zip(POLICIES,directories):
        print('Fitting and replaying nine held-tooth folds: '+policy,flush=True)
        results[policy] = train_variant(directory,folds)
    agreement = replay_anchors(results,branch,history)
    scores = side_by_side(out,results)
    differences = paired_differences(out,results)
    dump(out/'historical_probability_checks.json',agreement)
    figures(out,results['branch_rgb'],out,names={'M1':'M1 彩色白光','M2':'M2 QLF','M3':'M3 RGB＋QLF'},
            photo_input=lambda path:white_input(path,RAW_MODE))
    from .fusion_report import write_report
    write_report(out,branch,results['branch_rgb'],scores,differences)
    dump(out/'complete.json',dict(completed=timestamp(),frames=encoded['frames'],groups=encoded['groups'],
        teeth=encoded['teeth'],corrected_frames=encoded['corrected'],corrected_groups=encoded['corrected_groups'],
        uncorrected_frames=encoded['passthrough'],fixed_30_percent=True,identical_folds=True,
        group_probability_rows_per_policy=len(folds[folds.role.eq('test')])*len(MODELS),
        all_saved_models_replayed=True,reference_max_probability_difference=max(agreement.values()),
        sha256={n:sha(out/n) for n in ('model_comparison.csv','paired_differences.csv',
                'all_policy_group_predictions.csv','oof_groups.csv','report.md')}))
    print(scores[['white_policy','model','tooth_equal_macro_f1','s1_correct','correct']].to_string(index=False),flush=True)
