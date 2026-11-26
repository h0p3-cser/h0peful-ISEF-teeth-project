"""Jointly hold out two teeth whose catalog and historical study labels cross."""
import logging

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..files import load
from .cohort import MODELS, PROBS, STATES, ensure, save_csv
from .evaluation import read_result
from .training import fit_models, group_features, matrices, predict_matrix


def jointly_held_teeth(root,plan):
    frames = read_result(root,"paired_inputs.csv")
    conflict = frames[(frames.tooth_id != frames.rgb_catalog_tooth) | (frames.tooth_id != frames.qlf_catalog_tooth)]
    if conflict.empty:
        return None
    affected = sorted(set(conflict.tooth_id) | set(conflict.rgb_catalog_tooth) | set(conflict.qlf_catalog_tooth))
    ensure(affected == ["T004","T005"],"Review unexpected tooth aliases before running the T004/T005 sensitivity analysis")
    swaps = conflict.groupby(["tooth_id","rgb_catalog_tooth","qlf_catalog_tooth","state_id"],as_index=False).agg(
        frames=("sample_id","size"),groups=("group_id","nunique"))
    dest = root / "joint_tooth_holdout"
    dest.mkdir(exist_ok=True)
    save_csv(dest,"catalog_labels.csv",swaps)
    counts, predictions = [], []
    seed = plan["training_seeds"][0]
    logging.info("Sensitivity check using seed %d: holding out %s together, %d pairs with different catalog labels",
                 seed," and ".join(affected),len(conflict))
    for branch in plan["branches"]:
        ref = root / branch / f"seed_{seed}"
        samples = read_result(ref,"sample_manifest.csv")
        with np.load(ref/"frozen_frame_features.npz",allow_pickle=False) as cache:
            np.testing.assert_array_equal(samples.sample_id,cache["sample_ids"])
            embeddings = cache["white"]
        groups,visual,qlf = group_features(samples,embeddings)
        held = groups.tooth_id.isin(affected).to_numpy()
        train = ~held
        train_photos = samples[samples.group_id.isin(groups.loc[train,"group_id"])]
        ensure(not train_photos[["tooth_id","rgb_catalog_tooth","qlf_catalog_tooth"]].isin(affected).any().any(),
               "Jointly held teeth occur in training")
        cfg = load(ref/"run_config.json")
        y = groups.state_id.map(STATES.index).to_numpy()
        with threadpool_limits(limits=1,user_api="blas"):
            bundle = fit_models(cfg,visual[train],qlf[train],y[train])
            bundle["training_teeth"] = sorted(groups.loc[train,"tooth_id"].unique())
            bundle["held_teeth"] = affected
            ensure(len(bundle["training_teeth"]) == 7,"Expected seven other training teeth")
            joblib.dump(bundle,dest/(branch+".joblib"),compress=3)
            bundle = joblib.load(dest/(branch+".joblib"))
            design = matrices(bundle,visual[held],qlf[held])
            baseline = read_result(ref,"oof_groups.csv").set_index(["model","group_id"],verify_integrity=True)
            test = groups.loc[held,["group_id","tooth_id","state_id"]].reset_index(drop=True)
            for model in MODELS:
                predicted = predict_matrix(bundle,model,design[model])
                old = baseline.loc[model].loc[test.group_id].reset_index(drop=True)
                ensure(old.true_state.equals(test.state_id),"Wrong reference groups for joint holdout")
                details = pd.concat([test,predicted],axis=1).assign(branch=branch,seed=seed,previous_stage=old.pred_state)
                for key in PROBS:
                    details["previous_"+key] = old[key]
                predictions.append(details)
                old_correct = old.pred_state.eq(test.state_id)
                new_correct = predicted.pred_state.eq(test.state_id)
                s1 = test.state_id.eq("S1")
                counts.append(dict(branch=branch,model=model,seed=seed,held_teeth="|".join(affected),
                    train_teeth=len(bundle["training_teeth"]),train_groups=int(train.sum()),test_groups=len(test),
                    separate_holdouts_correct=int(old_correct.sum()),joint_holdout_correct=int(new_correct.sum()),
                    separate_s1_correct=int((old_correct&s1).sum()),joint_s1_correct=int((new_correct&s1).sum()),
                    s1_groups=int(s1.sum()),changed_decisions=int(old.pred_state.ne(predicted.pred_state).sum())))
    summary = pd.DataFrame(counts)
    save_csv(dest,"scores.csv",summary)
    save_csv(dest,"predictions.csv",pd.concat(predictions,ignore_index=True))
    logging.info("Joint T004/T005 check completed, %d classifiers",len(summary))
    return swaps,summary
