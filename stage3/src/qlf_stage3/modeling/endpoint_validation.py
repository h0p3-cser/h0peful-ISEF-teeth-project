"""Fit the two inference bundles and independently replay the published endpoint results."""
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..bundle import check_data
from ..files import PACKAGE, dataset_info, dump, load, sha
from .cohort import FEATURES, MODELS, ensure
from .evaluation import read_result
from .training import fit_models, group_features, matrices
from .validation import verify_predictions


def final_models(root, plan):
    directory = root / "models"
    directory.mkdir(exist_ok=True)
    first_seed = plan["training_seeds"][0]
    records = {}
    for branch in plan["branches"]:
        run = root / branch / f"seed_{first_seed}"
        cfg = load(run / "run_config.json")
        frames = read_result(run, "sample_manifest.csv")
        with np.load(run/"frozen_frame_features.npz", allow_pickle=False) as cache:
            np.testing.assert_array_equal(cache["sample_ids"], frames.sample_id)
            groups, white, qlf = group_features(frames, cache["white"])
        with threadpool_limits(limits=1, user_api="blas"):
            y = groups.state_id.map(cfg["states"].index).to_numpy()
            bundle = fit_models(cfg, white, qlf, y)
        bundle.update(training_teeth=sorted(groups.tooth_id.unique()), training_groups=len(groups),
                      scope="all-group inference only", endpoint_branch=branch)
        path = directory / f"{branch}.joblib"
        pending = path.with_suffix(".tmp")
        joblib.dump(bundle, pending, compress=3)
        pending.replace(path)
        replay = joblib.load(path)
        design = matrices(replay,white,qlf)
        for m in MODELS:
            np.testing.assert_allclose(replay["heads"][m].predict_proba(design[m]),
                                       bundle["heads"][m].predict_proba(design[m]),rtol=0,atol=1e-12)
        records[branch] = dict(path=path.relative_to(root).as_posix(),sha256=sha(path),seed=first_seed,
                               groups=len(groups),mode=cfg["white_light_mode"])
    dump(directory/"index.json",records)


def verify_stability(root, data=None, features=None, reference_study=False):
    from .seed_report import collect, summarize

    plan = load(root/"plan.json")
    audit = load(root/"verification.json")
    ensure(audit["complete"] and load(root/"status.json")["status"]=="complete","Expected a completed endpoint")
    dataset,descriptor = dataset_info(data or root/plan["dataset"])
    feature_csv = (features or root/plan["feature_csv"]).resolve()
    for path,expected in [(feature_csv,plan["sha256"]["features"]),(root/"paired_inputs.csv",plan["sha256"]["paired_inputs"]),
                          (dataset/"files.sha256",plan["sha256"]["dataset_index"])]:
        ensure(sha(path)==expected,f"Changed evaluation input: {path.name}")
    input_check = check_data(dataset)
    for name,digest in audit["sha256"].items():
        ensure(sha(root/name)==digest,"Changed endpoint output: "+name)
    scores,teeth,outputs = collect(root,plan)
    summary,deltas,_,_ = summarize(scores,teeth,outputs,plan)
    for expected,name in [(scores,"scores_by_seed.csv"),(summary,"stability_summary.csv"),(deltas,"paired_comparisons.csv")]:
        pd.testing.assert_frame_equal(expected,read_result(root,name),check_dtype=False,rtol=1e-10,atol=1e-12)
    heads = sum(verify_predictions(root/b/f"seed_{s}",feature_csv)["converged_heads"]
                for b in plan["branches"] for s in plan["training_seeds"])
    index = load(root/"models/index.json")
    for branch, mode in plan["branches"].items():
        entry = index[branch]
        ensure(entry["path"] == f"models/{branch}.joblib" and sha(root/entry["path"])==entry["sha256"],
               f"Changed inference model: {branch}")
        model = joblib.load(root/entry["path"])
        ensure(model["white_light_mode"]==mode and model["training_groups"]==plan["groups"] and
               model["training_teeth"]==plan["teeth"] and model["training_seed"]==plan["training_seeds"][0]
               and model["feature_columns"]==FEATURES,"Unexpected inference model schema")
    result = dict(valid=True, dataset_kind=descriptor["kind"], pairs=plan["paired_frames"], groups=plan["groups"],
                  held_teeth=len(plan["teeth"]), seeds=plan["training_seeds"], branches=list(plan["branches"]),
                  fold_classifiers_replayed=heads, inference_bundles=len(index), verified_dataset_files=input_check["verified_files"])
    if reference_study:
        ensure(descriptor["kind"]=="study","The reference scores apply to the authorized study data")
        reference=load(PACKAGE/"config/endpoint-reference.json")
        ensure(plan["training_seeds"]==reference["seeds"] and plan["paired_frames"]==reference["pairs"]
               and plan["groups"]==reference["groups"] and sha(feature_csv)==reference["features_sha256"],
               "Different reference cohort, seeds or feature version")
        for r in scores.itertuples():
            expected=reference["branches"][r.branch][r.model]
            np.testing.assert_allclose(r.tooth_macro_f1,expected["macro_f1"],rtol=0,atol=1e-8)
            ensure(r.correct==expected["correct"] and r.s1_correct==expected["s1_correct"],"Reference counts differ")
        result["reference_study_matches"]=True
    return result
