"""Replay both installed endpoint inference bundles on one unlabeled acquisition."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

from qlf_stage3.files import load, relative_path
from qlf_stage3.modeling.cohort import FEATURES, MODELS, PROBS, STATES
from qlf_stage3.modeling.training import matrices


def check(run, data, weights=None):
    root = run / "stability"
    plan = load(root / "plan.json")
    first_seed = plan["training_seeds"][0]
    assert len(plan["training_seeds"]) == 3
    assert sorted(plan["branches"]) == ["gray", "rgb"]
    errors = {}
    with TemporaryDirectory(prefix="qlf-predict-") as temp:
        work = Path(temp)
        for branch, mode in plan["branches"].items():
            evaluation = root / branch / f"seed_{first_seed}"
            frames = pd.read_csv(evaluation / "sample_manifest.csv", float_precision="round_trip")
            group_id = sorted(frames.loc[frames.group_frames > 1, "group_id"].unique())[0]
            members = frames[frames.group_id == group_id].sort_values("sample_id")
            batch = members[FEATURES].copy()
            batch.insert(0, "capture_id", "unlabeled-pair")
            batch["rgb_path"] = [relative_path(data / path, work) for path in members.rgb_resolved]
            csv = work / (branch + ".csv")
            batch.to_csv(csv, index=False)
            bundle_path = root / "models" / (branch + ".joblib")
            output = work / branch
            cmd = [sys.executable, "-m", "qlf_stage3", "predict", "--data", str(data),
                   "--input", str(csv), "--model", str(bundle_path), "--out", str(output)]
            if weights:
                cmd.extend(["--weights", str(weights)])
            subprocess.run(cmd, cwd=work, check=True)
            result = pd.read_csv(output / "predictions.csv", float_precision="round_trip").set_index("model")
            assert list(result.index) == MODELS
            assert (result.group_frames == len(members)).all() and (result.capture_id == "unlabeled-pair").all()
            probs = result[PROBS].to_numpy()
            np.testing.assert_allclose(probs.sum(axis=1), 1, rtol=0, atol=1e-10)
            np.testing.assert_array_equal(result.pred_state, np.asarray(STATES)[probs.argmax(axis=1)])
            bundle = joblib.load(bundle_path)
            assert bundle["white_light_mode"] == mode
            with np.load(evaluation / "frozen_frame_features.npz", allow_pickle=False) as cache:
                np.testing.assert_array_equal(cache["sample_ids"], frames.sample_id)
                white = cache["white"][members.index].astype(float).mean(axis=0, keepdims=True)
            qlf = members[FEATURES].to_numpy(float).mean(axis=0, keepdims=True)
            design = matrices(bundle, white, qlf)
            expected = np.vstack([bundle["heads"][m].predict_proba(design[m]) for m in MODELS])
            np.testing.assert_allclose(probs, expected, rtol=0, atol=1e-6)
            errors[branch] = float(np.abs(probs - expected).max())
    return dict(members=len(members), models=MODELS, max_probability_difference=errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.run.resolve(), args.data.resolve(), args.weights.resolve() if args.weights else None), indent=2))
