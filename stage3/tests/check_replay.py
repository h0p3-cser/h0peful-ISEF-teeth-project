"""Exercise the installed prediction CLI on an unlabeled held-out acquisition."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from qlf_stage3.files import relative_path
from qlf_stage3.modeling.cohort import FEATURES, MODELS, PROBS


def check_replay(run, data):
    evaluation = run / "evaluation" if (run / "pipeline.json").is_file() else run
    frames = pd.read_csv(evaluation / "sample_manifest.csv", float_precision="round_trip")
    active = frames.loc[frames.included & (frames.group_frames > 1)]
    group_id = sorted(active.group_id.unique())[0]
    members = active.loc[active.group_id == group_id].copy()
    held = members.tooth_id.iloc[0]
    model = evaluation / "models" / f"{held}.joblib"
    reference = pd.read_csv(evaluation / "oof_groups.csv", float_precision="round_trip")
    reference = reference.loc[reference.group_id == group_id].set_index("model")
    errors = {}
    with TemporaryDirectory(prefix="qlf-unlabelled-") as temp:
        work = Path(temp)
        batch = members[FEATURES].copy()
        batch.insert(0, "capture_id", "new-capture")
        batch["rgb_path"] = [relative_path(data / path, work) for path in members.rgb_resolved]
        for arm in ("all", "M1", "M2"):
            fields = ["capture_id"]
            if arm != "M1":
                fields.extend(FEATURES)
            if arm != "M2":
                fields.append("rgb_path")
            input_file = work / (arm + ".csv")
            batch[fields].to_csv(input_file, index=False)
            output = work / arm
            subprocess.run([sys.executable, "-m", "qlf_stage3", "predict", "--arm", arm,
                            "--model", str(model), "--data", str(data), "--input", str(input_file),
                            "--out", str(output)], cwd=work, check=True)
            result = pd.read_csv(output / "predictions.csv", float_precision="round_trip").set_index("model")
            expected_arms = MODELS if arm == "all" else [arm]
            assert result.index.tolist() == expected_arms
            assert (result.group_frames == len(members)).all() and (result.capture_id == "new-capture").all()
            expected = reference.loc[expected_arms]
            np.testing.assert_array_equal(result.pred_state, expected.pred_state)
            np.testing.assert_allclose(result[PROBS], expected[PROBS], rtol=0, atol=1e-6)
            errors[arm] = float(np.abs(result[PROBS] - expected[PROBS]).to_numpy().max())
    return dict(members=len(members), held_tooth=held, models=MODELS, largest_probability_difference=errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(check_replay(args.run.resolve(), args.data.resolve()), indent=2))
