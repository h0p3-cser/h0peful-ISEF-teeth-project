import unittest

import numpy as np
import pandas as pd

from qlf_stage3.modeling.cohort import MODELS, PROBS
from qlf_stage3.modeling.seed_report import differences
from qlf_stage3.modeling.training import fit_models, matrices


class SeedStabilityTests(unittest.TestCase):
    def test_seed_is_preserved_in_fitted_estimators(self):
        rng = np.random.default_rng(417)
        visual, qlf = rng.normal(size=(30,10)), rng.normal(size=(30,4))
        y = np.tile([0,1,2],10)
        base = dict(pca_components=4,weights_sha256="fixture",qlf_workflow="fixture",
                    classifier=dict(C=1.0,solver="lbfgs",max_iter=200))
        predictions = []
        for seed in (42,123):
            bundle = fit_models(dict(base,training_seed=seed),visual,qlf,y)
            self.assertEqual(bundle["training_seed"],seed)
            self.assertEqual(bundle["pca"].random_state,seed)
            self.assertTrue(all(bundle["heads"][model].random_state==seed for model in MODELS))
            x = matrices(bundle,visual,qlf)
            predictions.append([bundle["heads"][model].predict_proba(x[model]) for model in MODELS])
        np.testing.assert_array_equal(predictions[0],predictions[1])

    def test_comparison_uses_all_seed_pairs_and_counts_each_group_once(self):
        samples = []
        for probs in ([.50,.50,0.], [.75,.25,0.], [.25,.75,0.]):
            p = pd.DataFrame([probs,[0.,0.,1.]],columns=PROBS)
            p["pred_state"] = np.array(["S0","S1","S2"])[p[PROBS].to_numpy().argmax(axis=1)]
            samples.append(p)
        maximum, changed = differences(samples)
        self.assertEqual(maximum,.5)
        self.assertEqual(changed,1)
