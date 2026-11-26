import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from qlf_stage3.modeling.cohort import FEATURES, PROBS
from qlf_stage3.modeling.training import fit_models, matrices, predict_acquisition
from qlf_stage3.rgb.color import RAW_MODE


class RgbFusionBundleTest(unittest.TestCase):
    def test_replaying_a_color_bundle_uses_its_saved_input_mode(self):
        rng = np.random.default_rng(93)
        white, qlf = rng.normal(size=(24,10)), rng.uniform(0,4,(24,4))
        config = dict(pca_components=4, classifier=dict(C=1,max_iter=500),
                      weights_sha256='fixture', qlf_workflow='rgb-fixture',white_light_mode=RAW_MODE)
        bundle = fit_models(config,white,qlf,np.tile([0,1,2],8))
        self.assertEqual(bundle['white_light_mode'],RAW_MODE)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            joblib.dump(bundle,root/'model.joblib')
            incoming = pd.DataFrame(qlf[:2],columns=FEATURES).assign(capture_id='group',rgb_path=['a.png','b.png'])
            incoming.to_csv(root/'inputs.csv',index=False)
            net=object()
            with patch('qlf_stage3.modeling.training.imagenet18',return_value=(net,None,None)), \
                 patch('qlf_stage3.modeling.training.white_vectors',return_value=white[:2]) as encode:
                predict_acquisition(config,root/'model.joblib',root/'inputs.csv',root/'pred')
            self.assertEqual(encode.call_args.args[2],RAW_MODE)
            result=pd.read_csv(root/'pred/predictions.csv').set_index('model')
            design=matrices(bundle,white[:2].mean(axis=0,keepdims=True),qlf[:2].mean(axis=0,keepdims=True))
            for model,inputs in design.items():
                expected=bundle['heads'][model].predict_proba(inputs)[0]
                np.testing.assert_allclose(result.loc[model,PROBS].to_numpy(float),expected,atol=1e-12,rtol=0)


if __name__=='__main__':
    unittest.main()
