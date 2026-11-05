import copy
import json
import unittest
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from balanced_features import COLUMNS
from feature_model import validate_input,bundle_model,predict_frame
from research_controls import fit_preprocessing,transform_features
from linear_losses import fit_linear_loss,LossConfig


class FeatureModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(18)
        cls.frame=pd.DataFrame(rng.normal(size=(100,500)),columns=COLUMNS)
        cls.frame.iloc[:,-50:]=np.nan
        with threadpool_limits(limits=2):
            cls.stats=fit_preprocessing(cls.frame,COLUMNS)
            x=transform_features(cls.frame,cls.stats)
            model=fit_linear_loss(x,rng.normal(size=100),config=LossConfig(loss='mse',l2=.1))
            cls.bundle=bundle_model(model,cls.stats)

    def test_500_candidates_dynamic_active_dimensions(self):
        self.assertEqual(self.bundle['candidate_count'],500)
        self.assertEqual(len(self.bundle['linear_model']['coefficient']),450)
        self.assertEqual(list(self.bundle['coefficient_by_name']),self.stats['active_names'])

    def test_serialization_and_reordered_columns_same_predictions(self):
        loaded=json.loads(json.dumps(self.bundle))
        original=predict_frame(loaded,self.frame)
        reordered=self.frame.loc[:,list(reversed(COLUMNS))].assign(unused_metadata=1)
        np.testing.assert_allclose(original,predict_frame(loaded,reordered))

    def test_old_50_or_missing_factor_rejected(self):
        with self.assertRaisesRegex(ValueError,'missing'):validate_input(self.frame.iloc[:,:50])
        with self.assertRaisesRegex(ValueError,'count mismatch'):transform_features(np.zeros((2,50)),self.stats)
        with self.assertRaises(ValueError):predict_frame(self.bundle,self.frame.drop(columns=COLUMNS[-1]))

    def test_wrong_catalog_or_coefficient_map_rejected(self):
        bad=copy.deepcopy(self.bundle);bad['catalog_sha256']='old'
        with self.assertRaisesRegex(ValueError,'version mismatch'):predict_frame(bad,self.frame)
        bad=copy.deepcopy(self.bundle);bad['active_feature_names'].reverse()
        with self.assertRaisesRegex(ValueError,'metadata mismatch'):predict_frame(bad,self.frame)


if __name__=='__main__':unittest.main()
