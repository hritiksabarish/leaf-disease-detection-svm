import sys
import unittest
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from app import extract_features_pipeline, load_model, load_label_encoder
import cv2

class TestAppIntegration(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()
        cls.le, cls.expected_features = load_label_encoder()
        
    def test_model_loaded(self):
        self.assertIsNotNone(self.model)
        
    def test_feature_compatibility(self):
        # The expected features list should have 24 features
        self.assertEqual(len(self.expected_features), 24)
        
    def test_pipeline_on_synthetic_image(self):
        # Create a blank green image
        img = np.full((256, 256, 3), (35, 140, 45), dtype=np.uint8)
        
        feats_dict, leaf_mask, diseased_mask = extract_features_pipeline(img)
        
        # Verify 24 features are extracted
        for f in self.expected_features:
            self.assertIn(f, feats_dict)
            
        X_vec = np.array([feats_dict[c] for c in self.expected_features]).reshape(1, -1)
        self.assertEqual(X_vec.shape, (1, 24))
        
        # Verify prediction works
        pred_encoded = self.model.predict(X_vec)[0]
        predicted_label = self.le.inverse_transform([pred_encoded])[0]
        
        self.assertTrue(isinstance(predicted_label, str))
        self.assertTrue(len(predicted_label) > 0)

if __name__ == '__main__':
    unittest.main()
