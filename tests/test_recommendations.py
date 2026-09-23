import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recommendations import get_recommendation, normalize_class_name, KNOWLEDGE_BASE

class TestRecommendations(unittest.TestCase):
    
    def setUp(self):
        self.raw_labels = [
            "Pepper_bell___Bacterial_spot",
            "Pepper_bell___healthy",
            "Potato___Early_blight",
            "Potato___healthy",
            "Potato___Late_blight",
            "Tomato___Bacterial_spot",
            "Tomato___Early_blight",
            "Tomato___healthy",
            "Tomato___Late_blight",
            "Tomato___Leaf_Mold"
        ]

    def test_normalization_logic(self):
        # 5. Label normalization works for the actual labels in the dataset
        self.assertEqual(normalize_class_name("Tomato___Late_blight"), "tomato_late_blight")
        self.assertEqual(normalize_class_name("Pepper__bell___Bacterial_spot"), "pepper_bell_bacterial_spot")
        self.assertEqual(normalize_class_name("Potato___healthy"), "potato_healthy")
        self.assertEqual(normalize_class_name(" Tomato Leaf Mold "), "tomato_leaf_mold")

    def test_all_classes_covered(self):
        # 1. Every one of the 10 classes has a recommendation
        for raw_label in self.raw_labels:
            rec = get_recommendation(raw_label)
            self.assertNotEqual(rec['status'], 'unknown', f"Missing KB entry for {raw_label}")
            self.assertEqual(len(rec.keys()), 8, "Missing fields in recommendation schema")

    def test_healthy_vs_diseased_status(self):
        # 2 & 3. Every disease class returns status = "diseased", healthy returns "healthy"
        for raw_label in self.raw_labels:
            rec = get_recommendation(raw_label)
            if "healthy" in raw_label.lower():
                self.assertEqual(rec['status'], "healthy")
                self.assertEqual(rec['disease'], "None")
                self.assertEqual(rec['source'], [])
                self.assertIn("No chemical disease treatment is recommended", rec['management'][0])
            else:
                self.assertEqual(rec['status'], "diseased")
                self.assertNotEqual(rec['disease'], "None")

    def test_unknown_label_graceful_fail(self):
        # 4. Unknown classifier labels fail gracefully
        rec = get_recommendation("Apple___Apple_scab")
        self.assertEqual(rec['status'], "unknown")
        self.assertEqual(rec['crop'], "Unknown")
        self.assertEqual(rec['disease'], "Unknown")
        self.assertTrue("No knowledge base entry found" in rec['summary'])

    def test_no_empty_recommendations(self):
        # 6. No recommendation entry is empty
        for raw_label in self.raw_labels:
            rec = get_recommendation(raw_label)
            self.assertTrue(bool(rec['summary']))
            self.assertTrue(len(rec['management']) > 0)
            self.assertTrue(len(rec['prevention']) > 0)
            self.assertTrue(bool(rec['caution']))

    def test_sources_present_and_valid(self):
        # 7. Every disease recommendation has at least one source
        # 8. Sources are validly structured
        for raw_label in self.raw_labels:
            rec = get_recommendation(raw_label)
            if rec['status'] == 'diseased':
                self.assertTrue(len(rec['source']) > 0, f"No source for {raw_label}")
                for source in rec['source']:
                    self.assertIn('organization', source)
                    self.assertIn('title', source)
                    self.assertIn('url', source)
                    self.assertTrue(source['url'].startswith('http'))


if __name__ == '__main__':
    unittest.main()
