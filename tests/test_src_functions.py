"""run with pytest"""

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset_utils import calculate_sha256
from src.feature_extraction import (
    extract_region_rgb_features, extract_whole_image_rgb_features,
)
from src.patch_processing import (
    calculate_patch_overlap, extract_patches, extract_training_patches,
    iter_rgb_patches, label_patch, label_patch_class,
)
from src.model_evaluation import regional_predictions_to_image_labels


class SourceFunctionTests(unittest.TestCase):
    def test_file_hash(self):
        #Intended outcome: The file hash matches the SHA-256 digest of its contents.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.bin"
            path.write_bytes(b"image contents")
            self.assertEqual(calculate_sha256(path), hashlib.sha256(b"image contents").hexdigest())

    def test_rgb_feature_names_and_values(self):
        #Intended outcome: Both extractors return channel means of 5, 25 and 45,
        # with standard deviations of 5 and the correct region or whole-image feature names.
        image = np.array([[[0, 20, 40], [10, 30, 50]]], dtype=np.uint8)
        region = extract_region_rgb_features(image)
        whole = extract_whole_image_rgb_features(image)
        self.assertEqual(region, dict(mean_r=5, mean_g=25, mean_b=45, std_r=5, std_g=5, std_b=5))
        self.assertEqual(list(whole.values()), list(region.values()))
        self.assertEqual(list(whole), ["mean_red", "mean_green", "mean_blue", "std_red", "std_green", "std_blue"])

    def test_patch_boundaries_and_order(self):
        #Intended outcome: Patches follow row-first order and retain smaller boundary patches,
        # cover all 35 pixels, and contain the same pixels as the patch iterator.
        image = np.arange(5 * 7 * 3).reshape(5, 7, 3)
        patches = extract_patches(image, 4)
        self.assertEqual([(p["x"], p["y"], p["width"], p["height"]) for p in patches],
                         [(0, 0, 4, 4), (4, 0, 3, 4), (0, 4, 4, 1), (4, 4, 3, 1)])
        for patch, pixels in zip(patches, iter_rgb_patches(image, 4)):
            np.testing.assert_array_equal(patch["rgb"], pixels)
        self.assertEqual(sum(p["width"] * p["height"] for p in patches), 35)

    def test_different_patch_defaults_are_preserved(self):
        #Intended outcome: A 65-by-65 image produces four patches with the training
        # default of 64 pixels and nine patches with the standard default of 32 pixels.
        image = np.zeros((65, 65, 3))
        self.assertEqual(len(extract_training_patches(image)), 4)
        self.assertEqual(len(extract_patches(image)), 9)

    def test_overlap_and_label_return_contracts(self):
        #Intended outcome: Overlap of 20% is defective (1), 10% is ambiguous (-1),
        # and no overlap is normal (0); tuple results include overlap and category details.
        # Images without annotations return a normal label and no matched category.
        patch = dict(x=0, y=0, width=10, height=10)
        for box, expected_overlap, expected_label in [([0, 0, 2, 10], .2, 1), ([0, 0, 1, 10], .1, -1), ([20, 20, 10, 10], 0, 0)]:
            annotations = [dict(bbox=box, category_id=7)]
            self.assertEqual(calculate_patch_overlap(patch, box), expected_overlap)
            self.assertEqual(label_patch(patch, annotations), (expected_label, expected_overlap, 7 if expected_overlap else None))
            self.assertEqual(label_patch_class(patch, annotations), expected_label)
        self.assertEqual(label_patch(patch, []), (0, 0, None))
        self.assertEqual(label_patch_class(patch, []), 0)

    def test_regional_predictions_align_image_ids_and_strict_threshold(self):
        #Intended outcome: Counts follow the requested image-ID order, with one defect per image.
        # Both images PASS at a threshold of 1 and FAIL at 0 because counts must strictly
        # exceed the threshold; the recorded elapsed time is non-negative.
        class Model:
            def predict(self, features):
                return np.array([1, 1, 0])
        regions = pd.DataFrame({"image_id": [20, 10, 20]})
        images = pd.DataFrame({"image_id": [10, 20]})
        labels, counts, seconds = regional_predictions_to_image_labels(Model(), 1, "test", np.zeros((3, 6)), regions, images)
        np.testing.assert_array_equal(counts, [1, 1])
        np.testing.assert_array_equal(labels, ["PASS", "PASS"])
        labels, _, _ = regional_predictions_to_image_labels(Model(), 0, "test", np.zeros((3, 6)), regions, images)
        np.testing.assert_array_equal(labels, ["FAIL", "FAIL"])
        self.assertGreaterEqual(seconds, 0)


if __name__ == "__main__":
    unittest.main()
