"""Dataset preparation checks using synthetic files in temporary directories.

These tests never use or alter the project's actual dataset.
"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_clean_dataset import main, plan_dataset


class DatasetPreparationTests(unittest.TestCase):
    """Check copy isolation, duplicate policy and refusal conditions."""

    def make_source(self, root, conflicting_labels=False, cross_split=False):
        """Write three tiny synthetic COCO splits under a temporary root."""
        source = root / "raw"
        for split in ("train", "valid", "test"):
            folder = source / split
            folder.mkdir(parents=True)
            names = ["b.bin", "a.bin"] if split == "train" else ["one.bin"]
            images = []
            annotations = []
            for image_id, name in enumerate(names, start=1):
                content = b"train" if split == "train" or cross_split else split.encode()
                (folder / name).write_bytes(content)
                images.append({"id": image_id, "file_name": name})
                annotations.append({
                    "id": image_id, "image_id": image_id,
                    "category_id": 2 if conflicting_labels and image_id == 2 else 1,
                    "bbox": [0, 0, 10, 10],
                })
            (folder / "_annotations.coco.json").write_text(
                json.dumps({"images": images, "annotations": annotations,
                            "categories": [{"id": 1, "name": "defect"}]}),
                encoding="utf-8",
            )
        (source / "README.dataset.txt").write_text("Dataset attribution", encoding="utf-8")
        return source

    def invoke(self, source, output, apply=False):
        """Run the entry point only against explicitly supplied temporary paths."""
        args = ["prepare_clean_dataset.py", "--source-dir", str(source),
                "--output-dir", str(output)]
        if apply:
            args.append("--apply")
        with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
            main()

    def test_preview_is_read_only_and_keeps_lexicographically_first_copy(self):
        # Expected: retain a.bin despite its position after b.bin in COCO order;
        # exclude only the redundant training record and create no output copy.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            output = root / "cleaned"
            datasets, report = plan_dataset(source)
            self.assertEqual([item["file_name"] for item in datasets["train"]["images"]], ["a.bin"])
            self.assertEqual(report["splits"]["train"]["excluded_duplicates"][0]["filename"], "b.bin")
            self.assertEqual(len(datasets["train"]["annotations"]), 1)
            self.invoke(source, output)
            self.assertFalse(output.exists())
            self.assertTrue((source / "train/b.bin").exists())

    def test_apply_preserves_every_source_byte_and_records_the_copy(self):
        # Expected: source bytes are unchanged; validation/test JSON is logically
        # identical; cleaned output contains attribution and an accurate report.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            before = {p.relative_to(source): p.read_bytes()
                      for p in source.rglob("*") if p.is_file()}
            output = root / "cleaned"
            self.invoke(source, output, apply=True)
            after = {p.relative_to(source): p.read_bytes()
                     for p in source.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertTrue((output / "train/a.bin").is_file())
            self.assertFalse((output / "train/b.bin").exists())
            for split in ("valid", "test"):
                path = Path(split) / "_annotations.coco.json"
                self.assertEqual(json.loads(before[path]),
                                 json.loads((output / path).read_text(encoding="utf-8")))
            report = json.loads((output / "preprocessing_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["splits"]["train"]["images_before"], 2)
            self.assertEqual(report["splits"]["train"]["images_after"], 1)
            self.assertEqual((output / "README.dataset.txt").read_bytes(),
                             before[Path("README.dataset.txt")])

    def test_conflicting_annotations_stop_before_output_creation(self):
        # Expected: identical pixels with disagreeing labels require manual review.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root, conflicting_labels=True)
            output = root / "cleaned"
            with self.assertRaisesRegex(ValueError, "annotation conflict"):
                self.invoke(source, output, apply=True)
            self.assertFalse(output.exists())

    def test_cross_split_duplicates_stop_before_output_creation(self):
        # Expected: possible split leakage is reported, never silently cleaned.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root, cross_split=True)
            output = root / "cleaned"
            with self.assertRaisesRegex(ValueError, "across splits"):
                self.invoke(source, output, apply=True)
            self.assertFalse(output.exists())

    def test_existing_output_is_never_overwritten(self):
        # Expected: refusal preserves any pre-existing destination content.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            output = root / "cleaned"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.invoke(source, output, apply=True)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_output_cannot_overlap_source(self):
        # Expected: neither the raw folder nor a folder inside it can be output.
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_source(Path(directory))
            for output in (source, source / "cleaned", source.parent):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.invoke(source, output, apply=True)


if __name__ == "__main__":
    unittest.main()
