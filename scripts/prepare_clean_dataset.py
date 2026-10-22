"""Prepare a separate dataset copy; the default invocation only previews changes.

Run from the repository root:
    python scripts/prepare_clean_dataset.py
    python scripts/prepare_clean_dataset.py --apply

Only byte-identical training images with identical annotation contents are
excluded. Validation and test records are retained unchanged. This script never
moves or deletes input files and refuses to overwrite an existing output tree.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")


def file_digest(path):
    """Return a SHA-256 hex digest using bounded-memory binary reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def annotation_signature(annotations):
    """Compare annotation contents independently of record IDs and ordering."""
    return sorted(
        json.dumps(
            {key: value for key, value in annotation.items()
             if key not in {"id", "image_id"}},
            sort_keys=True,
        )
        for annotation in annotations
    )


def plan_dataset(source_dir):
    """Read COCO records and file hashes; return copied data and an audit plan.

    Parameters
    ----------
    source_dir : pathlib.Path
        Existing directory containing train, valid and test COCO exports.

    Returns
    -------
    tuple of dict and dict
        Split COCO objects and a JSON-serialisable report containing before/after
        counts, retained image hashes, and excluded duplicate records.

    Raises
    ------
    ValueError
        If IDs or filenames repeat, paths escape a split, annotation references
        are invalid, duplicate labels disagree, or bytes repeat across splits.
    OSError
        If input annotations or images cannot be read. No files are written.
    """
    datasets = {}
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(source_dir),
        "rule": "Within train only: retain the lexicographically first filename "
                "in each SHA-256 group, provided annotation contents match.",
        "splits": {},
    }
    hash_splits = defaultdict(set)
    for split in SPLITS:
        folder = source_dir / split
        with (folder / "_annotations.coco.json").open(encoding="utf-8") as stream:
            coco = json.load(stream)
        images = coco["images"]
        annotations = coco["annotations"]
        ids = [image["id"] for image in images]
        names = [image["file_name"] for image in images]
        if not images or len(set(ids)) != len(ids) or len(set(names)) != len(names):
            raise ValueError(f"Empty image list or repeated IDs/filenames in {split}")
        by_image = defaultdict(list)
        for annotation in annotations:
            by_image[annotation["image_id"]].append(annotation)
        if not set(by_image).issubset(set(ids)):
            raise ValueError(f"Annotations reference nonexistent images in {split}")
        groups = defaultdict(list)
        for image in sorted(images, key=lambda record: record["file_name"]):
            path = (folder / image["file_name"]).resolve()
            if not path.is_relative_to(folder.resolve()):
                raise ValueError(f"Image path escapes {split}: {image['file_name']}")
            digest = file_digest(path)
            groups[digest].append(image)
            hash_splits[digest].add(split)
        removed = []
        excluded_ids = set()
        retained = []
        for digest, group in groups.items():
            keep = group[0]
            if split == "train" and len(group) > 1:
                signature = annotation_signature(by_image[keep["id"]])
                if any(annotation_signature(by_image[item["id"]]) != signature
                       for item in group[1:]):
                    raise ValueError(
                        f"Duplicate annotation conflict: {keep['file_name']}; "
                        "manual review is required and no copy has been created."
                    )
                for item in group[1:]:
                    excluded_ids.add(item["id"])
                    removed.append({
                        "image_id": item["id"], "filename": item["file_name"],
                        "retained_image_id": keep["id"],
                        "retained_filename": keep["file_name"], "sha256": digest,
                    })
            retained.extend({"image_id": item["id"],
                             "filename": item["file_name"], "sha256": digest}
                            for item in group if item["id"] not in excluded_ids)
        cleaned = dict(coco)
        cleaned["images"] = [item for item in images if item["id"] not in excluded_ids]
        cleaned["annotations"] = [item for item in annotations
                                  if item["image_id"] not in excluded_ids]
        datasets[split] = cleaned
        report["splits"][split] = {
            "images_before": len(images), "images_after": len(cleaned["images"]),
            "annotations_before": len(annotations),
            "annotations_after": len(cleaned["annotations"]),
            "excluded_duplicates": removed, "retained_images": retained,
            "duplicate_groups": sum(len(group) > 1 for group in groups.values()),
        }
    cross_split = [digest for digest, splits in hash_splits.items() if len(splits) > 1]
    if cross_split:
        raise ValueError(
            f"Found {len(cross_split)} byte-identical groups across splits. "
            "Resolve split leakage explicitly; no copy has been created."
        )
    return datasets, report


def main():
    """Preview a cleaning plan, or explicitly create a new cleaned dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/cleaned")
    parser.add_argument("--apply", action="store_true",
                        help="Create the output copy; without this, only preview.")
    args = parser.parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        parser.error("Source and output directories must not overlap.")
    if args.apply and output.exists():
        parser.error("Output already exists. Choose a new --output-dir; never overwrite.")
    datasets, report = plan_dataset(source)
    for split, info in report["splits"].items():
        print(f"{split}: {info['images_before']} -> {info['images_after']} images; "
              f"{len(info['excluded_duplicates'])} duplicate copies excluded")
        for item in info["excluded_duplicates"]:
            print(f"  Exclude {item['filename']}; retain {item['retained_filename']}")
    if not args.apply:
        print("Preview only: no files written. Add --apply to create the cleaned copy.")
        return
    # Reserve a new destination only after the entire input plan has passed checks.
    output.mkdir(parents=True, exist_ok=False)
    print(f"Creating {output}. If copying fails, treat this destination as incomplete.")
    for split, coco in datasets.items():
        target = output / split
        target.mkdir()
        planned_hashes = {
            item["image_id"]: item["sha256"]
            for item in report["splits"][split]["retained_images"]
        }
        for image in coco["images"]:
            destination = target / image["file_name"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / split / image["file_name"], destination)
            expected = planned_hashes[image["id"]]
            if file_digest(destination) != expected:
                raise OSError(f"Copied bytes differ from the planned input: {destination}")
        with (target / "_annotations.coco.json").open("w", encoding="utf-8") as stream:
            json.dump(coco, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    for notice in source.glob("README*"):
        if notice.is_file():
            shutil.copy2(notice, output / notice.name)
    report["output_directory"] = str(output)
    with (output / "preprocessing_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print("Complete. Raw inputs are unchanged; preprocessing_report.json records the copy.")


if __name__ == "__main__":
    main()
