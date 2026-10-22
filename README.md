# MMA3001 Pork Rasher Machine Learning Inspection

## Project overview

This project investigates machine-learning methods for classifying packaged pork rasher images as **PASS** or **FAIL**. COCO bounding-box annotations provide the reference labels: an image with one or more recorded defect annotations is treated as **FAIL**, and an image with no recorded defect annotation is treated as **PASS**.

The notebooks compare whole-image RGB statistics with models that classify smaller image regions. Evaluation uses accuracy, balanced accuracy, macro F1, confusion matrices and error inspection. An annotation-derived PASS label means no target defect was recorded; it does not establish that a package is defect-free or safe.

## Clone and open the project (Windows / VS Code)

Install Python, Git, and VS Code with the **Python** and **Jupyter** extensions. The installed Python interpreter used for the reference dependency snapshot is Python 3.14.7. All 12 unit tests passed with that snapshot on 22 October 2026. A clean installation and full notebook workflow rerun remain unverified.

In PowerShell, open the folder where you want the repository, then run:

```powershell
git clone https://github.com/bencurwengreen/Pork-Rasher-Inspection.git
Set-Location Pork-Rasher-Inspection
code .
```

If `code` is unavailable, use **File > Open Folder** in VS Code. Open its PowerShell terminal at the repository root (the folder containing this README).

## Set up the Python environment

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name mma3001-pork-rasher --display-name "MMA3001 Pork Rasher ML"
```

Select **MMA3001 Pork Rasher ML** as each notebook's kernel. These commands use the environment's interpreter directly, so PowerShell activation is unnecessary. Use the same environment for all notebooks. If the environment is recreated, rerun installation and kernel registration.

`requirements.txt` declares runtime dependencies, including **scikit-learn >= 1.7**, which is required for the sample-weighted MLP training. Jinja2 supports pandas styled tables. `requirements-reference.txt` records the versions used for the passing unit-test run. It has not been validated by a clean installation or a full notebook execution. To try that exact snapshot instead, install it with the environment's Python and `-m pip install -r requirements-reference.txt`.

## Download and prepare the dataset

Download [Roboflow dataset version 4](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/dataset/4) as **COCO JSON**. Extract its `train`, `valid` and `test` folders into `data/raw/`. Each split must contain its images and `_annotations.coco.json`. Preserve the supplied dataset attribution files.

The export metadata describes 3,549 images, resized to 720 x 540 pixels with auto-orientation and exposure augmentation. The project's earlier preprocessing removed two duplicate training images, producing **3,347 train, 120 validation and 80 test images (3,547 total)**. The two-image difference is preprocessing, not a missing download. If your existing raw folder has already been cleaned, preparation may exclude zero additional duplicates.

Preparation is separate from Notebook 00. From the repository root, preview the plan:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_clean_dataset.py
```

After reviewing the listed exclusions, explicitly create the cleaned copy:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_clean_dataset.py --apply
```

The command leaves `data/raw/` unchanged and creates **`data/cleaned/`**, which all notebooks now read. Within each byte-identical training-image group, it retains the lexicographically first filename only when the associated annotation contents match. It removes the excluded image records and their annotations from the copied training JSON. Validation and test records are retained unchanged. Annotation conflicts or byte-identical images across splits stop preparation for manual review.

`data/cleaned/preprocessing_report.json` records the rule, UTC timestamp, input/output counts, retained image hashes and excluded filenames. Hashes detect identical bytes, not exposure-adjusted variants or other near duplicates. The command refuses to overwrite an existing output directory. For another run, select a different destination with `--output-dir`; using that alternative copy in notebooks also requires updating their dataset paths. A failed copy may leave an incomplete destination; only a successfully completed copy should be used. Allow enough disk space for a second dataset copy.

```text
data/
  raw/                  # Original download; preparation never modifies it
    train/
    valid/
    test/
  cleaned/              # Created by the explicit preparation command
    train/
    valid/
    test/
    preprocessing_report.json
  processed/            # Generated feature tables and evaluation outputs
models/                 # Trained models and decision configurations
notebooks/              # Run from this working directory
src/                    # Reusable Python functions
scripts/                # Explicit dataset preparation command
tests/                  # Unit tests
docs/                   # Generated HTML source documentation
```

Before running notebooks, create the output folders if needed:

```powershell
New-Item -ItemType Directory -Force data/processed, models | Out-Null
```

## Notebook execution

Dependencies are installed through the terminal setup above; Notebook 00 no longer installs packages or changes dataset files. Run the notebooks individually in numerical order, selecting **Restart Kernel and Run All** for each. Saved feature tables and models carry results between notebooks.

Set the kernel working directory to the project's **`notebooks/` folder**. Check it in a notebook with `from pathlib import Path; print(Path.cwd())`. If it is the repository root, run `%cd notebooks` before the notebook's other cells; after restarting the kernel, check again. Relative paths such as `../data/cleaned/train` and `../models` depend on this location.

| Order | Notebook | Purpose |
|---:|---|---|
| 00 | `00_validate_image_dataset.ipynb` | Validate the prepared images and annotations without changing dataset files. |
| 01 | `01_explore_coco_annotations.ipynb` | Explore annotations and inspect labelled and unlabelled examples. |
| 02 | `02_extract_rgb_data.ipynb` | Save whole-image RGB training features. |
| 03 | `03_train_baseline_classifier.ipynb` | Train whole-image logistic regression. |
| 04 | `04_validate_baseline_model.ipynb` | Validate whole-image logistic regression. |
| 05 | `05_extract_defect_region_features.ipynb` | Save labelled 32 x 32 patch features. |
| 06 | `06_train_defect_region_classifier.ipynb` | Train regional logistic regression. |
| 07 | `07_validate_defect_region_classifier.ipynb` | Validate regional logistic regression. |
| 08 | `08_train_neural_network_baseline.ipynb` | Train the whole-image neural network. |
| 09 | `09_validate_neural_network_baseline.ipynb` | Validate the whole-image neural network. |
| 10 | `10_experiment_decision_threshold.ipynb` | Explore probability thresholds on validation data. |
| 11 | `11_train_binary_regional_nn.ipynb` | Train the regional neural network and save its count rule. |
| 12 | `12_validate_binary_regional_nn.ipynb` | Validate the regional neural network. |
| 13 | `13_final_test_model_comparison.ipynb` | Evaluate configured models on held-out test images. |

Resolve notebook errors before continuing, then rerun from a restarted kernel. Notebook 08 saves `baseline_neural_network_identity.joblib`, and Notebooks 09, 10 and 13 all reuse that identity model. Notebook 10 changes only its decision threshold. The 0.45 comparison setting is not an established optimum for this model: validate and justify the threshold before final testing. Outputs in Notebooks 10 and 13 have been cleared because their model pairing changed; existing final-test CSVs describe the earlier experiment until explicitly regenerated.

## Final test evaluation

Run Notebook 13 after training and validation are complete and configurations are settled. Do not use test results to choose features, tune hyperparameters or select thresholds. Predictions and evaluation tables are saved under `data/processed/`. Saved notebook outputs are historical results until regenerated from the current code and environment.

## Tests and HTML documentation

The latest recorded run passed **12 of 12 unit tests** on **22 October 2026**, using Python 3.14.7 and the package versions in `requirements-reference.txt`. See the [dated HTML test report](reports/test-report-2026-10-22.html) and [environment record](reports/test-report-2026-10-22.json). This does not establish that the complete notebook workflow or a fresh installation succeeds.

The test suite also runs with Python's built-in runner, without pytest:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Optional development tools are declared separately:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

From the repository root, the following commands run tests and create a dated, self-contained HTML test report:

```powershell
New-Item -ItemType Directory -Force reports | Out-Null
$reportStamp = Get-Date -Format yyyy-MM-dd
.\.venv\Scripts\python.exe -m pytest tests --html="reports/test-report-$reportStamp.html" --self-contained-html
```

Tests can create temporary files and caches; these commands are not part of dataset preparation. A report records an actual test run and its outcome; the presence of a test file or pytest cache is not a passing report.

Open [the HTML source documentation](docs/index.html) locally. To regenerate it after source changes:

```powershell
.\.venv\Scripts\python.exe -B -m pdoc src scripts.prepare_clean_dataset -o docs
```

## Licensing and dataset attribution

The project's code is distributed under the **MIT License** in [LICENSE](LICENSE), copied from this repository's GitHub license file. Keep its copyright and permission notice with copies of the software.

The external **Pork Rasher Error (Packaging), version 4** dataset is separately identified as **CC BY 4.0** by its supplied [dataset notice](data/raw/README.dataset.txt). The MIT code license does not replace the dataset's license. Preserve the provider attribution and source link, identify version 4, link to [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), and describe preprocessing changes (here, removal of two duplicate training images). The preparation command copies root-level dataset README notices into the cleaned output.
