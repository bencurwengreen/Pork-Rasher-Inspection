# MMA3001 Pork Rasher Machine Learning Inspection

## Project overview

This project investigates machine-learning methods for classifying packaged pork rasher images as **PASS** or **FAIL**. COCO bounding-box annotations provide the project’s reference labels: an image with one or more recorded defect annotations is treated as **FAIL**, and an image with no recorded defect annotation is treated as **PASS**.

The notebooks compare whole-image RGB statistics with models that classify smaller image regions. The project uses accuracy, balanced accuracy, macro F1, confusion matrices and error inspection to assess the models. A PASS prediction means that no target defect annotation was recorded; it is not a guarantee that a package is defect-free or safe.

## Dataset setup

Download the dataset from [Roboflow Universe](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/dataset/4):

1. Select dataset version **4** and choose **COCO JSON** as the export format.
2. Download the ZIP archive to your computer and extract it.
3. Place the `train`, `valid` and `test` folders in `data/raw/` in the project. Each folder should contain its images and an `_annotations.coco.json` file.

The expected paths are:

```text
data/
├── raw/
│   ├── train/
│   │   ├── _annotations.coco.json
│   │   └── image files...
│   ├── valid/
│   │   ├── _annotations.coco.json
│   │   └── image files...
│   └── test/
│       ├── _annotations.coco.json
│       └── image files...
├── processed/       # Generated feature tables and evaluation outputs
```

If the output folders are not already present, create `data/processed/` and `models/` in the project root before running the notebooks.

## Set up the Python environment

The following instructions use Windows PowerShell and VS Code. Open the project in VS Code, then create a virtual environment from the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install numpy pandas pillow matplotlib scikit-learn opencv-python joblib ipykernel
.\.venv\Scripts\python.exe -m ipykernel install --user --name mma3001-pork-rasher --display-name "MMA3001 Pork Rasher ML"
```

In VS Code, select **MMA3001 Pork Rasher ML** as the kernel for the notebooks. The Python and Jupyter extensions for VS Code must be installed. The commands use the environment’s Python executable directly, so PowerShell script activation is not required.

## Notebook execution

Run the notebooks individually and in numerical order. Before running each one, select **Restart Kernel and Run All**. Saved CSV files and trained models carry results between notebooks, so each notebook can start with a clean kernel.

Set the notebook working directory to the project’s `notebooks/` folder. The notebooks use relative paths such as `../data/raw/train` and `../models`; running them with the project root as the working directory may cause file-not-found errors.

| Order | Notebook | Purpose |
|---:|---|---|
| 00 | `00_validate_image_dataset.ipynb` | Check the dataset files, image formats, annotations and duplicates. |
| 01 | `01_explore_coco_annotations.ipynb` | Explore the COCO annotation structure and inspect labelled and unlabelled examples. |
| 02 | `02_extract_rgb_data.ipynb` | Create whole-image RGB mean and standard-deviation features and save the training feature table. |
| 03 | `03_train_baseline_classifier.ipynb` | Train the whole-image logistic-regression baseline. |
| 04 | `04_validate_baseline_model.ipynb` | Evaluate the baseline on the validation images. |
| 05 | `05_extract_defect_region_features.ipynb` | Extract labelled RGB features from 32 × 32 image patches. |
| 06 | `06_train_defect_region_classifier.ipynb` | Train the regional logistic-regression classifier. |
| 07 | `07_validate_defect_region_classifier.ipynb` | Evaluate regional predictions at the image level on validation data. |
| 08 | `08_train_neural_network_baseline.ipynb` | Train a whole-image neural-network model. |
| 09 | `09_validate_neural_network_baseline.ipynb` | Evaluate the whole-image neural network on validation data. |
| 10 | `10_experiment_decision_threshold.ipynb` | Examine how decision thresholds affect validation results. |
| 11 | `11_train_binary_regional_nn.ipynb` | Train the binary regional neural network and save its image-level defect-count rule. |
| 12 | `12_validate_binary_regional_nn.ipynb` | Evaluate the regional neural network on validation data. |
| 13 | `13_final_test_model_comparison.ipynb` | Compare the configured models on the held-out test images. |

The training and validation notebooks save intermediate feature tables and model files for later stages. If a notebook stops with an error, resolve it before continuing, then rerun that notebook from a restarted kernel.

## Final test evaluation

Run Notebook 13 only after the training and validation workflow is complete and the model configurations have been settled. It evaluates the configured models on the held-out test set. Do not use test-set results to choose features, tune model parameters or select decision thresholds.

## Generated files

Running the workflow creates processed feature tables under `data/processed/` and trained model files and configuration files under `models/`. Notebook 13 also saves final evaluation outputs under `data/processed/`. These are generated from the raw dataset and notebook code.
