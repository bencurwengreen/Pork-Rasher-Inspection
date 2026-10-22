"""Model prediction aggregation and image-level evaluation helpers."""

__docformat__ = "markdown"

import time

import numpy as np
import pandas as pd


def regional_predictions_to_image_labels(region_model, max_defects, model_name, X_region_test, region_features_df, predictions_df):
    """Predict patch classes and aggregate counts in the requested image order.

    ## Parameters
    - **region_model** (`estimator`):
        Fitted estimator whose ``predict`` method returns one binary class
        per patch: 0 for normal and 1 for defective.
    - **max_defects** (`int`):
        Maximum allowable number of predicted defective patches. An image
        is FAIL only when its count is strictly greater than this value;
        equality remains PASS. Callers should supply a non-negative integer.
    - **model_name** (`str`):
        Model description included in assertion messages.
    - **X_region_test** (`array-like of shape (n_patches, n_features)`):
        Patch predictors in the same row order as ``region_features_df``.
    - **region_features_df** (`pandas.DataFrame`):
        One row per patch, including an ``image_id`` column identifying
        the source image. Row order must match ``X_region_test``.
    - **predictions_df** (`pandas.DataFrame`):
        Image records containing an ``image_id`` column. This column
        determines the order of both returned arrays.

    ## Returns
    - **predicted_labels** (`numpy.ndarray of str`):
        One PASS or FAIL label per row of ``predictions_df``.
    - **counts** (`numpy.ndarray of int`):
        Predicted defective-patch counts in the same image order.
    - **elapsed** (`float`):
        Elapsed prediction and aggregation time in seconds. Excludes
        image loading and feature extraction performed before this call.

    ## Raises
    - **AssertionError**:
        If prediction and patch-row counts differ, predicted classes are
        not binary, or a requested image has no aggregated patch count.
        These checks are inactive when Python assertions are disabled.
    - **KeyError**:
        If either dataframe lacks the required ``image_id`` column.

    ## Notes
    Counts use predicted patch classes, never ground-truth defect labels.
    Estimator errors propagate to the caller. Inputs must contain one scalar
    prediction per patch; patch-count thresholds depend on patch size and
    image resolution.
    """
    start = time.perf_counter()

    # Obtain one prediction per 32×32 patch (0 = normal, 1 = defective).
    patch_predictions = np.asarray(region_model.predict(X_region_test))
    assert len(patch_predictions) == len(region_features_df)
    assert np.isin(patch_predictions, [0, 1]).all(), (
        f'{model_name} produced unexpected regional class labels.'
    )

    # Attach the predicted patch classes to their originating image IDs.
    patch_results = pd.DataFrame({
        'image_id': region_features_df['image_id'].to_numpy(),
        'predicted_defect': patch_predictions.astype(int),
    })

    # Count only predicted defects, never actual COCO defects.
    image_counts = patch_results.groupby('image_id')['predicted_defect'].sum()

    # Match the test-image order using image IDs rather than sorting filenames.
    aligned_counts = image_counts.reindex(predictions_df['image_id'])
    assert aligned_counts.notna().all(), f'Missing patch counts for {model_name}.'
    counts = aligned_counts.to_numpy(dtype=int)

    # An image fails only if predicted defective regions STRICTLY EXCEED the threshold.
    predicted_labels = np.where(counts > max_defects, 'FAIL', 'PASS')
    elapsed = time.perf_counter() - start
    return predicted_labels, counts, elapsed
