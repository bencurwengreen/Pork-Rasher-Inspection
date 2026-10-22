"""Model prediction aggregation and image-level evaluation helpers."""

import time

import numpy as np
import pandas as pd


def regional_predictions_to_image_labels(region_model, max_defects, model_name, X_region_test, region_features_df, predictions_df):
    """Predict patches and aggregate defect counts in the requested image order.

    Parameters
    ----------
    region_model : estimator
        Fitted estimator exposing predict with binary patch labels.
    max_defects : int
        Images fail when the predicted defect count exceeds this threshold.
    model_name : str
        Model description used in validation messages.
    X_region_test : array-like
        Patch features in the same row order as region_features_df.
    region_features_df : pandas.DataFrame
        Patch records containing an image_id column.
    predictions_df : pandas.DataFrame
        Image records whose image_id column defines the output order.

    Returns
    -------
    tuple
        PASS/FAIL label array, defect count array and elapsed seconds.
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
