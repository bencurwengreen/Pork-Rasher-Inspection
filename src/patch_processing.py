"""Image patch extraction and COCO annotation overlap labelling."""

import numpy as np


def extract_training_patches(image, patch_size=64):
    """
    Divide an RGB image into non-overlapping square or
    rectangular patches, including image boundaries.

    Parameters
    ----------
    image : PIL.Image.Image
        The original RGB image.

    patch_size : int
        Maximum width and height of each patch, in pixels.

    Returns
    -------
    patches_list : list of dictionaries
        Each dictionary contains the patch's RGB data,
        position and actual dimensions.
    """

    # Convert the PIL image into a NumPy array.
    image_array = np.array(image)

    # Retrieve image dimensions.
    height, width, channels = image_array.shape

    # Initialise the list used to store patches.
    patches_list = []

    # Iterate in patch_size-pixel steps (calls below pass PATCH_SIZE = 32).
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):

            # Ensure the patch does not extend beyond the image.
            x_end = min(x + patch_size, width)
            y_end = min(y + patch_size, height)

            # Extract the RGB data from the current region.
            patch = image_array[y:y_end, x:x_end]

            # Record the actual patch dimensions.
            actual_width = x_end - x
            actual_height = y_end - y

            # Store the patch information.
            patches_list.append({
                "x": x,
                "y": y,
                "width": actual_width,
                "height": actual_height,
                "rgb": patch
            })

    return patches_list


def calculate_patch_overlap(patch, bbox):
    """
    Calculate the fraction of a patch covered by a COCO bounding box.

    Parameters
    ----------
    patch : dict
        Dictionary containing the patch coordinates and dimensions.

    bbox : list
        COCO bounding box in [x, y, width, height] format.

    Returns
    -------
    float
        Fraction of the patch overlapped by the bounding box,
        between 0 and 1.
    """

    # Determine the patch's rectangular boundaries.
    px1 = patch["x"]
    py1 = patch["y"]
    px2 = px1 + patch["width"]
    py2 = py1 + patch["height"]

    # Convert COCO bounding-box coordinates into boundaries.
    bx1, by1, bw, bh = bbox

    bx2 = bx1 + bw
    by2 = by1 + bh

    # Calculate the boundaries of the intersection.
    intersection_x1 = max(px1, bx1)
    intersection_y1 = max(py1, by1)

    intersection_x2 = min(px2, bx2)
    intersection_y2 = min(py2, by2)

    # Calculate the intersection dimensions.
    intersection_width = max(
        0, intersection_x2 - intersection_x1
    )

    intersection_height = max(
        0, intersection_y2 - intersection_y1
    )

    # Calculate the intersection area.
    intersection_area = (
        intersection_width * intersection_height
    )

    # Calculate the total patch area.
    patch_area = patch["width"] * patch["height"]

    # Return the fraction of the patch inside the bounding box.
    return intersection_area / patch_area


def label_patch(patch, image_annotations, defect_threshold=0.20):
    """
    Assign a label to a patch based on overlap with COCO annotations.

    Parameters
    ----------
    patch : dict
        Dictionary containing patch coordinates and dimensions.

    image_annotations : list of dict
        All COCO annotations associated with the current image.

    defect_threshold : float, optional
        Minimum overlap ratio required for a patch to be labelled
        as a defect. Default is 0.20.

    Returns
    -------
    label : int
        Patch label:
        1  = defect
        0  = normal
        -1 = ambiguous

    max_overlap : float
        Largest overlap ratio between the patch and any annotation.

    matched_category_id : int or None
        Category ID of the annotation with the greatest overlap.
        Returns None if there is no overlap.
    """

    # Track the greatest overlap found for this patch.
    max_overlap = 0.0

    # Track the category associated with the largest overlap.
    matched_category_id = None

    # Compare this patch to every annotation in the image.
    for annotation in image_annotations:

        overlap = calculate_patch_overlap(patch, annotation["bbox"])

        # Keep only the largest overlap encountered.
        if overlap > max_overlap:
            max_overlap = overlap
            matched_category_id = annotation["category_id"]

    # Assign label based on the greatest overlap.
    if max_overlap >= defect_threshold:
        label = 1

    elif max_overlap == 0:
        label = 0

    else:
        label = -1

    return label, max_overlap, matched_category_id


def extract_patches(image, patch_size=32):
    """
    Divide an RGB image into non-overlapping patches.

    Partial patches are retained along image boundaries.

    Parameters
    ----------
    image : PIL.Image.Image
        Original RGB image.

    patch_size : int
        Maximum patch width and height in pixels.

    Returns
    -------
    list of dict
        Patch coordinates, dimensions and RGB pixel arrays.
    """

    # Convert the RGB image to a NumPy array.
    image_array = np.array(image)

    # Retrieve the image dimensions.
    height, width, channels = image_array.shape

    # Initialise the list of extracted patches.
    patches_list = []

    # Traverse the full image in patch-sized increments.
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):

            # Clip the final patches to the image boundaries.
            x_end = min(x + patch_size, width)
            y_end = min(y + patch_size, height)

            # Extract the patch's RGB pixels.
            patch_rgb = image_array[y:y_end, x:x_end]

            # Record the patch and its position.
            patches_list.append({
                "x": x,
                "y": y,
                "width": x_end - x,
                "height": y_end - y,
                "rgb": patch_rgb
            })

    return patches_list


def label_patch_class(patch, image_annotations, defect_threshold=0.20):
    """Assign a patch class using its largest COCO bounding-box overlap.

    Parameters
    ----------
    patch : dict
        Numeric x and y coordinates plus positive width and height, in pixels.
        All coordinates must use the same image reference frame as the boxes.
    image_annotations : iterable of dict
        Each annotation contains bbox=[x, y, width, height] in COCO pixel
        coordinates. An empty iterable is allowed; category IDs are not used.
    defect_threshold : float, optional
        Minimum fraction of patch area covered by one box to label a defect;
        expected range is (0, 1], with default 0.20. Not validated internally.

    Returns
    -------
    int
        1 (DEFECT) when the maximum overlap is at least the threshold;
        0 (NORMAL) when there is no overlap, including no annotations;
        -1 (AMBIGUOUS) when overlap is positive but below the threshold.
        Overlaps from separate boxes are not added together.

    Raises
    ------
    KeyError
        If a required patch or annotation field is missing.
    ValueError
        If a bounding box cannot be unpacked into four values.
    ZeroDivisionError
        If Python numeric patch dimensions give a zero patch area during
        overlap calculation. NumPy scalar dimensions can instead yield NaN.
    TypeError
        If coordinates do not support the required numeric operations.

    Notes
    -----
    Callers must validate dimensions and the threshold before calling. Invalid
    negative dimensions or out-of-range thresholds may yield meaningless labels.
    """

    # Find the greatest overlap with any annotation.
    max_overlap = max(
        (
            calculate_patch_overlap(patch, annotation["bbox"])
            for annotation in image_annotations
        ),
        default=0.0
    )

    # Apply the same labelling rules as Notebook 06.
    if max_overlap >= defect_threshold:
        return 1

    elif max_overlap == 0:
        return 0

    else:
        return -1


def iter_rgb_patches(image_rgb, patch_size=32):
    """Yield RGB patches in row-first order, retaining smaller edge patches.

    Parameters
    ----------
    image_rgb : numpy.ndarray
        RGB array of shape (height, width, 3). Project images use uint8 values
        from 0 to 255. Values and dtype are preserved without normalisation.
    patch_size : int, optional
        Positive maximum patch height and width in pixels; default is 32.
        Values larger than the image are allowed and yield a single patch
        for a non-empty image. Positivity is a caller precondition.

    Yields
    ------
    numpy.ndarray
        Non-overlapping views of shape (patch_height, patch_width, 3), with
        dimensions at most patch_size. Patches traverse left to right, then
        top to bottom. Editing a yielded view can change the original array.

    Raises
    ------
    ValueError
        If the input does not have three dimensions, or patch_size is zero.
    AssertionError
        If the channel count is not three and assertions are enabled.
    TypeError
        If patch_size cannot be interpreted as an integer.

    Notes
    -----
    Checks occur when the generator is iterated. Negative patch_size currently
    yields no patches rather than raising; zero-sized spatial dimensions also
    yield none. Callers must enforce positive dimensions and patch_size.
    """
    height, width, channels = image_rgb.shape
    assert channels == 3, 'Expected an RGB image with three channels.'

    # Use the same row-first traversal as Notebooks 06, 08, 12 and 13.
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            y_end = min(y + patch_size, height)
            x_end = min(x + patch_size, width)
            yield image_rgb[y:y_end, x:x_end]
