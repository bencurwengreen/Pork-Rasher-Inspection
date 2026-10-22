"""RGB feature extraction for whole images and image regions."""

import numpy as np


def extract_image_rgb_features(image_path):
    """Calculate six image-level RGB statistics for a validation image.

    Parameters
    ----------
    image_path : str or pathlib.Path
        Location of the image to analyse.

    Returns
    -------
    dict
        Mean and standard deviation for each RGB colour channel.

    Raises
    ------
    ValueError
        If OpenCV cannot load the requested image.

    Notes
    -----
    OpenCV reads images in BGR order, so the loaded pixels are converted
    to RGB before calculating features to match the training feature set.
    """

    import cv2

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # OpenCV loads images as BGR
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    red = image_rgb[:, :, 0]
    green = image_rgb[:, :, 1]
    blue = image_rgb[:, :, 2]

    return {
        "mean_red": np.mean(red),
        "mean_green": np.mean(green),
        "mean_blue": np.mean(blue),
        "std_red": np.std(red),
        "std_green": np.std(green),
        "std_blue": np.std(blue)
    }


def extract_region_rgb_features(patch_rgb):
    """
    Calculate RGB colour statistics for an image patch.

    Parameters
    ----------
    patch_rgb : numpy.ndarray
        RGB image patch with dimensions
        (height, width, 3).

    Returns
    -------
    dict
        Dictionary containing the mean and standard deviation
        of the red, green and blue colour channels.
    """

    # Reshape the patch into an array where each row
    # represents one pixel and each column represents
    # an RGB colour channel.
    pixels = patch_rgb.reshape(-1, 3)

    # Calculate the mean intensity of each colour channel.
    mean_rgb = pixels.mean(axis=0)

    # Calculate the variation in intensity of each channel.
    std_rgb = pixels.std(axis=0)

    # Return the calculated features using descriptive names.
    return {
        "mean_r": mean_rgb[0],
        "mean_g": mean_rgb[1],
        "mean_b": mean_rgb[2],
        "std_r": std_rgb[0],
        "std_g": std_rgb[1],
        "std_b": std_rgb[2]
    }


def extract_whole_image_rgb_features(image_rgb):
    """Calculate whole-image RGB means and population standard deviations.

    Parameters
    ----------
    image_rgb : numpy.ndarray
        Non-empty numeric RGB array of shape (height, width, 3). Project
        images use uint8 intensities from 0 to 255. Values are used as supplied:
        no channel conversion, scaling, clipping or shape validation is applied.

    Returns
    -------
    dict of str to numpy.floating
        Keys in predictor order: mean_red, mean_green, mean_blue, std_red,
        std_green and std_blue. Standard deviations use ddof=0. All values
        retain the input intensity units, normally 8-bit RGB levels.

    Raises
    ------
    ValueError
        If the array cannot be reshaped into groups of three values.
    TypeError
        If the input dtype does not support NumPy mean or standard deviation.

    Notes
    -----
    Callers must enforce the documented shape and non-empty input. An empty
    array produces NaN statistics and NumPy warnings; some incorrectly shaped
    arrays can still be reshaped without an exception. Uses the same feature
    convention as Notebook 02's training table.
    """
    # Reshape RGB pixels so rows represent pixels, columns represent R/G/B.
    pixels = image_rgb.reshape(-1, 3)
    mean_rgb = pixels.mean(axis=0)
    std_rgb = pixels.std(axis=0)

    return {
        'mean_red': mean_rgb[0],
        'mean_green': mean_rgb[1],
        'mean_blue': mean_rgb[2],
        'std_red': std_rgb[0],
        'std_green': std_rgb[1],
        'std_blue': std_rgb[2],
    }
