import numpy as np


def bool_array_to_rgb(array: np.ndarray) -> np.ndarray:
    """
    Converts a boolean numpy array to an RGB tensor with equal values in each of the colors.

    :param array: 2d boolean numpy array
    :return: HxWx3 tensor
    """
    # assert array.dtype == np.bool, "array must be boolean"
    array = array.astype(np.uint8) * (2 ** 8 - 1)
    empty = np.zeros_like(array)
    return np.stack([array, empty, empty], axis=2)