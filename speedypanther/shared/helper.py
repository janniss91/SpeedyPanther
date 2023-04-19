import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalize the values in a numpy array.
    """
    return (x - np.min(x)) / (np.max(x) - np.min(x))
