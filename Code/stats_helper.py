import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler


def compute_mean_and_std(dir_name: str) -> Tuple[np.ndarray, np.array]:
    """
    Compute the mean and the standard deviation of the dataset.

    Note: convert the image in grayscale and then scale to [0,1] before computing
    mean and standard deviation

    Hints: use StandardScalar (check import statement)

    Args:
    -   dir_name: the path of the root dir
    Returns:
    -   mean: mean value of the dataset (np.array containing a scalar value)
    -   std: standard deviation of th dataset (np.array containing a scalar value)
    """

    mean = None
    std = None

    ############################################################################
    # Student code begin
    ############################################################################
    scaler = StandardScaler()
    for i in os.listdir(dir_name):
        i = dir_name+"/"+i
        for j in os.listdir(i):
            j = i+"/"+j
            for path in os.listdir(j):
                path = j+"/"+path
                image = Image.open(path, 'r').convert('L')
                image = np.array(image).flatten()
                image = image / 255
                scaler.partial_fit(image.reshape((-1,1)))
    ############################################################################
    # Student code end
    ############################################################################
    return scaler.mean_ , scaler.scale_
