import numpy as np

def array_convert(y) -> np.array:
    if isinstance(y, np.ndarray):
        return y
    return np.array(y)

