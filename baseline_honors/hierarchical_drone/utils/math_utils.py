import numpy as np


def set_seed(seed: int):
    np.random.seed(seed)


def clip_norm(v, max_norm):
    n = np.linalg.norm(v)
    if n <= max_norm or n < 1e-9:
        return v
    return v * (max_norm / n)
