import numpy as np


def vec(x: float, y: float) -> np.ndarray:
    return np.array([x, y], dtype=np.float64)


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def safe_normalize(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    if n < 1e-8:
        return np.zeros(2, dtype=np.float64)
    return v / n


def limit(v: np.ndarray, max_mag: float) -> np.ndarray:
    n = norm(v)
    if n > max_mag:
        return v * (max_mag / n)
    return v


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return norm(a - b)


def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))