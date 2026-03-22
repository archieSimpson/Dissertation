import numpy as np

def compute_statistics(positions):
    distances = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)

    stats = {
        "mean_x": np.mean(positions[:, 0]),
        "mean_y": np.mean(positions[:, 1]),
        "var_x": np.var(positions[:, 0]),
        "var_y": np.var(positions[:, 1]),
        "mean_r": np.mean(distances),
        "var_r": np.var(distances),
    }

    return stats


def mean_squared_displacement(history):
    msd = []

    for pos in history:
        squared = pos[:, 0]**2 + pos[:, 1]**2
        msd.append(np.mean(squared))

    return msd