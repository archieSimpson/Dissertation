import numpy as np
import matplotlib.pyplot as plt


def plot_bell_curve(positions):
    distances = np.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2)

    plt.figure(figsize=(8, 5))
    plt.hist(distances, bins=30, density=True, alpha=0.6, label="Observed distribution")

    mu = np.mean(distances)
    sigma = np.std(distances)

    x = np.linspace(distances.min(), distances.max(), 300)
    gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )

    plt.plot(x, gaussian, linewidth=2, label="Gaussian fit")
    plt.title("Bell Curve of Sheep Distribution")
    plt.xlabel("Distance from centre")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_gaussian_xy(positions):
    x_vals = positions[:, 0]

    plt.figure(figsize=(8, 5))
    plt.hist(x_vals, bins=30, density=True, alpha=0.6, label="Observed X distribution")

    mu = np.mean(x_vals)
    sigma = np.std(x_vals)

    x = np.linspace(x_vals.min(), x_vals.max(), 300)
    gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )

    plt.plot(x, gaussian, linewidth=2, label="Gaussian fit")
    plt.title("Gaussian Distribution of Sheep X Positions")
    plt.xlabel("X value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()