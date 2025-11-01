import matplotlib.pyplot as plt
import numpy as np


def plot(t: np.ndarray, y: np.ndarray, z: np.ndarray, ylabel: str):
    nz = y.shape[1]
    for i in range(nz):
        plt.plot(t, y[:, i], label=f"z={z[i]:.2f}")

    plt.xlabel("t")
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(f"./src/Assimulo/figures/{ylabel}.png")
    plt.close()
