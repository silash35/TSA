import numpy as np


def dydz(y: np.ndarray, dz: float) -> np.ndarray:
    """
    Computes the first derivative of y with respect to z.
    """
    dydz = np.zeros_like(y)

    dydz[0] = (-3 * y[0] + 4 * y[1] - y[2]) / (2 * dz)  # Forward at inlet
    dydz[1:-1] = (y[2:] - y[:-2]) / (2 * dz)  # Central
    dydz[-1] = (3 * y[-1] - 4 * y[-2] + y[-3]) / (2 * dz)  # Backward at outlet
    return dydz


def d2ydz2(y: np.ndarray, dz: float) -> np.ndarray:
    d2ydz2 = np.zeros_like(y)

    d2ydz2[0] = (2 * y[0] - 5 * y[1] + 4 * y[2] - y[3]) / dz**2  # Forward at inlet
    d2ydz2[1:-1] = (y[2:] - 2 * y[1:-1] + y[:-2]) / dz**2  # Central
    d2ydz2[-1] = (
        2 * y[-1] - 5 * y[-2] + 4 * y[-3] - y[-4]
    ) / dz**2  # Backward at outlet
    return d2ydz2
