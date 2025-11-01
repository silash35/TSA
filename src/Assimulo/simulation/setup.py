import numpy as np

from . import constants as C

n_z = 20  # Number of discrete points in space
z = np.linspace(0, C.L, n_z)  # Spatial grid
dz = z[1] - z[0]  # Spatial step size
