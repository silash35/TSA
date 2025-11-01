import numpy as np

from . import constants as C
from .constants import ng
from .setup import n_z

n_y = (
    n_z * ng  # Cg_CO2, Cg_CO, Cg_H2
    + n_z * ng  # q_CO2, q_CO, q_H2
    + n_z  # Tg
    + n_z  # Tp
    + n_z  # Tw
    + n_z * ng  # Csi_CO2, Csi_CO, Csi
    + n_z  # u0
    + 2  # P
)

yi0 = np.zeros((ng, n_z))
yi0[2, :] = 1
Cgt0 = C.Phigh / C.Rg / C.Tinlet

y0 = np.zeros(n_y)
offset = 0

y0[offset : offset + 3 * n_z] = (yi0 * Cgt0).flatten()  # Cg0
offset += 3 * n_z

# q0
# gPROMS code says q0 = 0. But the gPROMS result says other thing
q0 = np.zeros((ng, n_z))
# q0[0, 0] = 0.36358097
# q0[1, 0] = 0.021925615
# q0[2, 0] = 0.22066247
# q0[2, 1:] = 0.2443054

y0[offset : offset + 3 * n_z] = q0.flatten()
offset += 3 * n_z

y0[offset : offset + n_z] = np.full(n_z, C.Tinlet)  # Tg0
offset += n_z

y0[offset : offset + n_z] = np.full(n_z, C.Tinlet)  # Tp0
offset += n_z

y0[offset : offset + n_z] = np.full(n_z, C.Tinlet)  # Tw0
offset += n_z

# --- Algebric ---
y0[offset:] = 0.01

# Cs0 = np.zeros((ng, n_z))
# Cs0[0, 0] = 26.013372
# Cs0[1, 0] = 19.076473
# Cs0[2, 0] = 144.16516
# Cs0[2, 1:] = 189.255

# y0[offset : offset + 3 * n_z] = Cs0.flatten()
# offset += 3 * n_z

# u00 = np.zeros(n_z)
# u00[0] = 0.0047688824
# u00[1] = 0.0016191108
# u00[2:] = 2.6963236e-12
# u00[-1] = -7.340326e-13

# y0[offset : offset + n_z] = u00
# offset += n_z

# y0[offset] = y0[offset + 1] = C.Phigh
# offset += 2
