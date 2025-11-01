# %% Setup
import numpy as np
import simulation.constants as C
from assimulo.problem import Implicit_Problem
from assimulo.solvers import IDA
from lib.plot import plot
from simulation.constants import ng
from simulation.initial import y0
from simulation.model import rhs
from simulation.setup import n_z, z

tf = 98  # C.Tfeed  # Final simulation time
n_t = int(tf // 10)  # Number of discrete points in time
t = np.linspace(0, tf, n_t)
dt = t[1] - t[0]

# %%
yd0 = np.ones_like(y0)
res = rhs(t[0], y0, yd0)

# %%
problem = Implicit_Problem(
    rhs,
    t0=0.0,
    y0=y0,
    yd0=yd0,
    name="PSA",
)

algvar = np.ones_like(y0, dtype=bool)  # True - differential, False - algebraic
offset = 0

# Cg
for i in range(ng):
    algvar[offset] = False
    offset += n_z
    algvar[offset - 1] = False

offset += n_z * ng  # q

# Tg
algvar[offset] = False
offset += n_z
algvar[offset - 1] = False

offset += n_z  # Tp
offset += n_z  # Tw

# Apartir daqui é tudo algébrico
algvar[offset:] = False
problem.algvar = algvar

# %%
# Define an explicit solver
solver = IDA(problem)  # Create a IDA solver

# solver.atol = 1e-2  # Default 1e-6
# solver.rtol = 1e-2  # Default 1e-6
solver.verbosity = 10

solver.make_consistent("IDA_YA_YDP_INIT")

# Simulate
t, y, yd = solver.simulate(tf, ncp_list=t)


# %%
n_t = np.array(t).shape[0]
offset = 0

Cg = y[:, offset : offset + n_z * ng].reshape(n_t, ng, n_z)
offset += n_z * ng

q = y[:, offset : offset + n_z * ng].reshape(n_t, ng, n_z)
offset += n_z * ng

Tg = y[:, offset : offset + n_z]
offset += n_z
Tp = y[:, offset : offset + n_z]
offset += n_z
Tw = y[:, offset : offset + n_z]
offset += n_z

Cs = y[:, offset : offset + n_z * ng].reshape(n_t, ng, n_z)
offset += n_z * ng

u0 = y[:, offset : offset + n_z]
offset += n_z

Py = y[:, offset : offset + 2]
offset += 2

Cgt = np.sum(Cg, axis=1)

P = np.zeros((n_t, n_z))
P[:, 0] = Py[:, 0]
P[:, 1:-1] = (Cgt * Tg * C.Rg)[:, 1:-1]
P[:, -1] = Py[:, 1]


# %%
plot(t, Cg[:, 0, :], z, "Cg_CO2")
plot(t, Cg[:, 1, :], z, "Cg_CO")
plot(t, Cg[:, 2, :], z, "Cg_H2")

plot(t, q[:, 0, :], z, "q_CO2")
plot(t, q[:, 1, :], z, "q_CO")
plot(t, q[:, 2, :], z, "q_H2")

plot(t, Tg, z, "Tg")
plot(t, Tp, z, "Tp")
plot(t, Tw, z, "Tw")

plot(t, Cs[:, 0, :], z, "Cs_CO2")
plot(t, Cs[:, 1, :], z, "Cs_CO")
plot(t, Cs[:, 2, :], z, "Cs_H2")

plot(t, u0, z, "u0")

plot(t, P, z, "P")
