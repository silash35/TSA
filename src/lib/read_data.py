import numpy as np
import pandas as pd

df = pd.read_csv("../export/dados.csv", index_col=0)

cycles = df.index.values
t_feed = df["t_feed"].values
t_rinse = df["t_rinse"].values
t_blow = df["t_blow"].values
t_purge = df["t_purge"].values

purity_H2 = df["purity_H2"].values
H2_CO_ratio = df["H2_CO_ratio"].values
purity_CO2 = df["purity_CO2"].values
recovery_CO2 = df["recovery_CO2"].values
productivity = df["productivity"].values

y = np.array(
    [
        purity_H2,
        H2_CO_ratio,
        purity_CO2,
        recovery_CO2,
        productivity,
    ]
).T
u = np.array([t_feed, t_rinse, t_blow, t_purge]).T

# ciclo de referência
ref_idx = 1
y_ref = y[ref_idx]
u_ref = u[ref_idx]

# --- Calcular desvios ---
y_desvio = y - np.tile(y_ref, (y.shape[0], 1))
u_desvio = u - np.tile(u_ref, (u.shape[0], 1))

# --- Limites ---

u_min = np.array([600, 187, 130, 80])
u_max = np.array([715, 265, 140, 115])

u_min_desvio = u_min - u_ref
u_max_desvio = u_max - u_ref
