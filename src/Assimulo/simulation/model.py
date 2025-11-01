import numpy as np

from . import constants as C
from .constants import ng
from .derivatives import partial, partial2
from .setup import n_z


def rhs(t: float, y: np.ndarray, yd: np.ndarray):
    print("Solving for t=", t)

    # --- Variáveis diferenciais ---
    offset = 0
    # Concentração no gás
    Cg = y[offset : offset + ng * n_z].reshape(ng, n_z)
    yd_Cg = yd[offset : offset + ng * n_z].reshape(ng, n_z)
    offset += ng * n_z

    # Concentração adsorvida
    q = y[offset : offset + ng * n_z].reshape(ng, n_z)
    yd_q = yd[offset : offset + ng * n_z].reshape(ng, n_z)
    offset += ng * n_z

    # Temperatura do gás
    Tg = y[offset : offset + n_z]
    yd_Tg = yd[offset : offset + n_z]
    offset += n_z

    # Temperatura das partículas
    Tp = y[offset : offset + n_z]
    yd_Tp = yd[offset : offset + n_z]
    offset += n_z

    # Temperatura da parede
    Tw = y[offset : offset + n_z]
    yd_Tw = yd[offset : offset + n_z]
    offset += n_z

    # --- Variáveis algébricas ---
    Cs = y[offset : offset + ng * n_z].reshape(ng, n_z)
    offset += ng * n_z

    Py = y[offset : offset + 2]
    offset += 2
    P = np.zeros(n_z)
    P[0] = Py[0]
    P[-1] = Py[1]

    u0 = y[offset : offset + n_z]
    offset += n_z

    # --- Algebraic Equations ---
    # Explicit
    Cgt = np.sum(Cg, axis=0)

    yi = np.zeros((ng, n_z))
    for i in range(ng):
        yi[i, :] = Cg[i] / Cgt

    Mw = C.Mwi @ yi

    keq = np.zeros((ng, n_z))
    for i in range(ng):
        keq[i, :] = C.kinf[i] * np.exp(C.MDH1[i] / (C.Rg * Tp))

    Pmpbar = np.zeros((ng, n_z))
    for i in range(ng):
        Pmpbar[i, :] = Cs[i] * C.Rg * Tp / 1e5

    q_star = np.zeros((ng, n_z))
    for i in range(ng):
        q_star[i, :] = (
            C.qsat[i]
            * (keq[i] * Pmpbar[i])
            / (1 + (np.sum(np.abs(keq * Pmpbar), axis=0)))
        )

    P[1:-1] = (Cgt * Tg * C.Rg)[1:-1]

    # Residuals (Implicit)
    res = np.ones_like(y) * 999
    offset = 0

    res[offset : offset + n_z] = -partial(P) - (
        150 * C.visc * (1 - C.epsb) ** 2 / (C.epsb**3 * C.dp**2) * u0
        + 1.75 * (1 - C.epsb) / (C.epsb**3 * C.dp) * Cgt * Mw * u0 * np.abs(u0)
    )
    offset += n_z

    for i in range(ng):
        res[offset : offset + n_z] = 15 * C.dpi[i] / P * C.Phigh / (C.Rp**2) * (
            q_star[i] - q[i]
        ) - (C.apm * C.kf * (Cg[i] - Cs[i]) / C.roap)
        offset += n_z

    # --- Equações Diferenciais ---
    for i in range(ng):
        # O -2 é pq a equação não é valida nas bordas, segundo o código do gPROMS
        res[offset : offset + n_z - 2] = (
            (C.epsb * C.dax * (partial(Cgt) * partial(yi[i]) + Cgt * partial2(yi[i])))
            - partial(u0 * Cg[i])
            - C.epsb * yd_Cg[i]
            - (1 - C.epsb) * C.apm * C.kf * (Cg[i] - Cs[i])
        )[1:-1]
        offset += n_z - 2

    res[offset : offset + n_z - 2] = (
        (C.λ * partial2(Tg))
        - u0 * Cgt * C.Cpmix * partial(Tg)
        + C.epsb * C.Rg * Tg * yd_Cg.sum(axis=0)
        - (1 - C.epsb) * C.apm * C.hf * (Tg - Tp)
        - 4 * C.hw / C.Dw * (Tg - Tw)
        - C.epsb * Cgt * C.Cvmix * yd_Tg
    )[1:-1]
    offset += n_z - 2

    for i in range(ng):
        res[offset : offset + n_z] = (
            15 * C.dpi[i] / P * C.Phigh / C.Rp**2 * (q_star[i] - q[i]) - yd_q[i]
        )
        offset += n_z

    res[offset : offset + n_z] = (
        C.rob * (np.sum(C.MDH1[:, None] * yd_q, axis=0))
        + (1 - C.epsb) * C.apm * C.hf * (Tg - Tp)
        - (1 - C.epsb)
        * (C.roap * np.sum(q * C.Cvi[:, None], axis=0) + C.roap * C.Cps)
        * yd_Tp
    )
    offset += n_z

    res[offset : offset + n_z] = (
        C.alfaw * C.hw * (Tg - Tw)
        - C.alfawl * C.U * (Tw - C.Tinf)
        - C.row * C.Cpw * yd_Tw
    )
    offset += n_z

    # --- Condições de Contorno (Feed) ---
    for i in range(ng):
        # Inlet
        res[offset] = C.yin_feed[i] * C.Cinlet - (
            Cg[i, 0] - C.epsb * C.dax / C.u0inlet * partial(Cg[i])[0]
        )
        offset += 1

        # Outlet
        res[offset] = partial(Cg[i])[-1]
        offset += 1

    if P[-1] < C.Phigh:
        res[offset] = u0[0] * Cgt[0] - 0.5 * C.u0inlet * C.Cinlet
        offset += 1
        res[offset] = u0[-1]
        offset += 1
    else:
        res[offset] = P[-1] - C.Phigh
        offset += 1
        res[offset] = u0[0] * Cgt[0] - C.u0inlet * C.Cinlet
        offset += 1

    res[offset] = C.Tinlet - (
        Tg[0] - C.λ / (C.u0inlet * C.Cinlet * C.Cpmix) * partial(Tg)[0]
    )
    offset += 1

    res[offset] = partial(Tg)[-1]
    offset += 1

    return res
