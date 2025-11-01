from lib.derivatives import d2ydz2, dydz

from .setup import dz


def partial(y):
    return dydz(y, dz)


def partial2(y):
    return d2ydz2(y, dz)
