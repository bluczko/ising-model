cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport exp, fabs
from libc.stdlib cimport rand

cdef extern from "limits.h":
    int RAND_MAX


@cython.boundscheck(False)
@cython.wraparound(False)
cdef calc_energy(np.int8_t[:, :] system, int i, int j):
    cdef int n = system.shape[0]
    cdef int m = system.shape[1]

    return -2 * system[i, j] * (
        system[(i - 1) % n, j] +
        system[(i + 1) % n, j] +
        system[i, (j - 1) % m] +
        system[i, (j + 1) % m]
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def update_cells(np.int8_t[:, :] system, int epochs, float temperature):
    cdef int n = system.shape[0]
    cdef int m = system.shape[1]
    cdef int i, j

    for epoch in range(epochs):
        i = rand() % n
        j = rand() % m

        e = -1 * calc_energy(system, i, j)

        if e <= 0 or exp(-e / temperature) * RAND_MAX > rand():
            system[i, j] *= -1

    return np.array(system)


@cython.boundscheck(False)
@cython.wraparound(False)
def calc_internal_energy(np.int8_t[:, :] system):
    cdef int n = system.shape[0]
    cdef int m = system.shape[1]

    cdef long total_energy = 0

    for i in range(n):
        for j in range(m):
            total_energy += calc_energy(system, i, j)

    cdef float u = (1.0 / (n * m)) * total_energy
    cdef float u_sq = u * total_energy

    return u, u_sq


@cython.boundscheck(False)
@cython.wraparound(False)
def calc_heat_capacity(np.int8_t[:, :] system):
    cdef float u, u_sq

    u, u_sq = calc_internal_energy(system)

    return u_sq - u * u

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_magnetization(np.int8_t[:, :] system):
    cdef int n = system.shape[0]
    cdef int m = system.shape[1]

    cdef long total = 0

    for i in range(n):
        for j in range(m):
            total += system[i, j]

    return fabs(total / (n * m))
