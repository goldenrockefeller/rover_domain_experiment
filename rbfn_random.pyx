# distutils: libraries = NP_RANDOM_LIB
# distutils: include_dirs = NP_INCLUDE
# distutils: library_dirs = NP_RANDOM_PATH


import numpy as np
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from numpy.random cimport bitgen_t
from numpy.random import PCG64
from numpy.random.c_distributions cimport random_standard_normal


# Random numpy c-api from https://numpy.org/doc/stable/reference/random/extending.html
cdef const char *capsule_name = "BitGenerator"
cdef bitgen_t *rng
rnd_bitgen = PCG64()
capsule = rnd_bitgen.capsule
# Optional check that the capsule if from a BitGenerator
if not PyCapsule_IsValid(capsule, capsule_name):
    raise ValueError("Invalid pointer to anon_func_state")
# Cast the pointer
rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)

cpdef double random_normal() except *:
    return random_standard_normal(rng)