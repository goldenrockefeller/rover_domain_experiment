# distutils: language = c++
# distutils: include_dirs = RANDOM_INCLUDE
# distutils: sources = RANDOM_CPP
# distutils: extra_compile_args = -std=c++11

# disls: libraries = NP_RANDOM_LIB
# ditls: include_dirs = NP_INCLUDE RANDOM_INCLUDE
# diils: library_dirs = NP_RANDOM_PATH
# dils: sources = RANDOM_CPP


# import numpy as np
# from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
# from numpy.random cimport bitgen_t
# from numpy.random import PCG64
# from numpy.random.c_distributions cimport random_standard_normal
#
#
# # Random numpy c-api from https://numpy.org/doc/stable/reference/random/extending.html
# cdef const char *capsule_name = "BitGenerator"
# cdef bitgen_t *rng
# rnd_bitgen = PCG64()
# capsule = rnd_bitgen.capsule
# # Optional check that the capsule if from a BitGenerator
# if not PyCapsule_IsValid(capsule, capsule_name):
#     raise ValueError("Invalid pointer to anon_func_state")
# # Cast the pointer
# rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
#
# cpdef double random_normal() except *:
#     return random_standard_normal(rng)

cdef extern from "random_normal_mt19937_64.hpp":
    cdef cppclass RandomNormal:
            RandomNormal() except +
            double get()
#
# cdef class CyRandomNormal:
#     cdef RandomNormal rand
#
#     def __cinit__(self):
#         self.rand =  RandomNormal()
#
# rand = CyRandomNormal()

cdef RandomNormal rand = RandomNormal()

cpdef double random_normal() except *:
    return rand.get()