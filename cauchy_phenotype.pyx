cimport cython

from rockefeg.policyopt.evolution cimport DefaultPhenotype
from rockefeg.policyopt.evolution cimport init_DefaultPhenotype
from rockefeg.policyopt.map cimport BaseMap
from rockefeg.cyutil.array cimport DoubleArray

import numpy as np


@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class CauchyPhenotype(DefaultPhenotype):
    def __init__(self, policy):
        init_DefaultPhenotype(self, policy)

    cpdef copy(self, copy_obj = None):
        cdef CauchyPhenotype new_phenotype

        if copy_obj is None:
            new_phenotype = CauchyPhenotype.__new__(CauchyPhenotype)
        else:
            new_phenotype = copy_obj

        new_phenotype = DefaultPhenotype.copy(self, new_phenotype)

        return new_phenotype


    cpdef void mutate(self, args = None) except *:

        cdef BaseMap policy
        cdef DoubleArray parameters
        cdef object mutation
        cdef Py_ssize_t param_id
        cdef double[:] mutation_view
        cdef double mutation_step

        # TODO Optimize, getting mutation vector is done through python (numpy).
        policy = <BaseMap?>self.policy()

        parameters = <DoubleArray?> policy.parameters()

        mutation_step = np.random.standard_cauchy() * self.mutation_factor()

        # Get mutation direction.
        mutation = np.random.normal(size = (len(parameters), ))
        mutation /= np.linalg.norm(mutation)
        mutation *= mutation_step

        mutation_view = mutation

        for param_id in range(len(parameters)):
            parameters.view[param_id] += mutation_view[param_id]

        policy.set_parameters(parameters)