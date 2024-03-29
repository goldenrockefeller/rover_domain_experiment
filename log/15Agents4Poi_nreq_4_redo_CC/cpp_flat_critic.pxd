
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, unique_ptr

cdef extern from "<valarray>" namespace "std" nogil:
    cdef cppclass valarray[T]:
        valarray() except +
        void resize (size_t) except +
        size_t size() const
        valarray operator= (const valarray&)
        T& operator[] (size_t)

cdef extern from "cpp_core/flat_network_approximator.hpp" namespace "goldenrockefeller::policyopt" nogil:
    cdef cppclass Experience:
        valarray[double] observation
        valarray[double] action
        double reward

    cdef cppclass FlatNetwork:
        double leaky_scale

        FlatNetwork(size_t, size_t) except +
        unique_ptr[FlatNetwork] copy() except +
        valarray[double] parameters() except +
        void set_parameters(const valarray[double]&) except +
        size_t n_parameters() except +
        double eval(const valarray[double]&) except +
        valarray[double] grad_wrt_parameters(const valarray[double]&, double) except +

    cdef cppclass Approximator:
        double eval(const Experience&) except +
        void trajectory_update "update"(const vector[Experience] &) except +
        void update(const Experience &, double) except +



    cdef cppclass FlatNetworkApproximator(Approximator):
        double learning_rate
        shared_ptr[FlatNetwork] flat_network
        double grad_disturbance_factor
        double momentum_sustain
        double eps
        bint using_conditioner
        double conditioner_time_horizon

        FlatNetworkApproximator(size_t, size_t) except +

