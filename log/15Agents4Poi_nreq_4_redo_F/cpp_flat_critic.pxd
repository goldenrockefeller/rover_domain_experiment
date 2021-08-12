
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
        double eval(const valarray[double]&) except +
        valarray[double] grad_wrt_parameters(const valarray[double]&, double) except +


    cdef cppclass FlatNetworkOptimizer:
        double time_horizon
        double epsilon
        double learning_rate
        int learning_mode

    cdef cppclass Approximator:
        double eval(const Experience&) except +
        void update(const vector[Experience] &) except +
        void update(const Experience &, double) except +

    cdef cppclass FlatNetworkApproximator:
        FlatNetworkOptimizer optimizer

        FlatNetworkApproximator(size_t, size_t) except +
        double eval(const Experience&) except +
        void update(const vector[Experience] &) except +
        void update(const Experience &, double) except +

    cdef cppclass MonteFlatNetworkApproximator(FlatNetworkApproximator):
        FlatNetworkOptimizer optimizer

        MonteFlatNetworkApproximator(size_t, size_t) except +
        double eval(const Experience&) except +
        void update(const vector[Experience] &) except +
        void update(const Experience &, double) except +

    cdef cppclass DiscountFlatNetworkApproximator(FlatNetworkApproximator):
        FlatNetworkOptimizer optimizer

        DiscountFlatNetworkApproximator(size_t, size_t) except +
        double eval(const Experience&) except +
        void update(const vector[Experience] &) except +

    cdef cppclass QFlatNetworkApproximator(FlatNetworkApproximator):
        FlatNetworkOptimizer optimizer

        QFlatNetworkApproximator(size_t, size_t) except +
        double eval(const Experience&) except +
        void update(const vector[Experience] &) except +

    cdef cppclass UFlatNetworkApproximator(FlatNetworkApproximator):
        FlatNetworkOptimizer optimizer

        UFlatNetworkApproximator(size_t, size_t) except +
        double eval(const Experience&) except +
        void update(const vector[Experience] &) except +

    cdef cppclass UqFlatNetworkApproximator(FlatNetworkApproximator):
        UqFlatNetworkApproximator(shared_ptr[UFlatNetworkApproximator], shared_ptr[QFlatNetworkApproximator]) except +

        double eval(const Experience&) except +
        void update(const vector[Experience] &) except +

