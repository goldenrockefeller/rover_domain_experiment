

cdef extern from "<valarray>" namespace "std" nogil:
    cdef cppclass valarray[T]:
        valarray() except +
        void resize (size_t) except +
        size_t size() const
        valarray operator= (const valarray&)
        T& operator[] (size_t)

cdef extern from "cpp_core/relu_network_approximator.hpp" namespace "goldenrockefeller::policyopt" nogil:
     cdef cppclass ReluNetworkApproximator:
        double learning_rate

        ReluNetworkApproximator() except +

        ReluNetworkApproximator() except +
        ReluNetworkApproximator(size_t, size_t, size_t) except +

        valarray[double] eval(const valarray[double]&) except +
        void update(valarray[double], double feedback) except +


