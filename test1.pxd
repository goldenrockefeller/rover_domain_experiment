# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

cdef extern from "test.hpp":
    cdef cppclass Test:
        Test() except +
    cdef cppclass Tester(Test):
        Tester() except +
    size_t type_code(Test* test)
    bint can_cast(Test* test)

cdef class PyTest:
    cdef Test* test

cdef class PyTester(PyTest):
    pass