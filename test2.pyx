# distutils: language = c++
# distutils: extra_compile_args = -std=c++11
from test1 cimport PyTest, PyTester

cdef extern from "test.hpp":
    cdef cppclass Test:
        Test() except +
    cdef cppclass Tester(Test):
        Tester() except +
    size_t type_code(Test* test)
    bint can_cast(Test* test)

cdef class PyTest2(PyTest):
    pass

cdef class PyTester2(PyTester):
    pass

cpdef py_type_code(PyTest py_test):
    return type_code(py_test.test)

cpdef py_can_cast(PyTest py_test):
    return can_cast(py_test.test)