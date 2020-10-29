# distutils: language = c++
# distutils: extra_compile_args = -std=c++11
# distutils: sources = tester.cpp
from test1 cimport PyTest, PyTester
from libcpp.cast cimport dynamic_cast
from libcpp.memory cimport shared_ptr

cdef extern from "test.hpp":
    cdef cppclass Test:
        Test() except +
    cdef cppclass Tester(Test):
        Tester() except +
    size_t type_code(Test* test)
    bint can_cast(Test* test)

cdef class PyTest2(PyTest):
    def __cinit__(self):
        self.test = shared_ptr[Test](new Test())

cdef class PyTester2(PyTester):
    def __cinit__(self):
        self.test = shared_ptr[Test](new Tester())

cpdef py_type_code(PyTest py_test):
    return type_code(py_test.test.get())

cpdef py_can_cast(PyTest py_test):
    return can_cast(py_test.test.get())