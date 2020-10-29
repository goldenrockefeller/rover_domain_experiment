# distutils: language = c++
# distutils: extra_compile_args = -std=c++11
# distutils: sources = tester.cpp test.cpp
from test1 cimport PyTest, PyTester
from libcpp.cast cimport dynamic_cast
from libcpp.memory cimport shared_ptr

cdef extern from "test.hpp":
    cdef cppclass Test:
        Test() except +
    cdef cppclass Tester(Test):
        Test() except +

cdef extern from "tester.hpp":
    cdef cppclass Gester(Tester):
        Gester() except +

cdef class PyGester(PyTester):
    def __cinit__(self):
        self.test = shared_ptr[Test](new Gester())