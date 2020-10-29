cdef class PyTest:
    def __cinit__(self):
        self.test = new Test()

cdef class PyTester(PyTest):
    def __cinit__(self):
        self.test = new Tester()

cpdef py_type_code(PyTest py_test):
    return type_code(py_test.test)

cpdef py_can_cast(PyTest py_test):
    return can_cast(py_test.test)