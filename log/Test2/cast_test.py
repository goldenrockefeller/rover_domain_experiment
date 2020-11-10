import pyximport; pyximport.install()
import test1 as t1
import test2 as t2

p1 = t1.PyTest()
q1 = t1.PyTester()
r2 = t2.PyGester()


print('t1.py_can_cast(r2)', t1.py_can_cast(r2))


print()

print('t1.py_type_code(r2)', t1.py_type_code(r2))

