import pyximport; pyximport.install()
import test1 as t1
import test2 as t2

p1 = t1.PyTest()
q1 = t1.PyTester()
p2 = t2.PyTest2()
q2 = t2.PyTester2()

print('t2.py_can_cast(p1)', t2.py_can_cast(p1))
print('t2.py_can_cast(p2)', t2.py_can_cast(p2))
print('t2.py_can_cast(q1)', t2.py_can_cast(q1))
print('t2.py_can_cast(q2)', t2.py_can_cast(q2))
print('t1.py_can_cast(p1)', t1.py_can_cast(p1))
print('t1.py_can_cast(p2)', t1.py_can_cast(p2))
print('t1.py_can_cast(q1)', t1.py_can_cast(q1))
print('t1.py_can_cast(q2)', t1.py_can_cast(q2))

print()

print('t2.py_type_code(p1)', t2.py_type_code(p1))
print('t2.py_type_code(p2)', t2.py_type_code(p2))
print('t2.py_type_code(q1)', t2.py_type_code(q1))
print('t2.py_type_code(q2)', t2.py_type_code(q2))
print('t1.py_type_code(p1)', t1.py_type_code(p1))
print('t1.py_type_code(p2)', t1.py_type_code(p2))
print('t1.py_type_code(q1)', t1.py_type_code(q1))
print('t1.py_type_code(q2)', t1.py_type_code(q2))
