#ifndef TEST_HPP
#define TEST_HPP

#include <typeinfo>

class Test {
public:
    Test();
    virtual ~Test() {}
};

class Tester : public Test {
public:
    Tester();
};

std::size_t type_code(Test* test);

bool can_cast(Test* test);
#endif