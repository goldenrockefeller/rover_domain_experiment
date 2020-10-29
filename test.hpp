#ifndef TEST_HPP
#define TEST_HPP

#include <typeinfo>

class Test {
public:
    virtual ~Test() {}
};

class Tester : public Test {
};

std::size_t type_code(Test* test) {
    return typeid(*test).hash_code();
}

bool can_cast(Test* test) {
    return dynamic_cast<Tester*>(test);
}
#endif