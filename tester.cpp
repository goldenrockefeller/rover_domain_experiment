#include "test.hpp"

Test::Test() {}

Tester::Tester() {}

std::size_t type_code(Test* test) {
    return typeid(*test).hash_code();
}

bool can_cast(Test* test) {
    return dynamic_cast<Tester*>(test);
}