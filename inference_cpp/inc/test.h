#include <iostream>

class test {
public:
    void print();

    template<class T>
    T add(T a, T b)
    {
        return a + b;
    }

};