#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "QuantCore C++ backtesting engine";

    // Test function
    m.def("hello", []() {
        return "Hello from QuantCore C++!";
    }, "A simple test function");
}