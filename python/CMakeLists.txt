cmake_minimum_required(VERSION 3.15)
project(quantcore_python)

set(CMAKE_CXX_STANDARD 20)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find pybind11
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 REQUIRED HINTS ${pybind11_DIR})

# Create the Python module
pybind11_add_module(_core bindings.cpp)

# Include directories for C++ headers
target_include_directories(_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/orderbook
)

# Set output name
set_target_properties(_core PROPERTIES
    OUTPUT_NAME "_core"
    PREFIX ""
)