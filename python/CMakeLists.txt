cmake_minimum_required(VERSION 3.15)
project(quantcore_python CXX)

set(CMAKE_CXX_STANDARD 20)

# Force 64-bit build on Windows
if(WIN32)
    set(CMAKE_GENERATOR_PLATFORM x64)
endif()

find_package(Python3 COMPONENTS Interpreter Development.Module)

if(NOT Python3_FOUND)
    # Fallback for conda environments
    message(STATUS "Standard FindPython3 failed, trying fallback method...")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import sys; print(sys.prefix)"
        OUTPUT_VARIABLE PYTHON_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('include'))"
        OUTPUT_VARIABLE Python3_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    message(STATUS "Python prefix: ${PYTHON_PREFIX}")
    message(STATUS "Python include: ${Python3_INCLUDE_DIRS}")

    set(Python3_FOUND TRUE)
endif()

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