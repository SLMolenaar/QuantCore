cmake_minimum_required(VERSION 3.15...3.27)

project(
        ${SKBUILD_PROJECT_NAME}
        VERSION ${SKBUILD_PROJECT_VERSION}
        LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core MODULE ../bindings.cpp)

target_include_directories(_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/orderbook
)

target_compile_definitions(_core PRIVATE
        VERSION_INFO=${SKBUILD_PROJECT_VERSION}
)

install(TARGETS _core DESTINATION quantcore)