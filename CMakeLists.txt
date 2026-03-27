cmake_minimum_required(VERSION 3.15)
project(QuantCore VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific warning flags
if(MSVC)
    add_compile_options(/W4)
    add_compile_options($<$<CONFIG:Release>:/O2> $<$<CONFIG:Release>:/DNDEBUG>)
    add_compile_options($<$<CONFIG:Debug>:/Od> $<$<CONFIG:Debug>:/Zi>)
else()
    add_compile_options(-Wall -Wextra)
    add_compile_options($<$<CONFIG:Release>:-O3> $<$<CONFIG:Release>:-DNDEBUG>)
    add_compile_options($<$<CONFIG:Debug>:-g> $<$<CONFIG:Debug>:-O0>)
endif()

option(BUILD_TESTS           "Build tests"                  ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings"        OFF)
option(BUILD_EXAMPLES        "Build example main executable" OFF)

include_directories(
        ${PROJECT_SOURCE_DIR}/cpp
        ${PROJECT_SOURCE_DIR}/cpp/orderbook
)

if(BUILD_EXAMPLES)
    add_executable(QuantCore cpp/main.cpp)

    target_include_directories(QuantCore PRIVATE
            ${PROJECT_SOURCE_DIR}/cpp
            ${PROJECT_SOURCE_DIR}/cpp/orderbook
    )

    message(STATUS "Build examples: ON (requires data/ CSVs at runtime)")
endif()

if(BUILD_TESTS)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    file(GLOB_RECURSE TEST_SOURCES ${PROJECT_SOURCE_DIR}/cpp/tests/*.cpp)

    if(TEST_SOURCES)
        add_executable(quantcore_tests ${TEST_SOURCES})

        target_link_libraries(quantcore_tests gtest_main gtest)

        target_include_directories(quantcore_tests PRIVATE
                ${PROJECT_SOURCE_DIR}/cpp
                ${PROJECT_SOURCE_DIR}/cpp/orderbook
        )

        if(NOT MSVC)
            target_compile_options(quantcore_tests PRIVATE --coverage)
            target_link_options(quantcore_tests PRIVATE --coverage)
        endif()

        include(GoogleTest)
        gtest_discover_tests(quantcore_tests)
    endif()
endif()

option(BUILD_BENCHMARKS "Build performance benchmarks" ON)

if(BUILD_BENCHMARKS)
    add_executable(bench_backtest_engine benchmarks/bench_backtest_engine.cpp)

    target_include_directories(bench_backtest_engine PRIVATE
            ${PROJECT_SOURCE_DIR}/cpp
            ${PROJECT_SOURCE_DIR}/cpp/orderbook
    )

    target_compile_options(bench_backtest_engine PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/O2>
            $<$<CXX_COMPILER_ID:MSVC>:/DNDEBUG>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-DNDEBUG>
    )

    add_executable(bench_tick_data benchmarks/bench_tick_data.cpp)

    target_include_directories(bench_tick_data PRIVATE
            ${PROJECT_SOURCE_DIR}/cpp
            ${PROJECT_SOURCE_DIR}/cpp/orderbook
    )

    target_compile_options(bench_tick_data PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/O2>
            $<$<CXX_COMPILER_ID:MSVC>:/DNDEBUG>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-DNDEBUG>
    )

    message(STATUS "Build benchmarks: ON")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "Build examples: ${BUILD_EXAMPLES}")