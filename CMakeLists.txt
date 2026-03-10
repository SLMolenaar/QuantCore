cmake_minimum_required(VERSION 3.15)
project(QuantCore VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

# Include directories
include_directories(
        ${PROJECT_SOURCE_DIR}/cpp
        ${PROJECT_SOURCE_DIR}/cpp/orderbook
)

# Collect source files from cpp/ (excluding subdirectories for now)
file(GLOB CPP_SOURCES
        ${PROJECT_SOURCE_DIR}/cpp/*.cpp
)

# Create main executable
add_executable(QuantCore ${CPP_SOURCES}
        cpp/backtesting/event.h
        cpp/backtesting/event_queue.h
        cpp/backtesting/market_data_event.h
        cpp/backtesting/signal_event.h
        cpp/backtesting/order_event.h
        cpp/backtesting/fill_event.h
        cpp/backtesting/bar_data.h
        cpp/backtesting/data_loader.h
        cpp/backtesting/strategy.h
        cpp/backtesting/backtest_engine.h
        cpp/strategies/buy_and_hold.h
        cpp/strategies/sma_crossover.h
        cpp/strategies/mean_reversion.h
        cpp/backtesting/market_maker.h
        cpp/backtesting/position_sizer.h
)

# Link libraries if needed
# target_link_libraries(QuantCore pthread)

# Testing
if(BUILD_TESTS)
    enable_testing()

    # Fetch GoogleTest
    include(FetchContent)
    FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # Test executable
    file(GLOB_RECURSE TEST_SOURCES ${PROJECT_SOURCE_DIR}/cpp/tests/*.cpp)

    if(TEST_SOURCES)
        add_executable(quantcore_tests ${TEST_SOURCES})

        target_link_libraries(quantcore_tests
                gtest_main
                gtest
        )

        target_include_directories(quantcore_tests PRIVATE
                ${PROJECT_SOURCE_DIR}/cpp
                ${PROJECT_SOURCE_DIR}/cpp/orderbook
        )

        target_compile_options(quantcore_tests PRIVATE --coverage)
        target_link_options(quantcore_tests PRIVATE --coverage)

        include(GoogleTest)
        gtest_discover_tests(quantcore_tests)
    endif()
endif()


# Benchmarks
option(BUILD_BENCHMARKS "Build performance benchmarks" ON)

if(BUILD_BENCHMARKS)
    add_executable(bench_backtest_engine
        benchmarks/bench_backtest_engine.cpp
    )

    target_include_directories(bench_backtest_engine PRIVATE
        ${PROJECT_SOURCE_DIR}/cpp
        ${PROJECT_SOURCE_DIR}/cpp/orderbook
    )

    # benchmarks must be built in Release for meaningful numbers
    target_compile_options(bench_backtest_engine PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/O2 /DNDEBUG>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -DNDEBUG>
    )

    message(STATUS "Build benchmarks: ON")
endif()


# Print configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build tests: ${BUILD_TESTS}")