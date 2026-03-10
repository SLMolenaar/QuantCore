# Contributing

## Getting Started

Clone the repo and build the project:
```bash
git clone https://github.com/SLMolenaar/quantcore.git
cd quantcore
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

## Running Tests
```bash
cmake --build build --target quantcore_tests
./build/quantcore_tests
```

## Code Style

- follow the style of the surrounding code

## Submitting Changes

1. Fork the repo
2. Create a branch from `master`
3. Make your changes with tests covering new behavior
4. Open a pull request with a clear description of what changed and why

## Reporting Issues

Open a GitHub issue with a minimal reproduction case.