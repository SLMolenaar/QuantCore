"""
Build script for QuantCore Python bindings

This script handles:
- Finding CMake
- Configuring and building the C++ extension
- Copying the built module to the correct location
- Testing the build

Usage:
    python build_module.py [--clean] [--test]

Options:
    --clean     Clean build directory before building
    --test      Run tests after building
"""

import subprocess
import shutil
import os
import sys
import argparse
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output"""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(msg):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{msg:^60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}\n")


def print_success(msg):
    """Print success message"""
    print(f"{Colors.GREEN}✓{Colors.RESET} {msg}")


def print_error(msg):
    """Print error message"""
    print(f"{Colors.RED}✗{Colors.RESET} {msg}")


def print_warning(msg):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {msg}")


def print_info(msg):
    """Print info message"""
    print(f"  {msg}")


def find_cmake():
    """
    Find cmake.exe in common locations

    Returns:
        Path to cmake executable or None if not found
    """
    # Try common paths first
    possible_paths = [
        r"C:\Program Files\CMake\bin\cmake.exe",
        r"C:\Program Files (x86)\CMake\bin\cmake.exe",
        shutil.which("cmake"),
    ]

    for path in possible_paths:
        if path and Path(path).exists():
            return path

    # Try JetBrains CLion bundled CMake
    try:
        jetbrains_base = Path(r"C:\Program Files\JetBrains")
        if jetbrains_base.exists():
            clion_paths = list(jetbrains_base.glob("*/bin/cmake/win/*/bin/cmake.exe"))
            if clion_paths:
                return str(clion_paths[0])
    except:
        pass

    # Try Visual Studio bundled CMake
    try:
        vs_paths = [
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
            r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
            r"C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        ]
        for path in vs_paths:
            if Path(path).exists():
                return path
    except:
        pass

    # Try Linux/Mac paths
    linux_paths = [
        "/usr/bin/cmake",
        "/usr/local/bin/cmake",
        "/opt/homebrew/bin/cmake",
    ]

    for path in linux_paths:
        if Path(path).exists():
            return path

    return None


def check_dependencies():
    """
    Check that all dependencies are available

    Returns:
        tuple: (success, messages)
    """
    messages = []
    success = True

    # Check Python
    python_version = sys.version_info
    if python_version < (3, 8):
        messages.append(f"Python 3.8+ required (found {python_version.major}.{python_version.minor})")
        success = False
    else:
        messages.append(f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")

    # Check pybind11
    try:
        import pybind11
        messages.append(f"pybind11 {pybind11.__version__}")
    except ImportError:
        messages.append("pybind11 not found (pip install pybind11)")
        success = False

    # Check CMake
    cmake_path = find_cmake()
    if cmake_path:
        try:
            result = subprocess.run(
                [cmake_path, "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            version_line = result.stdout.split('\n')[0]
            messages.append(f"CMake: {version_line}")
        except:
            messages.append(f"CMake found but can't get version")
    else:
        messages.append("CMake not found")
        success = False

    return success, messages


def clean_build_dir(build_dir):
    """Remove build directory if it exists"""
    if build_dir.exists():
        print_info(f"Removing {build_dir}")
        shutil.rmtree(build_dir)
        print_success("Build directory cleaned")


def configure_cmake(python_dir, build_dir, cmake_path, python_exe):
    """
    Configure CMake

    Returns:
        bool: Success
    """
    print_info("Configuring CMake...")

    # Detect platform
    import platform
    is_windows = platform.system() == "Windows"

    cmake_args = [
        cmake_path,
        "-S", ".",
        "-B", "build",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DPython3_EXECUTABLE={python_exe}"
    ]

    # On Windows with Visual Studio, force 64-bit architecture
    if is_windows:
        cmake_args.extend(["-A", "x64"])

    try:
        subprocess.run(cmake_args, cwd=python_dir, check=True, capture_output=True, text=True)

        print_success("CMake configured successfully")
        return True

    except subprocess.CalledProcessError as e:
        print_error("CMake configuration failed")
        if e.stdout:
            print_info("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print_info("STDERR:")
            print(e.stderr)
        return False


def build_cmake(python_dir, cmake_path):
    """
    Build with CMake

    Returns:
        bool: Success
    """
    print_info("Building with CMake...")

    try:
        subprocess.run([
            cmake_path,
            "--build", "build",
            "--config", "Release"
        ], cwd=python_dir, check=True, capture_output=True, text=True)

        print_success("Build completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print_error("Build failed")
        if e.stdout:
            print_info("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print_info("STDERR:")
            print(e.stderr)
        return False


def find_built_module(build_dir):
    """
    Find the built module file

    Returns:
        Path to built module or None
    """
    # Possible extensions
    extensions = [".pyd", ".so", ".dll", ".dylib"]

    # Search paths
    search_paths = [
        build_dir / "Release",
        build_dir / "Debug",
        build_dir,
    ]

    for search_path in search_paths:
        if not search_path.exists():
            continue

        for ext in extensions:
            candidates = list(search_path.glob(f"*_core*{ext}"))
            if candidates:
                return candidates[0]

    return None


def copy_module(built_file, python_dir):
    """
    Copy built module to quantcore package

    Returns:
        Path to destination or None on failure
    """
    dest = python_dir / "quantcore" / built_file.name
    dest.parent.mkdir(exist_ok=True)

    try:
        shutil.copy(built_file, dest)
        print_success(f"Module copied to: {dest}")
        return dest
    except Exception as e:
        print_error(f"Failed to copy module: {e}")
        return None


def test_import(python_dir):
    """
    Test importing the module

    Returns:
        bool: Success
    """
    print_info("Testing module import...")

    try:
        # Add quantcore to path
        sys.path.insert(0, str(python_dir))

        import quantcore

        # Test basic functionality
        version = quantcore.version()
        hello = quantcore.hello()

        print_success(f"Import successful: {hello}")
        print_info(f"Version: {version}")
        return True

    except Exception as e:
        print_error(f"Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_tests(python_dir):
    """
    Run pytest if available

    Returns:
        bool: Success
    """
    print_info("Running tests...")

    try:
        import pytest
    except ImportError:
        print_warning("pytest not installed, skipping tests")
        return True

    test_file = python_dir / "tests" / "test_backtest.py"
    if not test_file.exists():
        print_warning(f"Test file not found: {test_file}")
        return True

    try:
        result = pytest.main([str(test_file), "-v"])
        if result == 0:
            print_success("All tests passed")
            return True
        else:
            print_warning("Some tests failed")
            return False
    except Exception as e:
        print_error(f"Error running tests: {e}")
        return False


def build(clean=False, run_test=False):
    """
    Main build function

    Args:
        clean: Whether to clean build directory first
        run_test: Whether to run tests after building

    Returns:
        bool: Success
    """
    print_header("QuantCore Python Bindings - Build Script")

    python_dir = Path(__file__).parent
    build_dir = python_dir / "build"

    # Check dependencies
    print_info("Checking dependencies...")
    deps_ok, messages = check_dependencies()
    for msg in messages:
        if "not found" in msg.lower() or "required" in msg.lower():
            print_error(msg)
        else:
            print_success(msg)

    if not deps_ok:
        print_error("\nMissing dependencies. Please install them first.")
        return False

    print()

    # Get paths
    cmake_path = find_cmake()
    python_exe = sys.executable

    print_info(f"CMake: {cmake_path}")
    print_info(f"Python: {python_exe}")
    print_info(f"Build dir: {build_dir}")
    print()

    # Clean if requested
    if clean and build_dir.exists():
        clean_build_dir(build_dir)
        print()

    # Configure
    if not configure_cmake(python_dir, build_dir, cmake_path, python_exe):
        return False

    # Build
    if not build_cmake(python_dir, cmake_path):
        return False

    # Find built module
    print_info("Locating built module...")
    built_file = find_built_module(build_dir)

    if not built_file:
        print_error("Could not find built module!")
        print_info("Build directory contents:")
        for item in build_dir.rglob("*"):
            if item.is_file():
                print_info(f"  {item}")
        return False

    print_success(f"Found: {built_file}")

    # Copy module
    dest = copy_module(built_file, python_dir)
    if not dest:
        return False

    print()

    # Test import
    if not test_import(python_dir):
        return False

    print()

    # Run tests if requested
    if run_test:
        if not run_tests(python_dir):
            print_warning("Tests failed but build succeeded")
        print()

    # Success!
    print_header("Build Successful!")
    print_success("Module ready to use")
    print()
    print_info("Try it out:")
    print_info('  python -c "import quantcore; print(quantcore.hello())"')
    print_info("Or run the examples:")
    print_info("  python python/examples/example_usage.py")
    print()

    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Build QuantCore Python bindings"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean build directory before building"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run tests after building"
    )

    args = parser.parse_args()

    try:
        success = build(clean=args.clean, run_test=args.test)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_error("\n\nBuild cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()