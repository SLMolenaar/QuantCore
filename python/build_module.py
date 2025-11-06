import subprocess
import shutil
import os
import sys
from pathlib import Path


def find_cmake():
    """Find cmake.exe in common locations"""
    possible_paths = [
        r"C:\Program Files\CMake\bin\cmake.exe",
        r"C:\Program Files (x86)\CMake\bin\cmake.exe",
        shutil.which("cmake"),
    ]

    for path in possible_paths:
        if path and Path(path).exists():
            return path

    clion_paths = list(Path(r"C:\Program Files\JetBrains").glob("*/bin/cmake/win/x64/bin/cmake.exe"))
    if clion_paths:
        return str(clion_paths[0])

    raise FileNotFoundError("CMake not found!")


def build():
    python_dir = Path(__file__).parent
    build_dir = python_dir / "build"

    cmake = find_cmake()
    python_exe = sys.executable

    print(f"Using CMake: {cmake}")
    print(f"Using Python: {python_exe}")
    print("Building with CMake...\n")

    subprocess.run([
        cmake,
        "-S", ".",
        "-B", "build",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DPython3_EXECUTABLE={python_exe}"
    ], cwd=python_dir, check=True)

    subprocess.run([
        cmake,
        "--build", "build",
        "--config", "Release"
    ], cwd=python_dir, check=True)

    built_file = None
    for ext in [".pyd", ".so", ".dll"]:
        search_paths = [
            build_dir / "Release",
            build_dir,
        ]

        for search_path in search_paths:
            if not search_path.exists():
                continue
            candidates = list(search_path.glob(f"*_core*{ext}"))
            if candidates:
                built_file = candidates[0]
                break
        if built_file:
            break

    if built_file:
        dest = python_dir / "quantcore" / built_file.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy(built_file, dest)
        print(f"\n✓ Built module successfully!")
        print(f"  Location: {dest}")
        print(f"\nTest with:")
        print(f'  python -c "import quantcore._core; print(\'Success!\')"')
    else:
        print("\n✗ Could not find built module!")
        print(f"Build directory contents:")
        for item in build_dir.rglob("*"):
            if item.is_file():
                print(f"  {item}")
        return False

    return True


if __name__ == "__main__":
    try:
        build()
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Build failed!")
        exit(1)