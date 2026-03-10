"""
Setup script for QuantCore Python bindings.

    pip install .
    pip install -e .  # editable mode
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = Path(sourcedir).absolute()


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        extdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()

        # Add cmake installed via pip to PATH
        cmake_python_bin = Path(sys.executable).parent
        env = os.environ.copy()
        env["PATH"] = str(cmake_python_bin) + os.pathsep + env.get("PATH", "")

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        build_args = ["--config", "Release"]

        if platform.system() == "Windows":
            cmake_args += [
                "-A", "x64",
                # On Windows, CMake appends the config name as a subdirectory
                # unless the config-specific variable is set explicitly.
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={extdir}",
                f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE={extdir}",
            ]
            build_args += ["--", "/m"]
        else:
            cpu_count = os.cpu_count() or 1
            build_args += ["--", f"-j{cpu_count}"]

        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        subprocess.check_call(
            ["cmake", str(ext.sourcedir)] + cmake_args,
            cwd=build_temp,
            env=env,
            )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=build_temp,
            env=env,
            )


here = Path(__file__).parent

readme = here / "README.md"
long_description = readme.read_text(encoding="utf-8") if readme.exists() else ""

setup(
    name="quantcore",
    version="0.1.3",
    author="Stefaan Molenaar",
    author_email="StefaanLMolenaar@gmail.com",
    description="High-performance C++20 backtesting engine with Python interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SLMolenaar/quantcore",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[CMakeExtension("quantcore._core", sourcedir=str(here / "python"))],
    cmdclass={"build_ext": CMakeBuild},
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
    },
    python_requires=">=3.8",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
    ],
    keywords="trading backtesting algorithmic-trading quantitative-finance",
    project_urls={
        "Bug Reports": "https://github.com/SLMolenaar/quantcore/issues",
        "Source": "https://github.com/SLMolenaar/quantcore",
    },
)