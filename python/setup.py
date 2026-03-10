"""
Setup script for QuantCore Python bindings

This allows installing QuantCore as a package:
    pip install .
    pip install -e .  # editable mode
"""

import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """Extension that uses CMake to build"""

    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = Path(sourcedir).absolute()


class CMakeBuild(build_ext):
    """Build extension using CMake"""

    def build_extension(self, ext):
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        extdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()

        # CMake configuration arguments
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPython3_EXECUTABLE={sys.executable}',
            '-DCMAKE_BUILD_TYPE=Release',
        ]

        # Build arguments
        build_args = [
            '--config', 'Release',
        ]

        # Create build directory
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        # Run CMake configuration
        subprocess.check_call(
            ['cmake', str(ext.sourcedir)] + cmake_args,
            cwd=build_temp
        )

        # Run CMake build
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args,
            cwd=build_temp
        )


# Read README
readme_file = Path(__file__).parent / 'PYTHON_README.md'
long_description = ''
if readme_file.exists():
    long_description = readme_file.read_text(encoding='utf-8')

# Read version
version = '0.1.0'

setup(
    name='quantcore',
    version=version,
    author='Stefaan Molenaar',
    author_email='StefaanLMolenaar@gmail.com',
    description='High-performance backtesting engine for trading strategies',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/SLMolenaar/quantcore',
    packages=find_packages(where='python'),
    package_dir={'': 'python'},
    ext_modules=[CMakeExtension('quantcore._core', sourcedir='python')],
    cmdclass={
        'build_ext': CMakeBuild,
    },
    install_requires=[
        'pybind11>=2.11.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
        ],
        'viz': [
            'matplotlib>=3.7.0',
            'seaborn>=0.12.0',
        ],
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Developers',
        'Topic :: Office/Business :: Financial :: Investment',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: C++',
    ],
    keywords='trading backtesting algorithmic-trading quantitative-finance',
    project_urls={
        'Bug Reports': 'https://github.com/SLMolenaar/quantcore/issues',
        'Source': 'https://github.com/SLMolenaar/quantcore',
    },
)