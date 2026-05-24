r"""
Mac-patched MinkowskiEngine v0.5.4 setup.py (CPU-only on Apple Silicon).

Changes vs upstream:
- libomp / -Xpreprocessor -fopenmp on Darwin (Apple clang)
- extra_link_args passed to the extension (fixes ___kmpc_for_static_fini)
- OpenBLAS include/lib dirs from Homebrew when --blas=openblas
- Do not force a missing /usr/local/opt/llvm clang as CC
"""
import sys

if sys.version_info < (3, 6):
    sys.stdout.write(
        "Minkowski Engine requires Python 3.6 or higher.\n"
    )
    sys.exit(1)

try:
    import torch
except ImportError:
    raise ImportError("Pytorch not found. Please install pytorch first.")

import warnings
import codecs
import os
import re
import subprocess
from sys import argv, platform
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
from pathlib import Path

if platform == "win32":
    raise ImportError("Windows is currently not supported.")

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def run_command(*args):
    subprocess.check_call(args)


def _brew_prefix(formula: str) -> str:
    try:
        return subprocess.check_output(["brew", "--prefix", formula], text=True).strip()
    except Exception:
        return ""


def _argparse(pattern, argv, is_flag=True, is_list=False):
    if is_flag:
        found = pattern in argv
        if found:
            argv.remove(pattern)
        return found, argv
    else:
        arr = [arg for arg in argv if pattern == arg.split("=")[0]]
        if is_list:
            if len(arr) == 0:
                return False, argv
            assert "=" in arr[0], f"{arr[0]} requires a value."
            argv.remove(arr[0])
            val = arr[0].split("=")[1]
            if "," in val:
                return val.split(","), argv
            return [val], argv
        if len(arr) == 0:
            return False, argv
        assert "=" in arr[0], f"{arr[0]} requires a value."
        argv.remove(arr[0])
        return arr[0].split("=")[1], argv


run_command("rm", "-rf", "build")
run_command("pip", "uninstall", "MinkowskiEngine", "-y")

CPU_ONLY, argv = _argparse("--cpu_only", argv)
FORCE_CUDA, argv = _argparse("--force_cuda", argv)
if not torch.cuda.is_available() and not FORCE_CUDA:
    warnings.warn(
        "torch.cuda.is_available() is False. MinkowskiEngine will compile with CPU_ONLY."
    )

CPU_ONLY = CPU_ONLY or not torch.cuda.is_available()
if FORCE_CUDA:
    CPU_ONLY = False

CUDA_HOME, argv = _argparse("--cuda_home", argv, False)
BLAS, argv = _argparse("--blas", argv, False)
BLAS_INCLUDE_DIRS, argv = _argparse("--blas_include_dirs", argv, False, is_list=True)
BLAS_LIBRARY_DIRS, argv = _argparse("--blas_library_dirs", argv, False, is_list=True)
MAX_COMPILATION_THREADS = 12

Extension = CUDAExtension
extra_link_args = []
include_dirs = []
libraries = []
CC_FLAGS = []
NVCC_FLAGS = []

if CPU_ONLY:
    print("--------------------------------")
    print("| WARNING: CPU_ONLY build set  |")
    print("--------------------------------")
    Extension = CppExtension
else:
    print("--------------------------------")
    print("| CUDA compilation set         |")
    print("--------------------------------")
    libraries.append("cusparse")

if not (CUDA_HOME is False):
    print(f"Using CUDA_HOME={CUDA_HOME}")

if sys.platform == "win32":
    vc_version = os.getenv("VCToolsVersion", "")
    if vc_version.startswith("14.16."):
        CC_FLAGS += ["/sdl"]
    else:
        CC_FLAGS += ["/sdl", "/permissive-"]
elif sys.platform == "darwin":
    libomp = _brew_prefix("libomp")
    if not libomp:
        raise RuntimeError("libomp not found. Install with: brew install libomp")
    CC_FLAGS += [
        "-Xpreprocessor",
        "-fopenmp",
        f"-I{libomp}/include",
        "-stdlib=libc++",
        "-std=c++17",
    ]
    extra_link_args += [f"-L{libomp}/lib", "-lomp"]
else:
    CC_FLAGS += ["-fopenmp"]

NVCC_FLAGS += ["--expt-relaxed-constexpr", "--expt-extended-lambda"]
FAST_MATH, argv = _argparse("--fast_math", argv)
if FAST_MATH:
    NVCC_FLAGS.append("--use_fast_math")

BLAS_LIST = ["openblas", "mkl", "atlas", "blas"]
if not (BLAS is False):
    assert BLAS in BLAS_LIST, f"Blas option {BLAS} not in valid options {BLAS_LIST}"
    if BLAS == "mkl":
        libraries.append("mkl_rt")
    else:
        libraries.append(BLAS)
    if not (BLAS_INCLUDE_DIRS is False):
        include_dirs += BLAS_INCLUDE_DIRS
    if not (BLAS_LIBRARY_DIRS is False):
        for lib_dir in BLAS_LIBRARY_DIRS:
            extra_link_args += [f"-L{lib_dir}", f"-Wl,-rpath,{lib_dir}"]
    if BLAS == "openblas" and sys.platform == "darwin":
        openblas = _brew_prefix("openblas")
        if openblas:
            include_dirs.append(f"{openblas}/include")
            extra_link_args += [f"-L{openblas}/lib", f"-Wl,-rpath,{openblas}/lib"]
else:
    try:
        import numpy.distutils.system_info as sysinfo
    except ImportError:
        raise ImportError(
            "numpy.distutils is unavailable (NumPy 2.x). "
            "Specify BLAS explicitly: python setup.py install --cpu_only --blas=openblas "
            "--blas_include_dirs=$(brew --prefix openblas)/include "
            "--blas_library_dirs=$(brew --prefix openblas)/lib"
        )
    for blas in BLAS_LIST:
        if "libraries" in sysinfo.get_info(blas):
            BLAS = blas
            libraries += sysinfo.get_info(blas)["libraries"]
            break
    else:
        raise ImportError(
            "BLAS not found. Use: python setup.py install --cpu_only --blas=openblas"
        )

print(f"\nUsing BLAS={BLAS}")

SOURCE_SETS = {
    "cpu": [
        CppExtension,
        [
            "math_functions_cpu.cpp",
            "coordinate_map_manager.cpp",
            "convolution_cpu.cpp",
            "convolution_transpose_cpu.cpp",
            "local_pooling_cpu.cpp",
            "local_pooling_transpose_cpu.cpp",
            "global_pooling_cpu.cpp",
            "broadcast_cpu.cpp",
            "pruning_cpu.cpp",
            "interpolation_cpu.cpp",
            "quantization.cpp",
            "direct_max_pool.cpp",
        ],
        ["pybind/minkowski.cpp"],
        ["-DCPU_ONLY"],
    ],
    "gpu": [
        CUDAExtension,
        [
            "math_functions_cpu.cpp",
            "math_functions_gpu.cu",
            "coordinate_map_manager.cu",
            "coordinate_map_gpu.cu",
            "convolution_kernel.cu",
            "convolution_gpu.cu",
            "convolution_transpose_gpu.cu",
            "pooling_avg_kernel.cu",
            "pooling_max_kernel.cu",
            "local_pooling_gpu.cu",
            "local_pooling_transpose_gpu.cu",
            "global_pooling_gpu.cu",
            "broadcast_kernel.cu",
            "broadcast_gpu.cu",
            "pruning_gpu.cu",
            "interpolation_gpu.cu",
            "spmm.cu",
            "gpu.cu",
            "quantization.cpp",
            "direct_max_pool.cpp",
        ],
        ["pybind/minkowski.cu"],
        [],
    ],
}

debug, argv = _argparse("--debug", argv)

HERE = Path(os.path.dirname(__file__)).absolute()
SRC_PATH = HERE / "src"

if "CC" in os.environ or "CXX" in os.environ:
    if "CXX" in os.environ:
        os.environ["CC"] = os.environ["CXX"]
        CC = os.environ["CXX"]
    else:
        CC = os.environ["CC"]
    print(f"Using {CC} for c++ compilation")
    if torch.__version__ < "1.7.0":
        NVCC_FLAGS += [f"-ccbin={CC}"]
else:
    print("Using the default compiler")

if debug:
    CC_FLAGS += ["-g", "-DDEBUG"]
    NVCC_FLAGS += ["-g", "-DDEBUG"]
else:
    CC_FLAGS += ["-O3"]
    NVCC_FLAGS += ["-O3"]

if "MAX_JOBS" not in os.environ and os.cpu_count() > MAX_COMPILATION_THREADS:
    os.environ["MAX_JOBS"] = str(MAX_COMPILATION_THREADS)

target = "cpu" if CPU_ONLY else "gpu"

Extension = SOURCE_SETS[target][0]
SRC_FILES = SOURCE_SETS[target][1]
BIND_FILES = SOURCE_SETS[target][2]
ARGS = SOURCE_SETS[target][3]
CC_FLAGS += ARGS
NVCC_FLAGS += ARGS

ext_modules = [
    Extension(
        name="MinkowskiEngineBackend._C",
        sources=[*[str(SRC_PATH / src_file) for src_file in SRC_FILES], *BIND_FILES],
        extra_compile_args={"cxx": CC_FLAGS, "nvcc": NVCC_FLAGS},
        extra_link_args=extra_link_args,
        libraries=libraries,
    ),
]

setup(
    name="MinkowskiEngine",
    version=find_version("MinkowskiEngine", "__init__.py"),
    install_requires=["torch", "numpy"],
    packages=["MinkowskiEngine", "MinkowskiEngine.utils", "MinkowskiEngine.modules"],
    package_dir={"MinkowskiEngine": "./MinkowskiEngine"},
    ext_modules=ext_modules,
    include_dirs=[str(SRC_PATH), str(SRC_PATH / "3rdparty"), *include_dirs],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    author="Christopher Choy",
    author_email="chrischoy@ai.stanford.edu",
    description="a convolutional neural network library for sparse tensors",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/NVIDIA/MinkowskiEngine",
    keywords=[
        "pytorch",
        "Minkowski Engine",
        "Sparse Tensor",
        "Convolutional Neural Networks",
        "3D Vision",
        "Deep Learning",
    ],
    zip_safe=False,
    classifiers=[
        "Environment :: Console",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
    python_requires=">=3.6",
)
