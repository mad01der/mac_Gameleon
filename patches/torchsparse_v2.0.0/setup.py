import glob
import os
import platform
import subprocess

import torch
import torch.cuda
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (CUDA_HOME, BuildExtension, CppExtension,
                                       CUDAExtension)

from torchsparse import __version__

if ((torch.cuda.is_available() and CUDA_HOME is not None)
        or (os.getenv('FORCE_CUDA', '0') == '1')):
    device = 'cuda'
else:
    device = 'cpu'

sources = [os.path.join('torchsparse', 'backend', f'pybind_{device}.cpp')]
for fpath in glob.glob(os.path.join('torchsparse', 'backend', '**', '*')):
    if ((fpath.endswith('_cpu.cpp') and device in ['cpu', 'cuda'])
            or (fpath.endswith('_cuda.cu') and device == 'cuda')):
        sources.append(fpath)

extension_type = CUDAExtension if device == 'cuda' else CppExtension


def _brew_prefix(formula: str) -> str:
    try:
        return subprocess.check_output(["brew", "--prefix", formula], text=True).strip()
    except Exception:
        return ""


extra_link_args = []
if platform.system() == "Darwin":
    libomp = _brew_prefix("libomp")
    if not libomp:
        raise RuntimeError("libomp not found. Install it with: brew install libomp")
    extra_compile_args = {
        "cxx": [
            "-g",
            "-O3",
            "-Xpreprocessor",
            "-fopenmp",
            f"-I{libomp}/include",
        ],
        "nvcc": ["-O3"],
    }
    extra_link_args = [f"-L{libomp}/lib", "-lomp"]
else:
    extra_compile_args = {
        "cxx": ["-g", "-O3", "-fopenmp", "-lgomp"],
        "nvcc": ["-O3"],
    }

setup(
    name='torchsparse',
    version=__version__,
    packages=find_packages(),
    ext_modules=[
        extension_type(
            'torchsparse.backend',
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
