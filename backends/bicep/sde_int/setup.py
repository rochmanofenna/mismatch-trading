# backends/bicep/sde_int/setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")

setup(
    name="sde_int",
    ext_modules=[
        CUDAExtension(
            name="sde_int",
            sources=["curand_kernel.cu", "binding.cpp"],
            include_dirs=[os.path.join(cuda_home, "include")],
            libraries=["curand"],
            extra_compile_args={"nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
