from setuptools import setup
from torch_musa.utils.musa_extension import (BuildExtension, MUSAExtension)
import os

# 指定 MUSA 编译器
os.environ['CUDACXX'] = '/usr/local/musa/bin/mcc'
os.environ['CUDA_HOME'] = '/usr/local/musa'

# 因为 .mu 文件不是标准 .cu，我们直接当作 CUDA 文件处理
sources = [
    'lib/dvr/dvr.cpp',
    'lib/dvr/dvr_kernel.mu',  # 可以保留 .mu
]

# extra flags
extra_compile_args = {
    'cxx': ['-std=c++17', '-fPIC'],
    'nvcc': ['-allow-unsupported-compiler', '-DMUSA_ENABLED', '-I/usr/local/musa/include']
}

extra_link_args = ['-L/usr/local/musa/lib', '-lmusart', '-lmusa']

# 使用 CUDAExtension 即使是 MUSA
setup(
    name='dvr',
    ext_modules=[
        MUSAExtension(
            name='dvr',
            sources=sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

