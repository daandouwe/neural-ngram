from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

ext_modules = [
    Extension(
        "_sample",
        sources=["_sample.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-ffast-math"]
    )
]

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
)
