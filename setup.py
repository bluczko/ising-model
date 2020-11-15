from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy

# define an extension that will be cythonized and compiled
ext = Extension(name="ising.core", sources=["ising/core.pyx"])
c_ext = cythonize(ext, language_level=3)

setup(ext_modules=c_ext, include_dirs=[numpy.get_include()])
