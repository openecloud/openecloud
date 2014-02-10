from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import sys

include_dirs = [numpy.get_include()]
library_dirs = []
include_dirs.append("/usr/local/include/")
library_dirs.append("/usr/local/lib/")
extra_compile_args = ["-O3"]                            # Use high degree of compiler optimization.
ecaGSL = ["-O3", "-lgsl", "-lgslcblas"]                 # Some modules use GSL.
elaGSL = ["-lgsl", "-lgslcblas"]

setup(
      cmdclass = {'build_ext': build_ext},
	  ext_modules = [	
	                 Extension("magneticField", ["magneticField.pyx"],
	                           include_dirs=include_dirs,
                               extra_compile_args=extra_compile_args),
	                 Extension("particleEmitter", ["particleEmitter.pyx"],
	                           include_dirs=include_dirs,
                               extra_compile_args=extra_compile_args),
      			     Extension("particles", ["particles.pyx"],
      				           include_dirs=include_dirs,
                               extra_compile_args=extra_compile_args),
      			     Extension("particleBoundary", ["particleBoundary.pyx"],
      				           include_dirs=include_dirs,
                               extra_compile_args=extra_compile_args),
      				 Extension("kdTree", ["kdTree.pyx"],
                               include_dirs=include_dirs,
                               extra_compile_args=extra_compile_args),
                     Extension("particleManagement", ["particleManagement.pyx"],
                               include_dirs=include_dirs,
                               extra_compile_args=extra_compile_args),
                     Extension("grid", ["grid.pyx"],
                               include_dirs=include_dirs,
                               extra_compile_args=extra_compile_args),
                     Extension("specFun", ["specFun.pyx"],
                               include_dirs=include_dirs,
                               library_dirs=library_dirs,
                               extra_link_args=elaGSL,
                               extra_compile_args=ecaGSL),
                     Extension("randomGen", ["randomGen.pyx"],
                               include_dirs=include_dirs,
                               library_dirs=library_dirs,
                               extra_link_args=elaGSL,
                               extra_compile_args=ecaGSL),
                     Extension("constants", ["constants.pyx"],
                               include_dirs=include_dirs)
			        ]
)
