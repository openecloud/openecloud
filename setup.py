from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import sys

include_dirs = [numpy.get_include()]
library_dirs = []
include_dirs.append("/usr/local/include/")
library_dirs.append("/usr/local/lib/")




setup(
      cmdclass = {'build_ext': build_ext},
	  ext_modules = [	
	                 Extension("particleEmitter", ["particleEmitter.pyx"]),
      			     Extension("particles", ["particles.pyx"],
      				           include_dirs=include_dirs),
      			     Extension("particleBoundary", ["particleBoundary.pyx"],
      				           include_dirs=include_dirs),
                     Extension("particleManagement", ["particleManagement.pyx"],
                               include_dirs=include_dirs),
                     Extension("grid", ["grid.pyx"],
                               include_dirs=include_dirs),
                     Extension("gslWrap", ["gslWrap.pyx"],
                               include_dirs=include_dirs,
                               library_dirs=library_dirs,
                               extra_link_args=['-lgsl', '-lgslcblas'],
                               extra_compile_args=['-lgsl', '-lgslcblas']),
                     Extension("constants", ["constants.pyx"],
                               include_dirs=include_dirs)
			        ]
)
