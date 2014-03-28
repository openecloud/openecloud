from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import sys

include_dirs = [numpy.get_include()]
library_dirs = []
include_dirs.append("/usr/local/include/")
library_dirs.append("/usr/local/lib/")
extra_compile_args = ["-O3"]                               # Use high degree of compiler optimization.
ecaGSL = ["-O3", "-lgsl", "-lcblas"]                       # Some modules use GSL.
elaGSL = ["-lgsl", "-lcblas"]

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
                     Extension("poissonSolver", ["poissonSolver.pyx"],
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
                               include_dirs=include_dirs),
                     Extension("util", ["util.pyx"],
                               include_dirs=include_dirs,
                               extra_compile_args=extra_compile_args)
			        ]
)



logoArt = """
                         ______ _____ _      ____  _    _ _____  
                        |  ____/ ____| |    / __ \| |  | |  __ \ 
   ___  _ __   ___ _ __ | |__ | |    | |   | |  | | |  | | |  | |
  / _ \| '_ \ / _ \ '_ \|  __|| |    | |   | |  | | |  | | |  | |
 | (_) | |_) |  __/ | | | |___| |____| |___| |__| | |__| | |__| |
  \___/| .__/ \___|_| |_|______\_____|______\____/ \____/|_____/ 
       | |                                                       
       |_|                                                       
                                                                 
"""
print logoArt
