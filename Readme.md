openecloud
==========

Particle-in-cell code for electron-cloud studies. The code mainly focuses on the buildup of the electron cloud, but is intended to provide tracking studies together with [PyORBIT](https://code.google.com/p/py-orbit/). The code is not feature complete yet and in general is intended as an open environment and community project.

Although the code already can be used for buidup simulation it will be extended and improved in future updates. We are very much interested in feedback and contributions of the community. The documentation will be extended in the future as well, but is intended to only contain a general description. For further details one can look at the examples or the comments in the source code.

Quick Start
---------

1. Download the code from the [repository](https://github.com/openecloud/openecloud).
2. Make sure you have installed the dependencies (see Sec. Dependencies).
3. Run the setup.py (under Linux in the command line: python setup.py build_ext --inplace)
4. Run example01.py (under Linux in the command line: python example01.py).

Module Descriptions
---------

Every module usually consists of a main class which is used to store relevant data for the specific purpose in an object. The object is then used to manipulate the conatining data with provided function. For example the module *particles* conatins the class *Particles* which stores the particle coordinates and a push of these particle coordinates can be done with one of the *borisPush* functions in the same module. Modules start with lowercase and classes with capital letters. Modules usually consist of the main class (with the same name, only capital initial letter) and some helper functions.

### grid

A basic module to handle the properties of the computational grid, like grid lengths and number of cells in each direction. Supports cut-cell boundaries. Only supports equidistant (in each direction) and cartesian grids for now, but this is mainly due to the particles module and not due to the *grid* and *poissonSolver* module. A *Grid* object is handed to every function which needs data of or on the grid. Note that the grid boundary for field computations is implemented here, but particle boundaries are implemented in the *ParticleBoundary* class.

### poissonSolver

Implements a Poisson solver based on the Finite Integration Technique (which is very similar to Finite Differences). The major trick here usually overlooked in other codes is that it is much faster to do a LU-decomposition once of the system matrix at the beginning of the simulation and then only doing the backwards substitution at each time step than using an iterative solver. The performance with the Super-LU library is comparable to FFT-Poisson solvers but much more flexible with cut-cell boundaries.

### particles

Module to store and modifiy particle data, e.g. doing a Boris push.


### particleBoundary

This is more or less a helper module to implement different particle boundaries. This boundary is not automatically the same physical boundary as the grid boundary in the *grid* module.

### particleEmitter

This module implements different particle emitters which for example take the particles absorbed by the *particleBoundary* class to calculate secondary electrons emitted from the boundary. Contains loaders for inital particle distributions as well.

### particleManagement

Provides methods to manage the particle number in the simulations. This is necessary due to the exponential growth of the electrons in buildup simulations. Currently only very simple algorithms are used.

### beam

Some simple methods to provide a rigid beam charge distribution which can be imprinted on the grid. The user can add his beam profile here.

### specFun

Wraps several special functions of the GSL mainly needed by the *particleEmitter* module.

### randomGen

Wraps some random number generation of the GSL.

### constants

Module to store and provide global constants.

### plot

Some plot routines.




Dependencies
---------

The code was mainly used and tested on Ubuntu 12.04 64-bit systems with:
- Cython 0.19
- Numpy 1.7.1
- Scipy 0.12.0
- GLS 1.15
- Matplotlib 1.1.1rc
- GCC 4.6.3

Only the latter is the standard repository version of Ubuntu. Otherwise older versions (e.g. standard Ubuntu) sometime exhibit faulty behavior. Short reasoning for the packages:

- Numpy (with C-bindings for Cython) and its ndarrays are used as data storage and interface between classes. This is intended for easy data handling from/with Python and to use the convenient garbage collection for the arrays. Within the classes/functions typed memoryviews are used for performance.
- Scipy is used for some exotic functions, but is increasingly replaced by Cython/C-functions or libraries for better performance. It is desirable to replace most or all Scipy dependencies in the future.
- For random number generation the Mersenne-Twister of the GSL is used and some special functions. 
- Matplotlib is mainly used in the examples for graphical output. It is highly recommended but isn't stricly necessary for running the code.
- To actually compile the Cython files a C compiler is obviously needed. The standards of Ubuntu 12.04 worked flawlessly for us. Install with sudo apt-get install build-essential, but this most likely is already on your system.

Furthermore the SuperLU library is used to solve the Poisson problem (currently through the Scipy implementation). UMFPACK provides similar sophisticated LU-decomposition and might be a better choice, or a slim/tailered Cython implementation of a suitable LU-decomposition.

Tips and Tricks
--------
- It may be necessary to replace 
  - include_dirs.append("/usr/local/include/")
  - library_dirs.append("/usr/local/lib/")
with
  - include_dirs.append("/usr/include/")
  - library_dirs.append("/usr/lib/")
in setup.py.

Outlook
---------
