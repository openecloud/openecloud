openecloud
==========

openECLOUD is a particle-in-cell code for electron-cloud studies. The code mainly focuses on the buildup of the electron cloud, but is intended to provide tracking studies together with [PyORBIT](https://code.google.com/p/py-orbit/) and methods for other related calculations. The code in general is intended as an open environment and community project. It is programmed in [Cython](http://cython.org/), which can be naively understood as a hybrid of Python and C/C++. The code can easily extended in Python, C/C++ or Cython; the latter is highly recommended.

Although the code already can be used for buidup simulation it will be extended and improved in future updates. We are very much interested in feedback and contributions of the community. The documentation will be extended in the future as well, but is intended to only contain a general description. For further details one can look at the examples or the comments in the source code.

Quick Start
---------

1. Download the latest version of the code from the [repository](https://github.com/openecloud/openecloud).
2. Make sure you have installed the dependencies (see Sec. **Dependencies**).
3. Run the setup (under Linux in the command line: python setup.py build_ext --inplace).
4. Run example01 (under Linux in the command line: python example01_basic.py).

New Changes
---------

- Added arbitrary cut-cell boundaries (see example05).
- Added a better particle management method *localRandom*. It is based on the [particle management with the help of kd-trees](http://arxiv.org/abs/1301.1552) (see example03).
- Added arbitrary magnetic fields in the *magneticFields* module (see example04).

Curent Issues
---------

- On some platforms the Poisson solver crashes in some cases at the construction of the system matrix. There is a problem with the scipy sparse matrices.



Module Descriptions
---------

Every module usually consists of a main class which is used to store relevant data for the specific purpose in an object. The object is then used to manipulate the containing data with provided function. For example the module *particles* contains the class *Particles* which stores the particle coordinates and a push of these particle coordinates can be done with one of the *borisPush* functions in the same module. Modules start with lowercase and classes with capital letters. Modules usually consist of the main class (with the same name, only capital initial letter) and some helper functions.

### grid

A basic module to handle the properties of the computational grid, like grid lengths and number of cells in each direction. Supports cut-cell boundaries. Only supports equidistant (in each direction) and cartesian grids for now, but this is mainly due to the particles module (performance!) and not due to the *grid* and *poissonSolver* module. A *Grid* object is handed to every function which needs data of or on the grid. Note that the grid boundary for field computations is implemented here, but particle boundaries are implemented in the *ParticleBoundary* class.

### poissonSolver

Implements a Poisson solver based on the Finite Integration Technique (which is very similar to Finite Differences). The major trick here usually overlooked in other codes is that it is much faster to do a LU-decomposition once of the system matrix at the beginning of the simulation and then only doing the backwards substitution at each time step than using an iterative solver. The performance with the Super-LU library is comparable to FFT-Poisson solvers on a rectangular domain but much more flexible with cut-cell boundaries.

### particles

Module to store and modifiy particle data, e.g. doing a Boris push.


### particleBoundary

This is more or less a helper module to implement different particle boundaries. Note that this boundary does not automatically represent the same physical boundary as the grid boundary in the *grid* module. Choose carefull.y

### particleEmitter

This module implements different particle emitters which for example take the particles absorbed by the *particleBoundary* class to calculate secondary electrons emitted from the boundary (most notably the [Furman-model](http://dx.doi.org/10.1103/PhysRevSTAB.5.124404 )). Contains loaders for inital particle distributions as well.

### particleManagement

Provides methods to manage the particle number in the simulations. This is necessary due to the exponential growth of the electrons in buildup simulations. Currently a very simple algorithm ("russian roulette") and [particle management with the help of kd-trees](http://arxiv.org/abs/1301.1552) are used.

### beam

Some simple methods to provide a rigid beam charge distribution which can be imprinted on the grid. The user can add his beam properties here.

### specFun

Wraps several special functions of the GSL mainly needed by the *particleEmitter* module and provides some additional ones.

### randomGen

Wraps some random number generation of the GSL.

### magneticField

Currently only provides a method to calculate an arbitrary magnetic field. 

### constants

Module to store and provide global constants like the speed of light.

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

Only GSL and GCC are the standard repository versions of Ubuntu 12.04. Otherwise older versions (e.g. standard Ubuntu) sometime exhibit faulty behavior. Newer versions of the packages should work most likely. There is no easy way to get the code working on Windows systems, but for all Linux systems it should be easy to adapt the code (see Sec. **Tips and Tricks**).

Short reasoning for the packages:

- Numpy with its C-bindings for Cython and its ndarrays are used as data storage and interface between classes. This is intended for easy data handling from/with Python and to use the convenient garbage collection for the arrays. Within the classes/functions typed memoryviews or raw C pointers are used for performance.
- Scipy is used for some exotic functions, but is increasingly replaced by Cython/C-functions or libraries for better performance. It is desirable to replace most or all Scipy dependencies in the future.
- For random number generation the Mersenne-Twister of the GSL is used and some special functions. 
- Matplotlib is mainly used in the examples for graphical output. It is highly recommended but isn't stricly necessary for running the code.
- To actually compile the Cython files a C compiler is obviously needed. The standards of Ubuntu 12.04 worked flawlessly for us. Install with *sudo apt-get install build-essential*, but this most likely is already on your system.

Furthermore the SuperLU library is used to solve the Poisson problem (currently through the Scipy implementation). UMFPACK provides similar sophisticated LU-decomposition and might be a better choice, or a slim/tailered Cython implementation of a suitable LU-decomposition.

Tips and Tricks
--------
- Depending on your system it may be necessary to replace 
  - include_dirs.append("/usr/local/include/")
  - library_dirs.append("/usr/local/lib/")
  
  with
  - include_dirs.append("/usr/include/")
  - library_dirs.append("/usr/lib/")
  
  in setup.py.


Outlook
---------

In this section some thoughts on future developments and plans are given. We would gladly accept any input from the community.

1. Include the generation of seed electrons by residual gas ionization, synchrotron radiation and beam losses. The problem here is that it is hard to do general models which are more than usable for just one specific scenario.
2. The coupling to PyORBIT already exists, but in a rather simple and unflexible form. The goal is to have several different coupling possiblities, e.g. up to a self-consistant one.
3. Already existing but not but in a publically usable form is the calculation of microwave transmission. This can be done with rather easy models (dielectric model for plasma and 2D) and on-the-fly during buildup simulations.
4. Include arbitrary staircase boundary for comparison with cut-cell one.

