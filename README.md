openecloud
==========

Particle-in-cell code for electron cloud studies. 
Although the code already can be used for buidup simulation it will be extended and improved in future updates. We are very much interested in feedback and contributions of the community.


Module Descriptions
---------

Every module usually consists of a main class which is used to store relevant data for the specific purpose in an object. The object is then used to manipulate the conatining data with provided function. For example the class *particles* stores the particle coordinates and a push of these particle coordinates can be done with one of the *borisPush* functions.

### grid

A basic module to handle the properties of the computational grid, like grid lengths and number of cells in each direction, and data on the grid. Supports cut-cell boundaries. Only supports equidistant (in each direction) and cartesian grids for now, but this is mainly due to the particles module and not due to the grid and poissonSolver module. A grid object is handed to every function which needs data of or on the grid. Note that the grid boundary for field computations is implemented here, but particle boundaries are implemented in the *particleBoundary* class.

### poissonSolver

Implements a Poisson solver based on the Finite Integration Technique (which is very similar to Finite Differences). The major trick here usually overlooked in other codes is that it is much faster to do a LU-decomposition once of the system matrix at the beginning of the simulation and then only doing the backwards substitution at each time step than using an iterative solver. The performance with the Super-LU library is comparable to FFT-Poisson solvers but much more flexible with cut-cell boundaries.

### particles

Module to store and modifiy particle data, e.g. doing a Boris push.


### particleBoundary

This is more or less a helper module to implement different particle boundaries. This boundary is different from the grid boundary in in the *grid* class (although they should fit in all sensible cases).

### particleEmitter

This module implements different particle emitters which for example take the particles absorbed by the *particleBoundary* class to calculate secondary electrons emitted from the boundary.


### beam

Some simple methods to provide a rigid beam charge distribution which can be imprinted on the grid.


Dependencies
---------

The code was mainly used and tested on Ubuntu 12.04 64-bit systems with:
- Cython 0.19
- Numpy 1.7.1
- Scipy 0.12.0
- GLS 1.15

Only the latter is the standard repository version of Ubuntu. Otherwise older versions (e.g. standard Ubuntu) sometime exhibit faulty behavior. Short reasoning for the packages:

- Numpy (with C-bindings for Cython) and its ndarrays are used as data storage and interface between classes. This is intended for easy data handling from/with Python and to use the convenient garbage collection for the arrays. Within the classes/functions typed memoryviews are used for performance.
- Scipy is used for some exotic functions, but is increasingly replaced by Cython/C-functions or libraries for better performance. It is desirable to replace most or all Scipy dependencies in the future.
- For random number generation the Mersenne-Twister of the GSL is used and some special functions. 

Furthermore the SuperLU library is used to solve the Poisson problem (currently through the Scipy implementation). UMFPACK provides similar sophisticated LU-decomposition and might be a better choice, or a slim/tailered Cython implementation of a suitable LU-decomposition.
