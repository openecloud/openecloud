openecloud
==========

Particle-in-cell code for electron cloud studies. 


Dependencies
==========

The code was mainly used and tested on Ubuntu 12.04 64-bit systems with:
- Cython 0.19
- Numpy 1.7.1
- Scipy 0.12.0
- GLS 1.15

Only the latter is the standard repository version of Ubuntu. Otherwise older versions (e.g. standard Ubuntu) sometime exhibit faulty behavior. Short reasoning for the packages:

- Numpy (with C-bindings for Cython) and its ndarrays are used as data storage and interface between classes. This is intended for easy data handling from/with Python. Within the classes/functions typed memoryviews are used for performance.
- Scipy is used for some exotic functions (e.g. gammainc), but is increasingly replaced by Cython/C-functions or libraries for better performance.
- The SuperLU library is used to solve the Poisson problem (currently through the Scipy implementation). UMFPACK might be a better choice, or a slim Cython implementation of a suitable LU-decomposition.
- For random number generation the Mersenne-Twister of the GSL is used and some special functions. 
