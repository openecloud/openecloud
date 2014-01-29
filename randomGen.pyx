#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True

import time

# GSL imports.
cdef extern from "gsl/gsl_rng.h":
    ctypedef struct gsl_rng_type
    ctypedef struct gsl_rng
    cdef gsl_rng_type* gsl_rng_mt19937    
    
    gsl_rng* gsl_rng_alloc(gsl_rng_type* T) nogil
    double gsl_rng_uniform(gsl_rng* r) nogil
    void gsl_rng_set(gsl_rng* r, unsigned long int seed) nogil
    unsigned long gsl_rng_uniform_int(const gsl_rng* r, unsigned long n) nogil
    
cdef extern from "gsl/gsl_randist.h":
    double gsl_ran_gaussian_ziggurat(gsl_rng* r, double sigma) nogil
    
cdef gsl_rng* r = gsl_rng_alloc(gsl_rng_mt19937)
gsl_rng_set(r, <unsigned long> time.time()*256)
        
cdef void seed(unsigned long x):  
    gsl_rng_set(r, x)

cdef double rand():
    return gsl_rng_uniform(r)

cdef double randn():
    return gsl_ran_gaussian_ziggurat(r, 1.)

cdef unsigned long randi(unsigned long n):
    return gsl_rng_uniform_int(r, n)

