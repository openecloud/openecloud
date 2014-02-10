import numpy
cimport numpy
cimport cython
from libc.stdlib cimport malloc, free

ctypedef struct node:
    unsigned int[:] inds
    

def testFunction():
    cdef node* testNode = <node*> malloc(sizeof(node))
    cdef unsigned int[:] tempInds
    tempInds = numpy.ones(3,dtype=numpy.uintc)
    testNode.inds = tempInds
