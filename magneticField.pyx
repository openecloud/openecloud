import numpy
cimport numpy
cimport cython
cimport particles
cimport grid
from constants cimport *

cdef extern from "math.h":
    double sqrt(double x) nogil
    double cos(double x) nogil
    double sin(double x) nogil
    double atan2(double y, double x) nogil

cdef inline unsigned int min(unsigned int a, unsigned int b):
    if a>=b:
        return a
    else:
        return b
 
# Multipole definition is such that no reference radius is used. The coefficients
# are in units of T/m^n, where n is the current order belonging to the respective coefficient.        
# Length of multipolesIn dictates order of expansion.         
cpdef numpy.ndarray multipoleExpansion(grid.Grid gridObj, multipolesIn, skewMultipolesIn = ()):
    cdef:
        unsigned int ii, jj, kk, currentInd
        double[:] multipoles = numpy.asarray(multipolesIn, dtype=numpy.double)
        unsigned int nMultipoles = multipoles.shape[0]
        double[:] skewMultipoles = numpy.zeros(nMultipoles, dtype=numpy.double)
        unsigned int nSkewMultipoles = skewMultipoles.shape[0]
        double[:] xMesh = gridObj.getXMesh()
        double[:] yMesh = gridObj.getYMesh()
        unsigned int nx = gridObj.getNx()
        unsigned int ny = gridObj.getNy()
        unsigned int np = gridObj.getNp()
        double[:] bAtGridPoints = numpy.empty(2*np, dtype=numpy.double)
        double r, phi, br, bphi
    
    for ii in range(min(nMultipoles,nSkewMultipoles)):
        skewMultipoles[ii] = skewMultipolesIn[ii]
    
    # Loops probably could be done more efficiently, but this suffices for one call for a simulation run.    
    for ii in range(nx):
        for jj in range(ny):            
            r = sqrt(xMesh[ii]**2 + yMesh[jj]**2)
            phi = atan2(yMesh[jj], xMesh[ii])
            if phi<0:
                phi+= 2*pi
            br = 0.
            bphi = 0.
            for kk in range(nMultipoles):
                br+= r**kk*(-skewMultipoles[kk]*cos((kk+1)*phi) + multipoles[kk]*sin((kk+1)*phi))
                bphi+= r**kk*(skewMultipoles[kk]*sin((kk+1)*phi) + multipoles[kk]*cos((kk+1)*phi))
            currentInd = ii+jj*nx    
            bAtGridPoints[currentInd] = br*cos(phi) - bphi*sin(phi)
            bAtGridPoints[currentInd+np] = br*sin(phi) + bphi*cos(phi)
    return numpy.asarray(bAtGridPoints)
    
