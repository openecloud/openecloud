import numpy
cimport numpy
cimport grid
import scipy.sparse as spsp
import scipy.sparse.linalg as spspl
from constants cimport *
import timeit


'''
Solves a poisson problem with the finite integration approach.

This method was written to be fast for solving the poisson problem
several times with different right hand side (i.e. the charge vector).
Change LU decomposition to iterative solver (e.g. spspl.bicgstab)
if used otherwise.

Requires scipy.sparse (spsp) and scipy.sparse.linalg (spspl).
'''
cdef class PoissonSolver:
    
    cdef:
        object insidePointsInd, ludecom, phi, phiToEAtGridPoints
        double[:] eAtGridPoints
    
    ''' 
    Constructor.
    
    Input:
            - gridObj:        An instance of the class Grid which stores all information on the used grid.
            
    '''
    def __init__(PoissonSolver self, grid.Grid gridObj):
        cdef:
            unsigned int nx, np
            object pxmt, pymt, st, a
        # Import needed grid parameters from the grid object.
        nx = gridObj.getNxExt()
        np = gridObj.getNpExt()
        self.insidePointsInd = numpy.nonzero(gridObj.getInsidePoints())[0]
         
        
        # Discrete derivatives, negated and transposed.
        pxmt = spsp.identity(np)-spsp.eye(np,np,-1)
        pymt = spsp.identity(np)-spsp.eye(np,np,-nx)
        
        # Discrete source matrix
        st = spsp.hstack([pxmt, pymt])  
         
        # Discrete material matrix.
        meps = epsilon_0*numpy.asarray(gridObj.getDsi())*numpy.roll(gridObj.getDst(),np)
        
        # System matrix. Grid points with fixed potential are removed, as they are not needed.
        # The LU decomposition needs a spare matrix in the csc format.
        a = st.dot(spsp.diags(meps,0).dot(st.transpose()))[:,self.insidePointsInd].tocsc()[self.insidePointsInd,:]

        # LU decomposition. Uses the SuperLU library. Parameters set for a diagonal dominant and symmetric (both
        # approximately) matrix. 
        self.ludecom = spspl.splu(a,permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh = 0.1)
  
        # Generate output vectors for later.
        self.phi = numpy.zeros(np, dtype=numpy.double)        
        self.eAtGridPoints = numpy.zeros(2*np, dtype=numpy.double)

        # Combine discrete gradient to calculate electric field with the interpolation to grid points, 
        # so it can be done later in one swoop.
        self.phiToEAtGridPoints = gridObj.getEdgeToNode().dot(spsp.diags(gridObj.getDsi(),0,(2*np,2*np)).dot(st.transpose()))


    '''
    Solves the actual poisson problem. 
    
    Note that in finite integration the charge -- NOT the charge density --
    is the right hand side.
    
    Input:
           - q:                Charge in dual cells.
    
    Output:
           - eAtGridPoints:    Electric field at grid points.
    
    '''
    def solve(self, q):
        # Solve for potential.
        self.phi[self.insidePointsInd] = self.ludecom.solve(q.take(self.insidePointsInd))
        # Potential to electric field on grid points.
        self.eAtGridPoints = self.phiToEAtGridPoints.dot(self.phi)
        return self.eAtGridPoints
    
    
    '''
    Getter functions from here on.
    
    '''
    def getPhi(self):
        return self.phi

    def getEAtGridPoints(self):
        return self.eAtGridPoints
