import scipy as sp
import scipy.sparse as spsp
import scipy.sparse.linalg as spspl
import scipy.constants as spc


'''
Solves a poisson problem with the finite integration approach.

This method was written to be fast for solving the poisson problem
several times with different right hand side (i.e. the charge vector).
Change LU decomposition to iterative solver (e.g. spspl.bicgstab)
if used otherwise.

Requires scipy (sp), scipy.sparse (spsp), scipy.sparse.linalg (spspl)
and scipy.constants (spc).
'''
class PoissonSolver:
    
    
    ''' 
    Constructor.
    
    Input:
            - gridObj:        An instance of the class Grid which stores all information on the used grid.
            
    '''
    def __init__(self, gridObj):

        # Import needed grid parameters from the grid object.
        nx = gridObj.getNx();   ny = gridObj.getNy()
        self.insidePointsInd = gridObj.getInsidePointsInd()
        self.gridObj = gridObj
        np = nx*ny  
        
        # Discrete derivatives, negated and transposed.
        pxmt = spsp.identity(np)-spsp.eye(np,np,-1)
        pymt = spsp.identity(np)-spsp.eye(np,np,-nx)
        
        # Discrete source matrix
        st = spsp.hstack([pxmt, pymt])  
         
        # Discrete material matrix.
        meps = spc.epsilon_0*gridObj.getDsi()*sp.roll(gridObj.getDst(),np)
        
        # System matrix. Grid points with fixed potential are removed, as they are not needed.
        # The LU decomposition needs a spare matrix in the csc format.
        a = st.dot(spsp.diags(meps,0).dot(st.transpose()))[self.insidePointsInd[:,sp.newaxis],self.insidePointsInd].tocsc()
        self.a = a
        # LU decomposition. Uses the SuperLU library. Parameters set for a diagonal dominant and symmetric (both
        # approximately) matrix. 
        self.ludecom = spspl.splu(a,permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh = 0.1)
  
        # Generate output vectors for later.
        self.phi = sp.zeros(np)        
        self.eAtGridPoints = sp.zeros(2*np)

        # Combine discrete gradient to calculate electric field with the interpolation to grid points, 
        # so it can be done later in one swoop.
        self.phiToEAtGridPoints = self.gridObj.getEdgeToNode().dot(spsp.diags(self.gridObj.getDsi(),0,(2*np,2*np)).dot(st.transpose()))


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
    
    def getA(self):
        return self.a