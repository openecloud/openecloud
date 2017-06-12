#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy
cimport numpy
cimport cython        
import scipy.sparse as spsp
from constants cimport *

# Import some C methods.
cdef extern from "math.h":
    double sin(double x) nogil     
    double cos(double x) nogil
    double asin(double x) nogil     
    double acos(double x) nogil
    double atan(double x) nogil
    double atan2(double x, double y) nogil
    double sqrt(double x) nogil
    double log2(double x) nogil

 
'''
Class to calculate and store grid quantities.
It looks like a lot happens here (especially the cut-cell stuff), 
but it is fairly simple. Just long.

Requires scipy.sparse (spsp) currently.
'''
cdef class Grid:

    '''
    Constructor.
    
    Input:
           - nx, ny:                   Number of grid points in x and y direction.
           - lx, ly:                   Grid size in x and y direction.
           - boundType:                Type of boundary. One of 0=rectangular, 1=elliptical or 2=arbitrary.
           - boundFunc:                Function which returns 1 if point is inside and 0 if outside domain (optional).
           - cutCellAcc:               Accuracy to which cut cell edge length are calculated (optional).
           - scanAroundPoint:          Relative area around each grid point which must be inside the grid 
                                       boundary for the point to be set as inside point. (optional).
                                       
    '''   
    def __init__(Grid self, unsigned int nx, unsigned int ny, double lx, double ly, 
                 unsigned short boundType, object boundFunc = None, double cutCellAcc = 1e-12, double scanAroundPoint = 1.e-3):

        # Calculate and store basic parameters of the uniform grid. For cut-cell some accuracy parameters.
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.dx = self.lx/(self.nx-1.)
        self.dy = self.ly/(self.ny-1.)
        self.cutCellAcc = cutCellAcc
        self.scanAroundPoint = scanAroundPoint    
        
        # Extend grid internally by 1 guard cell on each side. These guard cells are needed for
        # the particleBoundary and are outside of domain.
        self.nxExt = self.nx + 2
        self.nyExt = self.ny + 2
        self.npExt = self.nxExt*self.nyExt
        self.lxExt = self.lx + 2.*self.dx
        self.lyExt = self.ly + 2.*self.dy
        
        # Some position arrays for later.
        self.xMesh = numpy.linspace(-self.lxExt/2., self.lxExt/2, self.nxExt)
        self.yMesh = numpy.linspace(-self.lyExt/2., self.lyExt/2, self.nyExt) 

        # Allocate memory. Numpy is fast enough here and more convenient (garbage collection!).
        self.insidePoints = numpy.empty(self.npExt, dtype=numpy.ushort)
        self.insideEdges = numpy.empty(2*self.npExt, dtype=numpy.ushort)
        self.insideFaces = numpy.empty(self.npExt, dtype=numpy.ushort)
        self.ds = numpy.empty(2*self.npExt, dtype=numpy.double)
        self.dsi = numpy.empty(2*self.npExt, dtype=numpy.double)        
        self.da = numpy.empty(self.npExt, dtype=numpy.double)
        self.dai = numpy.empty(self.npExt, dtype=numpy.double)        
        self.dst = numpy.empty(2*self.npExt, dtype=numpy.double)
        self.dsti = numpy.empty(2*self.npExt, dtype=numpy.double)        
        self.dat = numpy.empty(self.npExt, dtype=numpy.double)
        self.dati = numpy.empty(self.npExt, dtype=numpy.double)  
        
        # Calculate other grid parameters depending on boundary.
        if boundType == 0:
            self.boundFunc = lambda x,y: _boundFuncRectangular(x,y,self.lx*0.5,self.ly*0.5)
            self.computeStaircaseGridGeom()  
        elif boundType == 1:
            self.boundFunc = lambda x,y: _boundFuncElliptical(x,y,4./self.lx**2,4./self.ly**2)
            self.computeCutCellGridGeom()
        elif boundType == 2:
            self.boundFunc = boundFunc
            self.computeStaircaseGridGeom()
        elif boundType == 3:
            self.boundFunc = boundFunc
            self.computeCutCellGridGeom()
        else:
            raise NotImplementedError("Not yet implemented.")
            

 
    '''
    Helper function for the setup of staircase geometries. Calculates grid quantities.
    
    '''      
    cdef void computeStaircaseGridGeom(Grid self):
        cdef:
            unsigned int ii, jj, kk
            unsigned int nx = self.nxExt, ny = self.nyExt, np = self.npExt
            double dx = self.dx, dy = self.dy, dxi = 1./dx, dyi = 1./dy
            double dxHalf = 0.5*dx, dyHalf = 0.5*dy, dxHalfi = 1./dxHalf, dyHalfi = 1./dyHalf
            double dxdy = dx*dy, dxdyi = 1./dxdy
            double dxMin = self.scanAroundPoint*dx, dyMin = self.scanAroundPoint*dy
            unsigned short* insidePoints = &self.insidePoints[0]
            unsigned short* insideEdges = &self.insideEdges[0]
            unsigned short* insideFaces = &self.insideFaces[0]
            double* ds = &self.ds[0]
            double* dsi = &self.dsi[0]
            double* da = &self.da[0]
            double* dai = &self.dai[0]
            double* dst = &self.dst[0]
            double* dsti = &self.dsti[0]
            double* dat = &self.dat[0]
            double* dati = &self.dati[0]  
            double* xMesh = &self.xMesh[0]
            double* yMesh = &self.yMesh[0]
            unsigned int[:] rows = numpy.empty(6*np, dtype=numpy.uintc)        # In worst case interpolation matrix has
            unsigned int[:] columns = numpy.empty(6*np, dtype=numpy.uintc)     # quite a lot of non-zero elements.
            double[:] values = numpy.empty(6*np, dtype=numpy.double)                                       
            object boundFunc = self.boundFunc
            
        # Check if grid point lies inside domain with some given tolerance.
        for ii in range(nx):
            insidePoints[ii] = 0
            insidePoints[ii+nx] = 0
            insidePoints[ii+(ny-2)*nx] = 0
            insidePoints[ii+(ny-1)*nx] = 0
        for jj in range(2,ny-2):
            insidePoints[jj*nx] = 0
            insidePoints[1+jj*nx] = 0
            insidePoints[(nx-2)+jj*nx] = 0  
            insidePoints[(nx-1)+jj*nx] = 0  
        for ii in range(2,nx-2):
            for jj in range(2,ny-2):
                if (boundFunc(xMesh[ii]+dxMin,yMesh[jj]+dyMin)==0 or
                    boundFunc(xMesh[ii]+dxMin,yMesh[jj]-dyMin)==0 or
                    boundFunc(xMesh[ii]-dxMin,yMesh[jj]+dyMin)==0 or
                    boundFunc(xMesh[ii]-dxMin,yMesh[jj]-dyMin)==0):
                    insidePoints[ii+jj*nx] = 0  
                else:
                    insidePoints[ii+jj*nx] = 1
       
        # Classify edges. 0 = outside, 1 = boundary edges and 2 = normal inside edges.  
        for ii in range(nx):
            insideEdges[ii] = 0
            insideEdges[ii+nx] = 0
            insideEdges[ii+(ny-2)*nx] = 0
            insideEdges[ii+(ny-1)*nx] = 0
            insideEdges[ii+np] = 0
            insideEdges[ii+(ny-1)*nx+np] = 0
        for jj in range(ny):
            insideEdges[jj*nx+np] = 0
            insideEdges[1+jj*nx+np] = 0
            insideEdges[(nx-2)+jj*nx+np] = 0
            insideEdges[(nx-1)+jj*nx+np] = 0
            insideEdges[jj*nx] = 0
            insideEdges[(nx-1)+jj*nx] = 0
        for jj in range(2,ny-2):
            for ii in range(1,nx-1):          
                insideEdges[ii+jj*nx] = insidePoints[ii+jj*nx] + insidePoints[ii+1+jj*nx]
        for jj in range(1,ny-1):
            for ii in range(2,nx-2):    
                insideEdges[ii+jj*nx+np] = insidePoints[ii+jj*nx] + insidePoints[ii+(jj+1)*nx]

        # Edges length. Set known values manually (e.g. guard cells). Fast!
        for ii in range(nx):
            ds[ii] = 0.
            ds[ii+nx] = 0.
            ds[ii+(ny-2)*nx] = 0.
            ds[ii+(ny-1)*nx] = 0.
            ds[ii+np] = 0.
            ds[ii+(ny-2)*nx+np] = 0.
            ds[ii+(ny-1)*nx+np] = 0.
            dsi[ii] = 0.
            dsi[ii+nx] = 0.
            dsi[ii+(ny-2)*nx] = 0.
            dsi[ii+(ny-1)*nx] = 0.
            dsi[ii+np] = 0.
            dsi[ii+(ny-2)*nx+np] = 0.
            dsi[ii+(ny-1)*nx+np] = 0.
        for jj in range(1,ny-2):
            ds[jj*nx+np] = 0.
            ds[1+jj*nx+np] = 0.
            ds[(nx-2)+jj*nx+np] = 0.
            ds[(nx-1)+jj*nx+np] = 0.
            dsi[jj*nx+np] = 0.
            dsi[1+jj*nx+np] = 0.
            dsi[(nx-2)+jj*nx+np] = 0.
            dsi[(nx-1)+jj*nx+np] = 0.
        for jj in range(2,ny-2):    
            ds[jj*nx] = 0.
            ds[(nx-2)+jj*nx] = 0.
            ds[(nx-1)+jj*nx] = 0. 
            dsi[jj*nx] = 0.
            dsi[(nx-2)+jj*nx] = 0.
            dsi[(nx-1)+jj*nx] = 0.        
        # X-edges.     
        for jj in range(2,ny-2):
            for ii in range(1,nx-2): 
                if insideEdges[ii+jj*nx]==0:
                    ds[ii+jj*nx] = 0.
                    dsi[ii+jj*nx] = 0.
                else:
                    ds[ii+jj*nx] = dx                                   # Only equidistant (in each direction) grid here.
                    dsi[ii+jj*nx] = dxi                                     
        # Y-edges.                        
        for jj in range(1,ny-2):
            for ii in range(2,nx-2): 
                if insideEdges[ii+jj*nx+np]==0:
                    ds[ii+jj*nx+np] = 0.
                    dsi[ii+jj*nx+np] = 0.
                else:
                    ds[ii+jj*nx+np] = dy                                # Only equidistant (in each direction) grid here.
                    dsi[ii+jj*nx+np] = dyi
                 
       

        # Calculation of faces. Primary grid. 
        # Note that these are not used for any essential calculation if a Poisson solver is used.      
        for ii in range(nx):
            insideFaces[ii] = 0
            insideFaces[ii+(ny-2)*nx] = 0
            insideFaces[ii+(ny-1)*nx] = 0
        for jj in range(1,ny-2):
            insideFaces[jj*nx] = 0
            insideFaces[(nx-2)+jj*nx] = 0
            insideFaces[(nx-1)+jj*nx] = 0
        for jj in range(1,ny-2):
            for ii in range(1,nx-2):
                insideFaces[ii+jj*nx] = insideEdges[ii+jj*nx] + insideEdges[ii+(jj+1)*nx] + \
                                        insideEdges[ii+jj*nx+np] + insideEdges[ii+1+jj*nx+np]       
        for ii in range(nx):
            da[ii] = 0.
            da[ii+(ny-2)*nx] = 0.
            da[ii+(ny-1)*nx] = 0.
            dai[ii] = 0.
            dai[ii+(ny-2)*nx] = 0.
            dai[ii+(ny-1)*nx] = 0.
        for jj in range(1,ny-2):
            da[jj*nx] = 0.
            da[(nx-2)+jj*nx] = 0.
            da[(nx-1)+jj*nx] = 0.
            dai[jj*nx] = 0.
            dai[(nx-2)+jj*nx] = 0.
            dai[(nx-1)+jj*nx] = 0.
        
        for jj in range(1,ny-2):
            for ii in range(1,nx-2):
                if insideFaces[ii+jj*nx]==0:
                    da[ii+jj*nx] = 0.
                    dai[ii+jj*nx] = 0.
                else:
                    da[ii+jj*nx] = dxdy
                    dai[ii+jj*nx] = dxdyi                    

        # Dual grid quantities are not affected. Always same as for rectangular. 
        # They are not needed for any essential
        # calculation in the Poisson problem, just for visualization of the charge density.
        for ii in range(nx):
            dst[ii] = 0.
            dst[ii+(ny-1)*nx] = 0.
            dst[ii+np] = 0.
            dst[ii+(ny-1)*nx+np] = 0.
            dsti[ii] = 0.
            dsti[ii+(ny-1)*nx] = 0.
            dsti[ii+np] = 0.
            dsti[ii+(ny-1)*nx+np] = 0.
        for jj in range(1,ny-1):
            dst[jj*nx] = 0.
            dst[(nx-1)+jj*nx] = 0.
            dst[jj*nx+np] = 0.
            dst[(nx-1)+jj*nx+np] = 0.
            dsti[jj*nx] = 0.
            dsti[(nx-1)+jj*nx] = 0.
            dsti[jj*nx+np] = 0.
            dsti[(nx-1)+jj*nx+np] = 0.
        for ii in range(2,nx-2):
            for jj in range(1,ny-1):
                dst[ii+jj*nx] = dx                          # Here assuming equidistant meshes (in each direction).
                dsti[ii+jj*nx] = dxi
        for ii in range(1,nx-1):
            for jj in range(2,ny-2):
                dst[ii+jj*nx+np] = dy
                dsti[ii+jj*nx+np] = dyi
        for jj in range(1,ny-1):
            dst[1+jj*nx] = dxHalf
            dst[(nx-2)+jj*nx] = dxHalf
            dsti[1+jj*nx] = dxHalfi
            dsti[(nx-2)+jj*nx] = dxHalfi
        for ii in range(1,nx-1):
            dst[ii+nx+np] = dyHalf
            dst[ii+(ny-2)*nx+np] = dyHalf
            dsti[ii+nx+np] = dyHalfi
            dsti[ii+(ny-2)*nx+np] = dyHalfi
        for ii in range(nx):
            for jj in range(ny):
                dat[ii+jj*nx] = dst[ii+jj*nx]*dst[ii+jj*nx+np]
                dati[ii+jj*nx] = dsti[ii+jj*nx]*dsti[ii+jj*nx+np] 

        
        # Standard interpolation stuff. Interpolation from edges to _inside_ grid points. Same as rectangular.         
        kk = 0
        for ii in range(1,nx-2):
            for jj in range(2,ny-2):
                currentInd = ii + jj*nx
                if insidePoints[currentInd]==1:
                    rows[kk] = currentInd
                    columns[kk] = currentInd
                    values[kk] = ds[currentInd]/(ds[currentInd-1]+ds[currentInd])   # Should works for non-equidistant grid.  
                    kk += 1
        for ii in range(2,nx-1):
            for jj in range(2,ny-2):
                currentInd = ii + jj*nx
                if insidePoints[currentInd]==1:
                    rows[kk] = currentInd
                    columns[kk] = currentInd-1
                    values[kk] = ds[currentInd-1]/(ds[currentInd-1]+ds[currentInd])
                    kk += 1
        for ii in range(2,nx-2):
            for jj in range(1,ny-2):
                currentInd = ii + jj*nx
                if insidePoints[currentInd]==1:
                    currentInd += np 
                    rows[kk] = currentInd
                    columns[kk] = currentInd
                    values[kk] = ds[currentInd]/(ds[currentInd-nx]+ds[currentInd])
                    kk += 1
        for ii in range(2,nx-2):
            for jj in range(2,ny-1):   
                currentInd = ii + jj*nx
                if insidePoints[currentInd]==1:      
                    currentInd += np      
                    rows[kk] = currentInd
                    columns[kk] = currentInd-nx
                    values[kk] = ds[currentInd-nx]/(ds[currentInd-nx]+ds[currentInd])
                    kk += 1
        
        self.edgeToNode = spsp.csc_matrix(spsp.coo_matrix((values[:kk],(rows[:kk],columns[:kk])),shape=(2*np,2*np)))
        
        
        # Boundary interpolation for correct fields at boundary.
        kk = 0
        for jj in range(1,ny-1):
            for ii in range(1,nx-1):
                currentInd = ii+jj*nx
                # Inside points are unchanged.
                if insidePoints[currentInd]==1:
                    rows[kk] = currentInd
                    columns[kk] = currentInd
                    values[kk] = 1.              
                    kk += 1  
                    rows[kk] = currentInd+np
                    columns[kk] = currentInd+np
                    values[kk] = 1.              
                    kk += 1    
                # Boundary edges.
                else:
                    if insideEdges[currentInd]==1:
                        # X in negative x-direction.
                        if insidePoints[currentInd+nx]==1 and insidePoints[currentInd-nx]==1:
                            rows[kk] = currentInd
                            columns[kk] = currentInd+1
                            values[kk] = 1.
                            kk += 1                       
                    elif insideEdges[currentInd-1]==1:
                        # X in positive x-direction
                        if insidePoints[currentInd+nx]==1 and insidePoints[currentInd-nx]==1:
                            rows[kk] = currentInd
                            columns[kk] = currentInd-1
                            values[kk] = 1.
                            kk += 1 
                    if insideEdges[currentInd+np]==1:
                        # Y in negative y-direction.
                        if insidePoints[currentInd+1]==1 and insidePoints[currentInd-1]==1:
                            rows[kk] = currentInd
                            columns[kk] = currentInd+nx
                            values[kk] = 1.
                            kk += 1
                    elif insideEdges[currentInd+np-nx]==1:
                        # Y in positive y-direction.
                        if insidePoints[currentInd+1]==1 and insidePoints[currentInd-1]==1:
                            rows[kk] = currentInd
                            columns[kk] = currentInd-nx
                            values[kk] = 1.
                            kk += 1
        
        self.edgeToNode = spsp.csc_matrix(spsp.coo_matrix((values[:kk],(rows[:kk],columns[:kk])),shape=(2*np,2*np))
                                          ).dot(self.edgeToNode)    

        
        
        
            
    '''
    Helper function for the setup of cut-cell geometries. Calculates grid quantities.
    
    '''      
    cdef void computeCutCellGridGeom(Grid self):
        cdef:
            unsigned int ii, jj, kk
            unsigned int nx = self.nxExt, ny = self.nyExt, np = self.npExt
            double dx = self.dx, dy = self.dy, dxi = 1./dx, dyi = 1./dy
            double dxHalf = 0.5*dx, dyHalf = 0.5*dy, dxHalfi = 1./dxHalf, dyHalfi = 1./dyHalf
            double dxdy = dx*dy, dxdyi = 1./dxdy
            double dxMin = self.scanAroundPoint*dx, dyMin = self.scanAroundPoint*dy
            double cutCellAcc = self.cutCellAcc
            unsigned int boundAccIter = <unsigned int> (log2(1./cutCellAcc)+2.) # Number of iterations for given accuracy. 
            unsigned short* insidePoints = &self.insidePoints[0]
            unsigned short* insideEdges = &self.insideEdges[0]
            unsigned short* insideFaces = &self.insideFaces[0]
            double* ds = &self.ds[0]
            double* dsi = &self.dsi[0]
            double* da = &self.da[0]
            double* dai = &self.dai[0]
            double* dst = &self.dst[0]
            double* dsti = &self.dsti[0]
            double* dat = &self.dat[0]
            double* dati = &self.dati[0]  
            double* xMesh = &self.xMesh[0]
            double* yMesh = &self.yMesh[0]
            double* boundaryPoints
            int* cutCellPointsInd
            unsigned int* boundaryPointsInd
            unsigned int nBoundaryPoints
            double* cutCellCenter
            double* cutCellNormalVectors
            unsigned int[:] rows = numpy.empty(18*np, dtype=numpy.uintc)        # In worst case interpolation matrix has
            unsigned int[:] columns = numpy.empty(18*np, dtype=numpy.uintc)     # quite a lot of non-zero elements.
            double[:] values = numpy.empty(18*np, dtype=numpy.double)     
            double[:] connectedEdgesInv = numpy.empty(np, dtype=numpy.double)
            double[:] cosAlpha = numpy.empty(np, dtype=numpy.double)
            double[:] sinAlpha = numpy.empty(np, dtype=numpy.double)
            double[:] cosBeta = numpy.empty(np, dtype=numpy.double)
            double[:] sinBeta = numpy.empty(np, dtype=numpy.double)                                    
            double x1, x2, x3, y
            double y1, y2, y3, x
            unsigned int tempuint
            double tempdouble
            unsigned short test01, test02, test03
            object boundFunc = self.boundFunc
            double bestDis
            unsigned int bestInd
            unsigned int searchInd1, searchInd2
            
        # Check if grid point lies inside domain with some given tolerance.
        for ii in range(nx):
            insidePoints[ii] = 0
            insidePoints[ii+nx] = 0
            insidePoints[ii+(ny-2)*nx] = 0
            insidePoints[ii+(ny-1)*nx] = 0
        for jj in range(2,ny-2):
            insidePoints[jj*nx] = 0
            insidePoints[1+jj*nx] = 0
            insidePoints[(nx-2)+jj*nx] = 0  
            insidePoints[(nx-1)+jj*nx] = 0  
        for ii in range(2,nx-2):
            for jj in range(2,ny-2):
                if (boundFunc(xMesh[ii]+dxMin,yMesh[jj]+dyMin)==0 or
                    boundFunc(xMesh[ii]+dxMin,yMesh[jj]-dyMin)==0 or
                    boundFunc(xMesh[ii]-dxMin,yMesh[jj]+dyMin)==0 or
                    boundFunc(xMesh[ii]-dxMin,yMesh[jj]-dyMin)==0):
                    insidePoints[ii+jj*nx] = 0  
                else:
                    insidePoints[ii+jj*nx] = 1
       
        # Classify edges. 0 = outside, 1 = possible cut cell edges and 2 = normal inside edges.  
        for ii in range(nx):
            insideEdges[ii] = 0
            insideEdges[ii+nx] = 0
            insideEdges[ii+(ny-2)*nx] = 0
            insideEdges[ii+(ny-1)*nx] = 0
            insideEdges[ii+np] = 0
            insideEdges[ii+(ny-1)*nx+np] = 0
        for jj in range(ny):
            insideEdges[jj*nx+np] = 0
            insideEdges[1+jj*nx+np] = 0
            insideEdges[(nx-2)+jj*nx+np] = 0
            insideEdges[(nx-1)+jj*nx+np] = 0
            insideEdges[jj*nx] = 0
            insideEdges[(nx-1)+jj*nx] = 0
        for jj in range(2,ny-2):
            for ii in range(1,nx-1):          
                insideEdges[ii+jj*nx] = insidePoints[ii+jj*nx] + insidePoints[ii+1+jj*nx]
        for jj in range(1,ny-1):
            for ii in range(2,nx-2):    
                insideEdges[ii+jj*nx+np] = insidePoints[ii+jj*nx] + insidePoints[ii+(jj+1)*nx]

        # Edges length. Set known values manually (e.g. guard cells). Fast!
        for ii in range(nx):
            ds[ii] = 0.
            ds[ii+nx] = 0.
            ds[ii+(ny-2)*nx] = 0.
            ds[ii+(ny-1)*nx] = 0.
            ds[ii+np] = 0.
            ds[ii+(ny-2)*nx+np] = 0.
            ds[ii+(ny-1)*nx+np] = 0.
            dsi[ii] = 0.
            dsi[ii+nx] = 0.
            dsi[ii+(ny-2)*nx] = 0.
            dsi[ii+(ny-1)*nx] = 0.
            dsi[ii+np] = 0.
            dsi[ii+(ny-2)*nx+np] = 0.
            dsi[ii+(ny-1)*nx+np] = 0.
        for jj in range(1,ny-2):
            ds[jj*nx+np] = 0.
            ds[1+jj*nx+np] = 0.
            ds[(nx-2)+jj*nx+np] = 0.
            ds[(nx-1)+jj*nx+np] = 0.
            dsi[jj*nx+np] = 0.
            dsi[1+jj*nx+np] = 0.
            dsi[(nx-2)+jj*nx+np] = 0.
            dsi[(nx-1)+jj*nx+np] = 0.
        for jj in range(2,ny-2):    
            ds[jj*nx] = 0.
            ds[(nx-2)+jj*nx] = 0.
            ds[(nx-1)+jj*nx] = 0. 
            dsi[jj*nx] = 0.
            dsi[(nx-2)+jj*nx] = 0.
            dsi[(nx-1)+jj*nx] = 0.        
        
        
        for jj in range(2,ny-2):
            for ii in range(1,nx-2): 
                currentInd = ii+jj*nx
                if insideEdges[currentInd]==2:
                    ds[currentInd] = dx                                   # Only equidistant (in each direction) grid here.
                    dsi[currentInd] = dxi
                elif insideEdges[currentInd]==0:
                    ds[currentInd] = 0.
                    dsi[currentInd] = 0.
                else:   
                    # Cut-cell edge.
                    # Bisect repeatedly the possible cut cell edges to get length of each. 
                    x1 = xMesh[ii]
                    x2 = xMesh[ii+1]
                    y = yMesh[jj]
                    test01 = boundFunc(x1,y)
                    test02 = boundFunc(x2,y)
                    if test01 == 1 and test02 == 1:
                        ds[currentInd] = dx#(1.-self.scanAroundPoint)*dx           # WORKAROUND. Maybe better possible.
                    else:
                        for kk in range(boundAccIter):
                            x3 = 0.5*(x1+x2)
                            test03 = boundFunc(x3,y)
                            if test01==test03:
                                x1 = x3
                                test01 = test03
                            else:
                                x2 = x3
                                test02 = test03
                        if insidePoints[currentInd]==1:
                            ds[currentInd] = 0.5*(x1+x2) - xMesh[ii]
                        else:
                            ds[currentInd] = xMesh[ii+1] - 0.5*(x1+x2)    
                    dsi[currentInd] = 1./ds[currentInd]
                        
        # Same stuff as above, only for y-edges.                        
        for jj in range(1,ny-2):
            for ii in range(2,nx-2): 
                currentInd = ii+jj*nx
                if insideEdges[currentInd+np]==2:
                    ds[currentInd+np] = dy                                   # Only equidistant (in each direction) grid here.
                    dsi[currentInd+np] = dyi
                elif insideEdges[currentInd+np]==0:
                    ds[currentInd+np] = 0.
                    dsi[currentInd+np] = 0.
                else:   
                    # Cut-cell edge.
                    # Bisect repeatedly the possible cut cell edges to get length of each. 
                    y1 = yMesh[jj]
                    y2 = yMesh[jj+1]
                    x = xMesh[ii]
                    test01 = boundFunc(x,y1)
                    test02 = boundFunc(x,y2)
                    if test01 == 1 and test02 == 1:
                        ds[currentInd+np] = dy#(1.-self.scanAroundPoint)*dy        # WORKAROUND. Maybe better possible.
                    else:
                        for kk in range(boundAccIter):
                            y3 = 0.5*(y1+y2)
                            test03 = boundFunc(x,y3)
                            if test01==test03:
                                y1 = y3
                                test01 = test03
                            else:
                                y2 = y3
                                test02 = test03
                        if insidePoints[currentInd]==1:
                            ds[currentInd+np] = 0.5*(y1+y2) - yMesh[jj]
                        else:
                            ds[currentInd+np] = yMesh[jj+1] - 0.5*(y1+y2)    
                    dsi[currentInd+np] = 1./ds[currentInd+np]                        
       

        # Determination of boundary points.
        # And save indices of moved grid points due to cut-cell for particle boundary later.
        nBoundaryPoints = 0
        for jj in range(1,ny-1):
            for ii in range(1,nx-1):
                currentInd = ii+jj*nx
                if insideEdges[currentInd]==1:
                    nBoundaryPoints += 1
                if insideEdges[currentInd+np]==1:
                    nBoundaryPoints += 1
        self.boundaryPoints = numpy.empty((nBoundaryPoints,2), dtype=numpy.double)
        boundaryPoints = &self.boundaryPoints[0,0]
        self.boundaryPointsInd = numpy.empty(nBoundaryPoints, dtype=numpy.uintc) 
        boundaryPointsInd = &self.boundaryPointsInd[0]
        self.cutCellPointsInd = -numpy.ones((np,2), dtype=numpy.intc)
        cutCellPointsInd = &self.cutCellPointsInd[0,0] 
        kk = 0
        for jj in range(1,ny-1):
            for ii in range(1,nx-1):
                currentInd = ii+jj*nx
                if insideEdges[currentInd]==1:
                    if insidePoints[currentInd]==1:
                        boundaryPoints[2*kk] = xMesh[ii] + ds[currentInd]
                        boundaryPoints[2*kk+1] = yMesh[jj]
                        boundaryPointsInd[kk] = currentInd+1
                    else:
                        boundaryPoints[2*kk] = xMesh[ii+1] - ds[currentInd]
                        boundaryPoints[2*kk+1] = yMesh[jj]
                        boundaryPointsInd[kk] = currentInd
                    if cutCellPointsInd[2*currentInd]==-1:
                        cutCellPointsInd[2*currentInd] = kk
                    else:
                        cutCellPointsInd[2*currentInd+1] = kk
                    if cutCellPointsInd[2*(currentInd-nx)]==-1:
                        cutCellPointsInd[2*(currentInd-nx)] = kk
                    else:
                        cutCellPointsInd[2*(currentInd-nx)+1] = kk
                    kk += 1
                if insideEdges[currentInd+np]==1:
                    if insidePoints[currentInd]==1:
                        boundaryPoints[2*kk] = xMesh[ii]
                        boundaryPoints[2*kk+1] = yMesh[jj] + ds[currentInd+np]
                        boundaryPointsInd[kk] = currentInd+nx+np
                    else:
                        boundaryPoints[2*kk] = xMesh[ii]
                        boundaryPoints[2*kk+1] = yMesh[jj+1] - ds[currentInd+np]
                        boundaryPointsInd[kk] = currentInd+np
                    if cutCellPointsInd[2*currentInd]==-1:
                        cutCellPointsInd[2*currentInd] = kk
                    else:
                        cutCellPointsInd[2*currentInd+1] = kk
                    if cutCellPointsInd[2*(currentInd-1)]==-1:
                        cutCellPointsInd[2*(currentInd-1)] = kk
                    else:
                        cutCellPointsInd[2*(currentInd-1)+1] = kk
                    kk += 1
        
        # Calculation of faces. Primary grid. Linear approximation.
        # Note that these are not used for any essential calculation if a Poisson solver is used.      
        for ii in range(nx):
            insideFaces[ii] = 0
            insideFaces[ii+(ny-2)*nx] = 0
            insideFaces[ii+(ny-1)*nx] = 0
        for jj in range(1,ny-2):
            insideFaces[jj*nx] = 0
            insideFaces[(nx-2)+jj*nx] = 0
            insideFaces[(nx-1)+jj*nx] = 0
        for jj in range(1,ny-2):
            for ii in range(1,nx-2):
                insideFaces[ii+jj*nx] = insideEdges[ii+jj*nx] + insideEdges[ii+(jj+1)*nx] + \
                                        insideEdges[ii+jj*nx+np] + insideEdges[ii+1+jj*nx+np]
        
        # Sort boundary points of each cell such that counter-clockwise.
        self.cutCellCenter = numpy.zeros((np,2), dtype=numpy.double)
        cutCellCenter = &self.cutCellCenter[0,0] 
        for jj in range(1,ny-1):
            for ii in range(1,nx-1):
                currentInd = ii + jj*nx
                if insideFaces[currentInd] != 0 and insideFaces[currentInd] != 8:     
                    # First calculate center of faces.          
                    if insideFaces[currentInd]==2:
                        if insidePoints[currentInd]==1:
                            cutCellCenter[2*currentInd] = xMesh[ii] + ds[currentInd]/3.
                            cutCellCenter[2*currentInd+1] = yMesh[jj] + ds[currentInd+np]/3.
                        elif insidePoints[currentInd+1]==1:
                            cutCellCenter[2*currentInd] = xMesh[ii+1] - ds[currentInd]/3.
                            cutCellCenter[2*currentInd+1] = yMesh[jj] + ds[currentInd+np+1]/3.
                        elif insidePoints[currentInd+nx]==1:
                            cutCellCenter[2*currentInd] = xMesh[ii] + ds[currentInd+nx]/3.
                            cutCellCenter[2*currentInd+1] = yMesh[jj+1] - ds[currentInd+np]/3.
                        elif insidePoints[currentInd+nx+1]==1:
                            cutCellCenter[2*currentInd] = xMesh[ii+1] - ds[currentInd+nx]/3.
                            cutCellCenter[2*currentInd+1] = yMesh[jj+1] - ds[currentInd+np+1]/3.
                    elif insideFaces[currentInd]==4:
                        if insidePoints[currentInd]==1 and insidePoints[currentInd+1]==1:
                            cutCellCenter[2*currentInd] = (xMesh[ii] + xMesh[ii+1])*0.5
                            cutCellCenter[2*currentInd+1] = yMesh[jj] + (ds[currentInd+np] + ds[currentInd+np+1])*0.25
                        elif insidePoints[currentInd+nx]==1 and insidePoints[currentInd+nx+1]==1:
                            cutCellCenter[2*currentInd] = (xMesh[ii] + xMesh[ii+1])*0.5
                            cutCellCenter[2*currentInd+1] = yMesh[jj+1] - (ds[currentInd+np] + ds[currentInd+np+1])*0.25
                        elif insidePoints[currentInd]==1 and insidePoints[currentInd+nx]==1:
                            cutCellCenter[2*currentInd] = xMesh[ii] + (ds[currentInd] + ds[currentInd+nx])*0.25
                            cutCellCenter[2*currentInd+1] = (yMesh[jj] + yMesh[jj+1])*0.5
                        elif insidePoints[currentInd+1]==1 and insidePoints[currentInd+nx+1]==1:
                            cutCellCenter[2*currentInd] = xMesh[ii+1] - (ds[currentInd] + ds[currentInd+nx])*0.25
                            cutCellCenter[2*currentInd+1] = (yMesh[jj] + yMesh[jj+1])*0.5
                    elif insideFaces[currentInd]==6:
                        if insidePoints[currentInd]==0:
                            cutCellCenter[2*currentInd] = (2.*xMesh[ii] + 3.*xMesh[ii+1] - ds[currentInd])*0.2
                            cutCellCenter[2*currentInd+1] = (2.*yMesh[jj] + 3.*yMesh[jj+1] - ds[currentInd+np])*0.2
                        elif insidePoints[currentInd+1]==0:
                            cutCellCenter[2*currentInd] = (3.*xMesh[ii] + 2.*xMesh[ii+1] + ds[currentInd])*0.2
                            cutCellCenter[2*currentInd+1] = (2.*yMesh[jj] + 3.*yMesh[jj+1] - ds[currentInd+np+1])*0.2
                        elif insidePoints[currentInd+nx]==0:
                            cutCellCenter[2*currentInd] = (2.*xMesh[ii] + 3.*xMesh[ii+1] - ds[currentInd+nx])*0.2
                            cutCellCenter[2*currentInd+1] = (3.*yMesh[jj] + 2.*yMesh[jj+1] + ds[currentInd+np])*0.2
                        elif insidePoints[currentInd+nx+1]==0:
                            cutCellCenter[2*currentInd] = (3.*xMesh[ii] + 2.*xMesh[ii+1] + ds[currentInd+nx])*0.2
                            cutCellCenter[2*currentInd+1] = (3.*yMesh[jj] + 2.*yMesh[jj+1] + ds[currentInd+np+1])*0.2
                    # Now sort boundary points in cutCellPointsInd counter-clockwise.
                    if ( (boundaryPoints[2*cutCellPointsInd[2*currentInd]] - cutCellCenter[2*currentInd]) *
                         (boundaryPoints[2*cutCellPointsInd[2*currentInd+1]+1] - cutCellCenter[2*currentInd+1]) <
                         (boundaryPoints[2*cutCellPointsInd[2*currentInd]+1] - cutCellCenter[2*currentInd+1]) *
                         (boundaryPoints[2*cutCellPointsInd[2*currentInd+1]] - cutCellCenter[2*currentInd]) ):
                        tempuint = cutCellPointsInd[2*currentInd]
                        cutCellPointsInd[2*currentInd] = cutCellPointsInd[2*currentInd+1]
                        cutCellPointsInd[2*currentInd+1] = tempuint
                                                    
        # Cut-cell normal vectors.
        self.cutCellNormalVectors = numpy.zeros((np,2), dtype=numpy.double)
        cutCellNormalVectors = &self.cutCellNormalVectors[0,0]
        for jj in range(1,ny-1):
            for ii in range(1,nx-1):
                currentInd = ii + jj*nx
                if insideFaces[currentInd] != 0 and insideFaces[currentInd] != 8:
                    cutCellNormalVectors[2*currentInd] = boundaryPoints[2*cutCellPointsInd[2*currentInd]+1] - \
                                                         boundaryPoints[2*cutCellPointsInd[2*currentInd+1]+1]  # TODO CHECK
                    cutCellNormalVectors[2*currentInd+1] = boundaryPoints[2*cutCellPointsInd[2*currentInd+1]] - \
                                                           boundaryPoints[2*cutCellPointsInd[2*currentInd]] 
                    tempdouble = 1./sqrt(cutCellNormalVectors[2*currentInd]**2 + cutCellNormalVectors[2*currentInd+1]**2)
                    cutCellNormalVectors[2*currentInd] *= tempdouble
                    cutCellNormalVectors[2*currentInd+1] *= tempdouble

        # Continue with calculation of faces.
        for ii in range(nx):
            da[ii] = 0.
            da[ii+(ny-2)*nx] = 0.
            da[ii+(ny-1)*nx] = 0.
            dai[ii] = 0.
            dai[ii+(ny-2)*nx] = 0.
            dai[ii+(ny-1)*nx] = 0.
        for jj in range(1,ny-2):
            da[jj*nx] = 0.
            da[(nx-2)+jj*nx] = 0.
            da[(nx-1)+jj*nx] = 0.
            dai[jj*nx] = 0.
            dai[(nx-2)+jj*nx] = 0.
            dai[(nx-1)+jj*nx] = 0.
        
        for jj in range(1,ny-2):
            for ii in range(1,nx-2):
                if insideFaces[ii+jj*nx]==8:
                    da[ii+jj*nx] = dxdy
                    dai[ii+jj*nx] = dxdyi
                elif insideFaces[ii+jj*nx]==0:
                    da[ii+jj*nx] = 0.
                    dai[ii+jj*nx] = 0.
                elif insideFaces[ii+jj*nx]==2:
                    if insidePoints[ii+jj*nx]==1:
                        da[ii+jj*nx] = 0.5*ds[ii+jj*nx]*ds[ii+jj*nx+np]
                    elif insidePoints[ii+1+jj*nx]==1:
                        da[ii+jj*nx] = 0.5*ds[ii+jj*nx]*ds[ii+1+jj*nx+np]
                    elif insidePoints[ii+(jj+1)*nx]==1:
                        da[ii+jj*nx] = 0.5*ds[ii+(jj+1)*nx]*ds[ii+jj*nx+np]
                    else:
                        da[ii+jj*nx] = 0.5*ds[ii+(jj+1)*nx]*ds[ii+1+jj*nx+np]
                    dai[ii+jj*nx] = 1./da[ii+jj*nx]
                elif insideFaces[ii+jj*nx]==4:
                    da[ii+jj*nx] = 0.5*(ds[ii+jj*nx] + ds[ii+(jj+1)*nx])*(ds[ii+jj*nx+np] + ds[ii+1+jj*nx+np])
                    dai[ii+jj*nx] = 1./da[ii+jj*nx]
                elif insideFaces[ii+jj*nx]==6:
                    if insidePoints[ii+jj*nx]==0:
                        da[ii+jj*nx] = dxdy-0.5*(dx-ds[ii+jj*nx])*(dy-ds[ii+jj*nx+np])      # Only for equidistant.
                    elif insidePoints[ii+1+jj*nx]==0:
                        da[ii+jj*nx] = dxdy-0.5*(dx-ds[ii+jj*nx])*(dy-ds[ii+1+jj*nx+np])
                    elif insidePoints[ii+(jj+1)*nx]==0:
                        da[ii+jj*nx] = dxdy-0.5*(dx-ds[ii+(jj+1)*nx])*(dy-ds[ii+jj*nx+np])
                    else:
                        da[ii+jj*nx] = dxdy-0.5*(dx-ds[ii+(jj+1)*nx])*(dy-ds[ii+1+jj*nx+np])
                    dai[ii+jj*nx] = 1./da[ii+jj*nx]
                    
        # Dual grid quantities are not affected by cut-cell stuff. Same as for rectangular. 
        # They are not needed for any essential
        # calculation in the poisson problem, just for visualization of the charge density.
        for ii in range(nx):
            dst[ii] = 0.
            dst[ii+(ny-1)*nx] = 0.
            dst[ii+np] = 0.
            dst[ii+(ny-1)*nx+np] = 0.
            dsti[ii] = 0.
            dsti[ii+(ny-1)*nx] = 0.
            dsti[ii+np] = 0.
            dsti[ii+(ny-1)*nx+np] = 0.
        for jj in range(1,ny-1):
            dst[jj*nx] = 0.
            dst[(nx-1)+jj*nx] = 0.
            dst[jj*nx+np] = 0.
            dst[(nx-1)+jj*nx+np] = 0.
            dsti[jj*nx] = 0.
            dsti[(nx-1)+jj*nx] = 0.
            dsti[jj*nx+np] = 0.
            dsti[(nx-1)+jj*nx+np] = 0.
        for ii in range(2,nx-2):
            for jj in range(1,ny-1):
                dst[ii+jj*nx] = dx                          # Here assuming equidistant meshes (in each direction).
                dsti[ii+jj*nx] = dxi
        for ii in range(1,nx-1):
            for jj in range(2,ny-2):
                dst[ii+jj*nx+np] = dy
                dsti[ii+jj*nx+np] = dyi
        for jj in range(1,ny-1):
            dst[1+jj*nx] = dxHalf
            dst[(nx-2)+jj*nx] = dxHalf
            dsti[1+jj*nx] = dxHalfi
            dsti[(nx-2)+jj*nx] = dxHalfi
        for ii in range(1,nx-1):
            dst[ii+nx+np] = dyHalf
            dst[ii+(ny-2)*nx+np] = dyHalf
            dsti[ii+nx+np] = dyHalfi
            dsti[ii+(ny-2)*nx+np] = dyHalfi
        for ii in range(nx):
            for jj in range(ny):
                # TODO Area calculation could be improved to make plots of cut cells nicer
                dat[ii+jj*nx] = dst[ii+jj*nx]*dst[ii+jj*nx+np]
                dati[ii+jj*nx] = dsti[ii+jj*nx]*dsti[ii+jj*nx+np] 

        # Standard interpolation stuff. Interpolation from edges to _inside_ grid points. Same as rectangular.         
        kk = 0
        for ii in range(2,nx-2):
            for jj in range(2,ny-2):
                currentInd = ii + jj*nx
                if insidePoints[currentInd]==1:
                    rows[kk] = currentInd
                    columns[kk] = currentInd
                    values[kk] = ds[currentInd]/(ds[currentInd-1]+ds[currentInd])   # Should work for non-equidistant grid.  
                    kk += 1
                    rows[kk] = currentInd
                    columns[kk] = currentInd-1
                    values[kk] = ds[currentInd-1]/(ds[currentInd-1]+ds[currentInd])
                    kk += 1

                    currentInd += np 
                    rows[kk] = currentInd
                    columns[kk] = currentInd
                    values[kk] = ds[currentInd]/(ds[currentInd-nx]+ds[currentInd])
                    kk += 1         
                    rows[kk] = currentInd
                    columns[kk] = currentInd-nx
                    values[kk] = ds[currentInd-nx]/(ds[currentInd-nx]+ds[currentInd])
                    kk += 1
        
        self.edgeToNode = spsp.csc_matrix(spsp.coo_matrix((values[:kk],(rows[:kk],columns[:kk])),shape=(2*np,2*np)))

        
#        # Boundary interpolation for correct cut-cell fields at boundary.
#        # This is the only really messy and hard part in this whole class. One has to interpolate
#        # at the boundary by using the continuity conditions, which leads to lots of angles and so on...
#        # I am very sure that this is in principle is much better than many other people do it (they just
#        # ignore the cut cells and work on the equidistant mesh). For openEcloud we need good field
#        # interpolation at the wall, so no way around it.
#        # There is some stuff included to extrapolate accurately to the equidistant grid, but the last
#        # step will be done in the next block.
#        for jj in range(1, ny-1):
#            for ii in range(1, nx-1):
#                currentInd = ii+jj*nx
#                if insideEdges[currentInd]==0 or insideEdges[currentInd]==2:
#                    cosAlpha[currentInd] = 0.
#                    sinAlpha[currentInd] = 0.              
#                else:
#                    # TODO check angle calculation again
#                    # Take average of the two cells adjescent to the edge as the edge-boundary impact angle
#                    temp = 0.5 * ( atan2(cutCellNormalVectors[2*currentInd+1],cutCellNormalVectors[2*currentInd]) + 
#                                   atan2(cutCellNormalVectors[2*(currentInd-nx)+1],cutCellNormalVectors[2*(currentInd-nx)]) )
#                    cosAlpha[currentInd] = cos(temp)
#                    sinAlpha[currentInd] = sin(temp)
##                    print ii, jj, temp/pi*180, cosAlpha[currentInd], sinAlpha[currentInd]
#                if insideEdges[currentInd+np]==0 or insideEdges[currentInd+np]==2:
#                    cosBeta[currentInd] = 0.
#                    sinBeta[currentInd] = 0.
#                else:
#                    # TODO check angle calculation again
#                    # Take average of the two cells adjescent to the edge as the edge-boundary impact angle
#                    temp = 0.5 * ( atan2(-cutCellNormalVectors[2*currentInd],cutCellNormalVectors[2*currentInd+1]) + 
#                                   atan2(-cutCellNormalVectors[2*(currentInd-1)],cutCellNormalVectors[2*(currentInd-1)+1]) )
#                    cosBeta[currentInd] = cos(temp)
#                    sinBeta[currentInd] = sin(temp)
##                    print ii, jj, temp/pi*180, cosBeta[currentInd], sinBeta[currentInd]
#        for ii in range(nx):
#            for jj in range(ny):
#                currentInd = ii+jj*nx
#                connectedEdgesInv[currentInd] = 0.
#        for jj in range(1,ny-1):
#            for ii in range(1,nx-1):
#                currentInd = ii+jj*nx
#                if insidePoints[currentInd] == 0:
#                    connectedEdgesInv[currentInd] = insideEdges[currentInd] + insideEdges[currentInd-1] + \
#                                                    insideEdges[currentInd+np] + insideEdges[currentInd-nx+np]
#                if connectedEdgesInv[currentInd] > 1.:
#                    connectedEdgesInv[currentInd] = 1./connectedEdgesInv[ii+jj*nx]     
#        kk = 0
#        for jj in range(1,ny-1):
#            for ii in range(1,nx-1):
#                currentInd = ii+jj*nx
#                # Fields at inside points are unchanged.
#                if insidePoints[currentInd] == 1:
#                    rows[kk] = currentInd
#                    columns[kk] = currentInd
#                    values[kk] = 1.              
#                    kk += 1 
#                    rows[kk] = currentInd+np
#                    columns[kk] = currentInd+np
#                    values[kk] = 1.              
#                    kk += 1    
#                # Boundary edges.
#                else:
#                    if insideEdges[currentInd] == 1:
#                        # X to x in negative x-direction.
#                        rows[kk] = currentInd
#                        columns[kk] = currentInd+1
#                        values[kk] = connectedEdgesInv[currentInd]*(1. - sinAlpha[currentInd]**2*dx/ds[currentInd])
#                        kk += 1
#                        # Y to x in negative x-direction.
#                        rows[kk] = currentInd
#                        columns[kk] = currentInd+np+1
#                        values[kk] = -connectedEdgesInv[currentInd]*(cosAlpha[currentInd]*sinAlpha[currentInd]*dx/ds[currentInd])
#                        kk += 1
#                        # Y to y in negative x-direction.
#                        rows[kk] = currentInd+np
#                        columns[kk] = currentInd+np+1
#                        values[kk] = connectedEdgesInv[currentInd]*(1. - cosAlpha[currentInd]**2*dx/ds[currentInd])            
#                        kk += 1
#                        # X to y in negative x-direction
#                        rows[kk] = currentInd+np
#                        columns[kk] = currentInd+1
#                        values[kk] = -connectedEdgesInv[currentInd]*(cosAlpha[currentInd]*sinAlpha[currentInd]*dx/ds[currentInd])
#                        kk += 1
#                    elif insideEdges[currentInd-1] == 1:
#                        # X to x in positive x-direction.
#                        rows[kk] = currentInd
#                        columns[kk] = currentInd-1
#                        values[kk] = connectedEdgesInv[currentInd]*(1. - sinAlpha[currentInd]**2*dx/ds[currentInd-1])                
#                        kk += 1   
#                        # Y to x in positive x-direction.
#                        rows[kk] = currentInd
#                        columns[kk] = currentInd+np-1
#                        values[kk] = -connectedEdgesInv[currentInd]*(cosAlpha[currentInd-1]*sinAlpha[currentInd-1]*dx/ds[currentInd-1])                                    
#                        kk += 1  
#                        # Y to y in positive x-direction.
#                        rows[kk] = currentInd+np
#                        columns[kk] = currentInd+np-1
#                        values[kk] = connectedEdgesInv[currentInd]*(1. - cosAlpha[currentInd-1]**2*dx/ds[currentInd-1])            
#                        kk += 1
#                        # X to y in positive x-direction
#                        rows[kk] = currentInd+np
#                        columns[kk] = currentInd-1
#                        values[kk] = -connectedEdgesInv[currentInd]*(cosAlpha[currentInd-1]*sinAlpha[currentInd-1]*dx/ds[currentInd-1])
#                        kk += 1
#                    if insideEdges[currentInd+np] == 1:
#                        # X to x in negative y-direction.
#                        rows[kk] = currentInd
#                        columns[kk] = currentInd+nx
#                        values[kk] = connectedEdgesInv[currentInd]*(1. - cosBeta[currentInd]**2*dx/ds[currentInd+np])
#                        kk += 1
#                        # Y to x in negative y-direction.
#                        rows[kk] = currentInd
#                        columns[kk] = currentInd+np+nx
#                        values[kk] = -connectedEdgesInv[currentInd]*(cosBeta[currentInd]*sinBeta[currentInd]*dx/ds[currentInd+np])
#                        kk += 1
#                        # Y to y in negative y-direction.
#                        rows[kk] = currentInd+np
#                        columns[kk] = currentInd+np+nx
#                        values[kk] = connectedEdgesInv[currentInd]*(1. - sinBeta[currentInd]**2*dx/ds[currentInd+np])            
#                        kk += 1                            
#                        # X to y in negative y-direction
#                        rows[kk] = currentInd+np
#                        columns[kk] = currentInd+nx
#                        values[kk] = -connectedEdgesInv[currentInd]*(cosBeta[currentInd]*sinBeta[currentInd]*dx/ds[currentInd+np])
#                        kk += 1
#                    elif insideEdges[currentInd+np-nx] == 1:
#                        # X to x in positive y-direction.
#                        rows[kk] = currentInd
#                        columns[kk] = currentInd-nx
#                        values[kk] = connectedEdgesInv[currentInd]*(1. - cosBeta[currentInd-nx]**2*dx/ds[currentInd+np-nx])                
#                        kk += 1   
#                        # Y to x in positive y-direction.
#                        rows[kk] = currentInd
#                        columns[kk] = currentInd+np-nx
#                        values[kk] = -connectedEdgesInv[currentInd]*(cosBeta[currentInd-nx]*sinBeta[currentInd-nx]*dx/ds[currentInd+np-nx])                                    
#                        kk += 1  
#                        # Y to y in positive y-direction.
#                        rows[kk] = currentInd+np
#                        columns[kk] = currentInd+np-nx
#                        values[kk] = connectedEdgesInv[currentInd]*(1. - sinBeta[currentInd-nx]**2*dx/ds[currentInd+np-nx])            
#                        kk += 1
#                        # X to y in positive y-direction
#                        rows[kk] = currentInd+np
#                        columns[kk] = currentInd-nx
#                        values[kk] = -connectedEdgesInv[currentInd]*(cosBeta[currentInd-nx]*sinBeta[currentInd-nx]*dx/ds[currentInd+np-nx])
#                        kk += 1
#        
#        self.edgeToNode = spsp.csc_matrix(spsp.coo_matrix((values[:kk],(rows[:kk],columns[:kk])),shape=(2*np,2*np))
#                                          ).dot(self.edgeToNode)    

# Second order extrapolation
#                    if insideEdges[currentInd] == 1:
#                        if currentInd < np-3 and insidePoints[currentInd+2] == 1:
#                            rows[kk] = currentInd
#                            columns[kk] = currentInd+2
#                            values[kk] = -connectedEdgesInv[currentInd]
#                            kk += 1
#                            rows[kk] = currentInd
#                            columns[kk] = currentInd+1
#                            values[kk] = 2*connectedEdgesInv[currentInd]
#                            kk += 1
#                            rows[kk] = currentInd+np
#                            columns[kk] = currentInd+np+2
#                            values[kk] = -connectedEdgesInv[currentInd]
#                            kk += 1
#                            rows[kk] = currentInd+np
#                            columns[kk] = currentInd+np+1
#                            values[kk] = 2*connectedEdgesInv[currentInd]
#                            kk += 1
#                        else:
#                            rows[kk] = currentInd
#                            columns[kk] = currentInd+1
#                            values[kk] = connectedEdgesInv[currentInd]
#                            kk += 1
#                            rows[kk] = currentInd+np
#                            columns[kk] = currentInd+np+1
#                            values[kk] = connectedEdgesInv[currentInd]
#                            kk += 1                    
#                    elif insideEdges[currentInd-1] == 1:
#                        if currentInd > 2 and insidePoints[currentInd-2] == 1:
#                            rows[kk] = currentInd-1
#                            columns[kk] = currentInd+1
#                            values[kk] = -connectedEdgesInv[currentInd-1]
#                            kk += 1
#                            rows[kk] = currentInd-1
#                            columns[kk] = currentInd
#                            values[kk] = 2*connectedEdgesInv[currentInd-1]
#                            kk += 1
#                            rows[kk] = currentInd+np-1
#                            columns[kk] = currentInd+np+1
#                            values[kk] = -connectedEdgesInv[currentInd-1]
#                            kk += 1
#                            rows[kk] = currentInd+np-1
#                            columns[kk] = currentInd+np
#                            values[kk] = 2*connectedEdgesInv[currentInd-1]
#                            kk += 1
#                        else:
#                            rows[kk] = currentInd-1
#                            columns[kk] = currentInd
#                            values[kk] = connectedEdgesInv[currentInd-1]
#                            kk += 1
#                            rows[kk] = currentInd+np-1
#                            columns[kk] = currentInd+np
#                            values[kk] = connectedEdgesInv[currentInd-1]
#                            kk += 1  

        # Simplified boundary extrapolation for correct cut-cell fields at boundary.
        # This is the only inaccurate step of the whole field calculation, but trying to do
        # anything more sophisticated actually worsens the field quality.
        # The only reason to improve this is to move to different meshes and/or finite elements.
        for ii in range(nx):
            for jj in range(ny):
                currentInd = ii+jj*nx
                connectedEdgesInv[currentInd] = 0.
        for jj in range(1,ny-1):
            for ii in range(1,nx-1):
                currentInd = ii+jj*nx
                if insidePoints[currentInd] == 0:
                    connectedEdgesInv[currentInd] = insideEdges[currentInd] + insideEdges[currentInd-1] + \
                                                    insideEdges[currentInd+np] + insideEdges[currentInd-nx+np]
                if connectedEdgesInv[currentInd] > 1.:
                    connectedEdgesInv[currentInd] = 1./connectedEdgesInv[ii+jj*nx]     
        kk = 0
        for jj in range(1,ny-1):
            for ii in range(1,nx-1):
                currentInd = ii+jj*nx
                # Fields at inside points are unchanged.
                if insidePoints[currentInd] == 1:
                    rows[kk] = currentInd
                    columns[kk] = currentInd
                    values[kk] = 1.              
                    kk += 1 
                    rows[kk] = currentInd+np
                    columns[kk] = currentInd+np
                    values[kk] = 1.              
                    kk += 1    
                # Boundary edges.
                else:
                    if insideEdges[currentInd] == 1:
                        rows[kk] = currentInd
                        columns[kk] = currentInd+1
                        values[kk] = connectedEdgesInv[currentInd]
                        kk += 1
                        rows[kk] = currentInd+np
                        columns[kk] = currentInd+np+1
                        values[kk] = connectedEdgesInv[currentInd]
                        kk += 1                    
                    elif insideEdges[currentInd-1] == 1:
                        rows[kk] = currentInd
                        columns[kk] = currentInd-1
                        values[kk] = connectedEdgesInv[currentInd]
                        kk += 1
                        rows[kk] = currentInd+np
                        columns[kk] = currentInd+np-1
                        values[kk] = connectedEdgesInv[currentInd]
                        kk += 1  
                    if insideEdges[currentInd+np] == 1:
                        rows[kk] = currentInd
                        columns[kk] = currentInd+nx
                        values[kk] = connectedEdgesInv[currentInd]
                        kk += 1
                        rows[kk] = currentInd+np
                        columns[kk] = currentInd+np+nx
                        values[kk] = connectedEdgesInv[currentInd]
                        kk += 1   
                    if insideEdges[currentInd+np-nx] == 1:
                        rows[kk] = currentInd
                        columns[kk] = currentInd-nx
                        values[kk] = connectedEdgesInv[currentInd]
                        kk += 1
                        rows[kk] = currentInd+np
                        columns[kk] = currentInd+np-nx
                        values[kk] = connectedEdgesInv[currentInd]
                        kk += 1 
        
        self.edgeToNode = spsp.csc_matrix(spsp.coo_matrix((values[:kk],(rows[:kk],columns[:kk])),shape=(2*np,2*np))
                                          ).dot(self.edgeToNode)    

           
        # Last step of the extrapolation to equidistant grid for cells with three points. 
        # Makes interpolation later easier/faster.
        kk = 0
        for jj in range(1,ny-1):
            for ii in range(1,nx-1):
                currentInd = ii+jj*nx
                # Fields at inside points, completely outside points and non-triangular boundary points are unchanged
                rows[kk] = currentInd
                columns[kk] = currentInd
                values[kk] = 1.              
                kk += 1 
                rows[kk] = currentInd+np
                columns[kk] = currentInd+np
                values[kk] = 1.              
                kk += 1 
                if insidePoints[currentInd]==0:
                    # Fields at triangular boundary cells need to be extrapolated
                    # +X -Y
                    if insidePoints[currentInd-1]==0 and insidePoints[currentInd+nx]==0 and insidePoints[currentInd-1+nx]==1:
                        rows[kk] = currentInd
                        columns[kk] = currentInd-1+nx
                        values[kk] = -1.
                        kk += 1
                        rows[kk] = currentInd
                        columns[kk] = currentInd-1
                        values[kk] = 1.
                        kk += 1
                        rows[kk] = currentInd
                        columns[kk] = currentInd+nx
                        values[kk] = 1.
                        kk += 1
                        rows[kk] = currentInd+np
                        columns[kk] = currentInd+np-1+nx
                        values[kk] = -1.
                        kk += 1
                        rows[kk] = currentInd+np
                        columns[kk] = currentInd-1+np
                        values[kk] = 1.
                        kk += 1
                        rows[kk] = currentInd+np
                        columns[kk] = currentInd+nx+np
                        values[kk] = 1.
                        kk += 1
                    # -X -Y
                    elif insidePoints[currentInd+1]==0 and insidePoints[currentInd+nx]==0 and insidePoints[currentInd+1+nx]==1:
                        rows[kk] = currentInd
                        columns[kk] = currentInd+1+nx
                        values[kk] = -1.
                        kk += 1                        
                        rows[kk] = currentInd
                        columns[kk] = currentInd+1
                        values[kk] = 1.
                        kk += 1
                        rows[kk] = currentInd
                        columns[kk] = currentInd+nx
                        values[kk] = 1.
                        kk += 1
                        rows[kk] = currentInd+np
                        columns[kk] = currentInd+np+1+nx
                        values[kk] = -1.
                        kk += 1
                        rows[kk] = currentInd+np
                        columns[kk] = currentInd+1+np
                        values[kk] = 1.
                        kk += 1
                        rows[kk] = currentInd+np
                        columns[kk] = currentInd+nx+np
                        values[kk] = 1.
                        kk += 1
                    # -X +Y
                    elif insidePoints[currentInd+1]==0 and insidePoints[currentInd-nx]==0 and insidePoints[currentInd+1-nx]==1:
                        rows[kk] = currentInd
                        columns[kk] = currentInd+1-nx
                        values[kk] = -1.
                        kk += 1
                        rows[kk] = currentInd
                        columns[kk] = currentInd+1
                        values[kk] = 1.
                        kk += 1
                        rows[kk] = currentInd
                        columns[kk] = currentInd-nx
                        values[kk] = 1.
                        kk += 1
                        rows[kk] = currentInd+np
                        columns[kk] = currentInd+np+1-nx
                        values[kk] = -1.
                        kk += 1
                        rows[kk] = currentInd+np
                        columns[kk] = currentInd+1+np
                        values[kk] = 1.
                        kk += 1
                        rows[kk] = currentInd+np
                        columns[kk] = currentInd-nx+np
                        values[kk] = 1.
                        kk += 1
                    # +X +Y
                    elif insidePoints[currentInd-1]==0 and insidePoints[currentInd-nx]==0 and insidePoints[currentInd-1-nx]==1:
                        rows[kk] = currentInd
                        columns[kk] = currentInd-1-nx
                        values[kk] = -1.
                        kk += 1
                        rows[kk] = currentInd
                        columns[kk] = currentInd-1
                        values[kk] = 1.
                        kk += 1
                        rows[kk] = currentInd
                        columns[kk] = currentInd-nx
                        values[kk] = 1.
                        kk += 1
                        rows[kk] = currentInd+np
                        columns[kk] = currentInd+np-1-nx
                        values[kk] = -1.
                        kk += 1
                        rows[kk] = currentInd+np
                        columns[kk] = currentInd-1+np
                        values[kk] = 1.
                        kk += 1
                        rows[kk] = currentInd+np
                        columns[kk] = currentInd-nx+np
                        values[kk] = 1.
                        kk += 1
                    

        self.edgeToNode = spsp.csc_matrix(spsp.coo_matrix((values[:kk],(rows[:kk],columns[:kk])),shape=(2*np,2*np))
                                          ).dot(self.edgeToNode)


    
    '''
    Getter functions from here on.
    
    '''
    cpdef double[:] getXMesh(Grid self):
        # Uniform grid for cut cell as well.
        return self.xMesh
    
    cpdef double[:] getYMesh(Grid self):
        # Uniform grid for cut cell as well.
        return self.yMesh   

    cpdef double getLx(Grid self):
        return self.lx
    
    cpdef double getLy(Grid self):
        return self.ly
    
    cpdef double getLxExt(Grid self):
        return self.lxExt
    
    cpdef double getLyExt(Grid self):
        return self.lyExt
    
    cpdef double getDx(Grid self):
        return self.dx
    
    cpdef double getDy(Grid self):
        return self.dy    
    
    cpdef unsigned int getNxExt(Grid self):
        return self.nxExt
    
    cpdef unsigned int getNyExt(Grid self):
        return self.nyExt 
    
    cpdef unsigned int getNpExt(Grid self):
        return self.npExt 
        
    cpdef unsigned int getNx(Grid self):
        return self.nx 
    
    cpdef unsigned int getNy(Grid self):
        return self.ny 
    
    cpdef unsigned int getNp(Grid self):
        return self.np 
        
    cpdef double[:,:] getBoundaryPoints(Grid self):
        return self.boundaryPoints
    
    cpdef unsigned int[:] getBoundaryPointsInd(Grid self):
        return self.boundaryPointsInd
            
    cpdef int[:,:] getCutCellPointsInd(Grid self):
        return self.cutCellPointsInd
    
    cpdef double[:,:] getCutCellCenter(Grid self):
        return self.cutCellCenter
        
    cpdef double[:,:] getCutCellNormalVectors(Grid self):
        return self.cutCellNormalVectors
    
    cpdef unsigned int[:] getInCell(Grid self):
        return self.inCell
        
    def getInsidePoints(self):
        return self.insidePoints
    
    def getInsideEdges(self):
        return self.insideEdges
    
    cpdef unsigned short[:] getInsideFaces(Grid self):
        return self.insideFaces
    
    cpdef double[:] getDs(Grid self):
        return self.ds
    
    def getDsi(self):
        return self.dsi
    
    def getDa(self):
        return self.da
    
    def getDai(self):
        return self.dai

    cpdef double[:] getDst(Grid self):
        return self.dst
     
    def getDat(self):
        return self.dat
       
    def getDati(self):
        return self.dati
    
    def getEdgeToNode(self):
        return self.edgeToNode    
        
    cpdef object getBoundFunc(Grid self):
        return self.boundFunc


cpdef unsigned short _boundFuncRectangular(double x, double y, double lxHalf, double lyHalf):      
    if x<lxHalf and x>-lxHalf and y<lyHalf and y>-lyHalf:
        return 1
    else:
        return 0    
        
cpdef unsigned short _boundFuncElliptical(double x, double y, double aSqInv, double bSqInv):      
    if 1 > x**2*aSqInv + y**2*bSqInv:
        return 1
    else:
        return 0    




