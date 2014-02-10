import numpy
cimport numpy
cimport cython
import scipy as sp              
import scipy.sparse as spsp

# Import some C methods.
cdef extern from "math.h":
    double sin(double x) nogil     
    double cos(double x) nogil
    double sqrt(double x) nogil
    
'''
Class to calculate and store grid quantities.

Requires scipy (sp), scipy.sparse (spsp)
'''
cdef class Grid:

    # cdef:
        # unsigned int nx, ny, np
        # double lx, ly, dx, dy, cutCellAcc, cutCellMinEdgeLength, scanAroundPoint
        # numpy.ndarray xMesh, yMesh, cosAlpha, sinAlpha, cosBeta, sinBeta 
        # numpy.ndarray insidePoints, insidePointsInd, insideEdges, insideEdgesInd
        # numpy.ndarray insideCells, insideCellsInd
        # numpy.ndarray ds, dsi, da, dai, dst, dsti, dat, dati
        # object gridBoundaryObj, edgeToNode
    '''
    Constructor.
    
    Input:
           - nxy:                      Vector of the number of grid points in x and y direction [nx, ny].
           - lxy:                      Vector of grid size in x and y direction [lx, ly].
           - boundType:                Type of boundary. One of 'rectangular', 'elliptical' or 'arbitrary'.
           - cutCellAcc:               Accuracy to which cut cell edge length are calculated (optional).
           - cutCellMinEdgeLength:     Minimum length (relative to grid step length) of the cut cells 
                                       before they are set to zero (optional).
           - scanAroundPoint:          Relative area around each grid point which must be inside the grid 
                                       boundary for the point to be set as inside point. (optional).
                                       
    '''   
    def __init__(self, nxy, lxy, boundType, boundFunc = None, cutCellAcc = 1e-12, cutCellMinEdgeLength = 1e-6, scanAroundPoint = 1e-6):

        # Calculate and store basic parameters of the uniform grid. For cut-cell some accuracy parameters.
        self.nx = <unsigned int> nxy[0]
        self.ny = <unsigned int> nxy[1]
        self.lx = <double> lxy[0]
        self.ly = <double> lxy[1]
        self.dx = self.lx/(self.nx-1.)
        self.dy = self.ly/(self.ny-1.)
        self.np = self.nx*self.ny
        self.cutCellAcc = cutCellAcc
        self.cutCellMinEdgeLength = cutCellMinEdgeLength
        self.scanAroundPoint = scanAroundPoint    
        self.xMesh = sp.linspace(-self.lx/2., self.lx/2, self.nx)
        self.yMesh = sp.linspace(-self.ly/2., self.ly/2, self.ny) 
        
        # Calculate other grid parameters depending on boundary.
        if boundType=='rectangular':
            self.setupRectangular()  
        elif boundType=='elliptical':
            self.setupElliptical()
        elif boundType=='arbitrary':
            self.setupArbitrary(boundFunc)
        else:
            raise NotImplementedError("Not yet implemented.")
            

    '''
    Helper function for the setup of rectangular geometry. Calculates grid quantities.
    
    '''    
    def setupRectangular(self):
        
        # Points inside domain, as 1/0 array and array of indexes of the inside points.         
        self.insidePoints = sp.concatenate( (sp.zeros(self.nx), sp.tile(sp.concatenate(([0],sp.ones(self.nx-2),[0])),self.ny-2), sp.zeros(self.nx)) )
        self.insidePointsInd = sp.nonzero(self.insidePoints)[0]
        
        # Edges inside domain, as 1/0 array and array of indexes of the inside edges.    
        self.insideEdges = sp.concatenate( (sp.zeros(self.nx), sp.tile(sp.concatenate((sp.ones(self.nx-1),[0])),self.ny-2), sp.zeros(self.nx), 
                                            sp.tile(sp.concatenate(([0],sp.ones(self.nx-2),[0])),self.ny-1), sp.zeros(self.nx) ) )
        self.insideEdgesInd = sp.nonzero(self.insideEdges)[0]
        
        # Vectors of edge lengths and inverse (2D: x,y), vectors of faces and inverse (2D: z).
        # Primary grid quantities.                                 
        self.ds = sp.concatenate( (self.insideEdges[:self.np]*self.dx, self.insideEdges[self.np:2*self.np]*self.dy) )
        self.dsi = sp.zeros(2*self.np)
        self.dsi[self.insideEdgesInd] = 1/(self.ds[self.insideEdgesInd])
        self.da = self.dx*self.dy*self.insidePoints
        self.dai = sp.zeros(self.np)
        self.dai[self.insidePointsInd] = 1/self.da[self.insidePointsInd]
        # Dual grid quantities.
        self.dst = sp.concatenate( (sp.tile(sp.repeat([.5,1.,.5],[1,self.nx-2,1]),self.ny)*self.dx,
                                    sp.repeat([.5,1.,.5,],[self.nx,(self.ny-2)*self.nx,self.nx])*self.dy) 
                                  )
        self.dsti = 1/self.dst
        self.dat = self.dst[:self.np]*self.dst[self.np:2*self.np]
        self.dati = self.dsti[:self.np]*self.dsti[self.np:2*self.np]
        
        # Calculation of electric field on edges from finite integration quantity and interpolation to grid points.
        # All is combined here to a sparse matrix to reduce effort to just a matrix multiplication at each time step.      
        dsnpnm1i = sp.zeros(self.np)
        temp = self.ds[:self.np] + sp.concatenate(([0], self.ds[:self.np-1]))
        dsnpnm1i[sp.nonzero(temp)] = 1/temp[sp.nonzero(temp)]
        dsnpnmnxi = sp.zeros(self.np)
        temp = self.ds[self.np:2*self.np] + sp.concatenate((sp.zeros(self.nx), self.ds[self.np:2*self.np-self.nx]))
        dsnpnmnxi[sp.nonzero(temp)] = 1/temp[sp.nonzero(temp)]      
        self.edgeToNode = spsp.block_diag((
                                           spsp.dia_matrix(([sp.tile(sp.concatenate((sp.ones(self.nx-1),[0])),self.ny)*self.ds[0:self.np]*dsnpnm1i, 
                                                             sp.tile(sp.concatenate((sp.ones(self.nx-1),[0])),self.ny)*self.ds[0:self.np]*sp.concatenate((dsnpnm1i[1:],[0]))], 
                                                            [0, -1]), shape=(self.np, self.np)),
                                           spsp.dia_matrix(([sp.concatenate((sp.ones(self.nx),sp.ones(self.nx*(self.ny-2)),sp.zeros(self.nx)))*self.ds[self.np:2*self.np]*dsnpnmnxi,
                                                             sp.concatenate((sp.ones(self.nx*(self.ny-2)),sp.ones(self.nx),sp.zeros(self.nx)))*self.ds[self.np:2*self.np]*sp.concatenate((dsnpnmnxi[self.nx:],sp.zeros(self.nx)))], 
                                                            [0, -self.nx]), shape=(self.np, self.np))
                                           ))


    '''
    Helper function for the setup of elliptical geometry. Calculates grid quantities.
    
    '''  
    def setupElliptical(self):
         
        # Grid boundary object provides some helper functions for e.g. normal vector calculation.
        self.gridBoundaryObj = GridBoundaryElliptical(self)
               
        # Need cosine of angle to normal vector to interpolate field correctly at boundary.
        (self.cosAlpha, self.sinAlpha) = self.gridBoundaryObj.CosSinNormalAngleX(self.yMesh[1:-1])
        (self.cosBeta, self.sinBeta) = self.gridBoundaryObj.CosSinNormalAngleY(self.xMesh[1:-1])
         
        # Rest is identical for every cut cell domain.
        self.computeCutCellGridGeom()     


    def setupArbitrary(self, boundFunc):
         
        # Grid boundary object provides some helper functions for e.g. normal vector calculation.
        self.gridBoundaryObj = GridBoundaryArbitrary(self, boundFunc)
               
        # Need cosine of angle to normal vector to interpolate field correctly at boundary.
        (self.cosAlpha, self.sinAlpha) = self.gridBoundaryObj.CosSinNormalAngleX(self.yMesh[1:-1])
        (self.cosBeta, self.sinBeta) = self.gridBoundaryObj.CosSinNormalAngleY(self.xMesh[1:-1])
         
        # Rest is identical for every cut cell domain.
        self.computeCutCellGridGeom()  
 
    
    '''
    Helper function for the setup of cut cell geometries. Calculates grid quantities.
    
    '''      
    def computeCutCellGridGeom(self):
    
        # Check if grid point lies inside domain with some given tolerance.
        dxMin = self.scanAroundPoint*self.dx
        dyMin = self.scanAroundPoint*self.dy
        xPoints = sp.tile(self.xMesh, self.ny)
        yPoints = sp.repeat(self.yMesh, self.nx)
        upXUpY = self.gridBoundaryObj.isInside(xPoints+dxMin,yPoints+dyMin)
        upXLowY = self.gridBoundaryObj.isInside(xPoints+dxMin,yPoints-dyMin)
        lowXUpY = self.gridBoundaryObj.isInside(xPoints-dxMin,yPoints+dyMin)
        lowXLowY = self.gridBoundaryObj.isInside(xPoints-dxMin,yPoints-dyMin)
        self.insidePoints = upXUpY*upXLowY*lowXUpY*lowXLowY

        # Number of iterations needed to calculate edge length for a given accuracy.
        boundAccIter = sp.int0(sp.log2(1/self.cutCellAcc))+2
        
        # Classify edges. 0 = outside, 1 = possible cut cell edges and 2 = normal inside edges.       
        edgesX = sp.int0(self.insidePoints)+sp.int0(sp.roll(self.insidePoints,-1))        
        dsx = sp.clip(edgesX,0,1)*self.dx      
        cutCellEdgesX = sp.logical_and( edgesX == 1, sp.logical_not(sp.logical_and(self.gridBoundaryObj.isInside(xPoints,yPoints),self.gridBoundaryObj.isInside(xPoints + self.dx,yPoints))) )
        
        # Bisect repeatedly the possible cut cell edges to get length of each. 
        x1 = xPoints[cutCellEdgesX]
        x2 = x1 + self.dx
        y = yPoints[cutCellEdgesX]        
        for ii in range(boundAccIter):  # @UnusedVariable
            x3 = x1+(x2-x1)/2;
            test01 = self.gridBoundaryObj.isInside(x1,y)
            test02 = self.gridBoundaryObj.isInside(x2,y)
            test03 = self.gridBoundaryObj.isInside(x3,y)
            
            temp01 = sp.logical_xor(test01, test03)
            x2[temp01] = x3[temp01]
            temp02 = sp.logical_xor(test02, test03)
            x1[temp02] = x3[temp02]
        
        # Save calculated lengths. Some edges are cut at their negative, some at the positive ends.
        dsx[sp.logical_and(cutCellEdgesX,self.insidePoints == 1)] = ((x2+x1)*0.5)[(self.insidePoints == 1)[cutCellEdgesX]]-xPoints[sp.logical_and(cutCellEdgesX,self.insidePoints == 1)]
        dsx[sp.logical_and(cutCellEdgesX,sp.roll(self.insidePoints,-1) == 1)] = (-(x2+x1)*0.5)[sp.roll(self.insidePoints,-1)[cutCellEdgesX]]+self.dx+xPoints[sp.logical_and(cutCellEdgesX,sp.roll(self.insidePoints,-1) == 1)]                                                                          
        
        # Fail safe if some edges were identified as cut cell but aren't. 
        dsx[dsx>(1-self.cutCellAcc)*self.dx] = self.dx
        temp = dsx < self.cutCellAcc*self.dx
        self.insidePoints *= sp.logical_not(temp)
        self.insidePoints[1:] *= sp.logical_not(temp)[1:]
        dsx[temp] = 0

        # Same calculations as above, just for the edges in y-direction.
        edgesY = sp.int0(self.insidePoints)+sp.int0(sp.roll(self.insidePoints,-self.nx))       
        dsy = sp.clip(edgesY,0,1)*self.dy

        cutCellEdgesY = sp.logical_and( edgesY == 1, sp.logical_not(sp.logical_and(self.gridBoundaryObj.isInside(xPoints,yPoints),self.gridBoundaryObj.isInside(xPoints,yPoints + self.dy))) )
        
        y1 = yPoints[cutCellEdgesY]
        y2 = y1 + self.dy
        x = xPoints[cutCellEdgesY]       
        for ii in range(boundAccIter):  # @UnusedVariable
            y3 = y1+(y2-y1)/2;
            test01 = self.gridBoundaryObj.isInside(x,y1)
            test02 = self.gridBoundaryObj.isInside(x,y2)
            test03 = self.gridBoundaryObj.isInside(x,y3)
            
            temp01 = sp.logical_xor(test01, test03)
            y2[temp01] = y3[temp01]
            temp02 = sp.logical_xor(test02, test03)
            y1[temp02] = y3[temp02]
        
        dsy[sp.logical_and(cutCellEdgesY,self.insidePoints == 1)] = ((y2+y1)*0.5)[(self.insidePoints == 1)[cutCellEdgesY]]-yPoints[sp.logical_and(cutCellEdgesY,self.insidePoints == 1)]
        dsy[sp.logical_and(cutCellEdgesY,sp.roll(self.insidePoints,-self.nx) == 1)] = (-(y2+y1)*0.5)[sp.roll(self.insidePoints,-self.nx)[cutCellEdgesY]]+self.dy+yPoints[sp.logical_and(cutCellEdgesY,sp.roll(self.insidePoints,-self.nx) == 1)]                                                                          
        
        dsy[dsy>(1-self.cutCellAcc)*self.dy] = self.dy
        temp = (dsy < self.cutCellAcc*self.dy)
        self.insidePoints *= sp.logical_not(temp)
        self.insidePoints[self.nx:] *= sp.logical_not(temp)[self.nx:]
        dsy[temp] = 0

        # Calculation of faces. Primary grid. Linear approximation.
        # Note that these are not used for any essential calculation.       
        facesZ = edgesX+sp.roll(edgesX,-self.nx)+edgesY+sp.roll(edgesY,-1)     
        self.da = ( (facesZ == 2)*0.5*(dsx*dsy + sp.roll(dsx,-self.nx)*dsy + dsx*sp.roll(dsy,-1) + sp.roll(dsx,-self.nx)*sp.roll(dsy,-1)) +
                    (facesZ == 4)*0.5*(dsx + sp.roll(dsx,-self.nx))*(dsy + sp.roll(dsy,-1)) +
                    (facesZ == 6)*(self.dx*self.dy - 0.5*((self.dx-dsx)*(self.dy-dsy)+(self.dx-sp.roll(dsx,-self.nx))*(self.dy-dsy)+(self.dx-dsx)*(self.dy-sp.roll(dsy,-1))+(self.dx-sp.roll(dsx,-self.nx))*(self.dy-sp.roll(dsy,-1)))) +
                    (facesZ == 8)*self.dx*self.dy
                    )        
        self.insideCells = 1.*(facesZ == 8) + 1.*(facesZ > 0)         

        # Calculation of the final grid quantity vectors from calculations above.
        self.insidePointsInd = sp.nonzero(self.insidePoints)[0]          
        self.insideEdges =  sp.concatenate( (edgesX, edgesY) )
        self.insideEdgesInd = sp.nonzero( self.insideEdges )[0]                                        
        self.ds = sp.concatenate( (dsx, dsy) )
        self.dsi = sp.zeros(2*self.np)
        self.dsi[self.insideEdgesInd] = 1/(self.ds[self.insideEdgesInd])
        self.dai = sp.zeros(self.np)
        self.dai[self.insidePointsInd] = 1/self.da[self.insidePointsInd]       
        self.dst = sp.concatenate( (sp.tile(sp.repeat([.5,1.,.5],[1,self.nx-2,1]),self.ny)*self.dx,
                                    sp.repeat([.5,1.,.5,],[self.nx,(self.ny-2)*self.nx,self.nx])*self.dy) 
                                  )
        self.dsti = 1/self.dst
        # Dual grid faces are approximated from the primary grid faces. They are not needed for any essential
        # calculation, just for visualization of the charge density.
        self.dat = 0.25*( self.da + sp.roll(self.da,+1) + sp.roll(self.da,+self.nx) + sp.roll(self.da,1+self.nx) )
        self.dati = sp.zeros(self.np)
        self.dati[self.dat != 0] = 1./self.dat[self.dat != 0]
        
        # Standard interpolation stuff. Interpolation from edges to inside grid points.
        dsnpnm1i = sp.zeros(self.np)
        temp = self.ds[:self.np] + sp.concatenate(([0], self.ds[:self.np-1]))
        dsnpnm1i[sp.nonzero(temp)[0]] = 1/temp[sp.nonzero(temp)]
        dsnpnmnxi = sp.zeros(self.np)
        temp = self.ds[self.np:2*self.np] + sp.concatenate((sp.zeros(self.nx), self.ds[self.np:2*self.np-self.nx]))
        dsnpnmnxi[sp.nonzero(temp)[0]] = 1/temp[sp.nonzero(temp)]        

        diagX0 = sp.roll(self.ds[:self.np],1)*dsnpnm1i * self.insidePoints
        diagXM1 = sp.roll(self.ds[:self.np]*dsnpnm1i,-1) * sp.roll(self.insidePoints,-1)
        diagY0 = sp.roll(self.ds[self.np:2*self.np],self.nx)*dsnpnmnxi * self.insidePoints
        diagYMNx = sp.roll(self.ds[self.np:2*self.np]*dsnpnmnxi,-self.nx) * sp.roll(self.insidePoints,-self.nx)

        self.edgeToNode = spsp.bmat([
                                     [spsp.dia_matrix(([diagX0, diagXM1],[0, -1]), shape=(self.np, self.np)),
                                      None],
                                     [None,
                                      spsp.dia_matrix(([diagY0,diagYMNx], [0, -self.nx]), shape=(self.np, self.np))]
                                     ]
                                    )
        
        # Boundary interpolation.
        boundaryEdgesX = (edgesX == 1)
        boundaryEdgesY = (edgesY == 1)        
        temp01 = sp.tile(self.xMesh<0.,self.ny)*boundaryEdgesX
        temp02 = sp.tile(self.xMesh>0.,self.ny)*boundaryEdgesX
        temp03 = sp.repeat(self.yMesh<0.,self.nx)*boundaryEdgesY
        temp04 = sp.repeat(self.yMesh>0.,self.nx)*boundaryEdgesY

        outsidePoints = sp.logical_not(self.insidePoints)
        connectedEdges = (edgesX+sp.roll(edgesX,1)+edgesY+sp.roll(edgesY,self.nx))*outsidePoints*1.
        connectedEdges[connectedEdges != 0] = 1/connectedEdges[connectedEdges != 0]
        
        diagX0 = self.insidePoints*1.
        diagXM1 = sp.zeros(self.np)
        diagXM1[temp02] = ( self.cosAlpha**2 * self.dx/self.ds[temp02] - 
                            (self.dx - self.ds[temp02])/self.ds[temp02]
                            ) * connectedEdges[sp.roll(temp02,1)]
        diagXP1 = sp.zeros(self.np)
        diagXP1[sp.roll(temp01,1)] = ( self.cosAlpha**2 * self.dx/self.ds[temp01] - 
                                       (self.dx - self.ds[temp01])/self.ds[temp01] 
                                       ) * connectedEdges[temp01]
        diagYM1ToX = sp.zeros(self.np)
        diagYM1ToX[temp02] = ( self.cosAlpha*self.sinAlpha * self.dx/self.ds[temp02]) * connectedEdges[sp.roll(temp02,1)]
        diagYP1ToX = sp.zeros(self.np)
        diagYP1ToX[sp.roll(temp01,1)] = ( -self.cosAlpha*self.sinAlpha * self.dx/self.ds[temp01]) * connectedEdges[temp01]                      

                
        temp = temp03*1.
        temp = sp.reshape(temp,(self.ny,self.nx)).transpose()
        viewTemp03 = sp.reshape(temp03,(self.ny,self.nx))
        temp[temp == 1.] = self.cosBeta    
        self.cosBeta = temp.transpose()[viewTemp03]
        temp = temp03*1.
        temp = sp.reshape(temp,(self.ny,self.nx)).transpose()
        temp[temp == 1.] = self.sinBeta         
        self.sinBeta = temp.transpose()[viewTemp03]
        
        diagXMNx = sp.zeros(self.np)
        diagXMNx[temp04] = ( self.sinBeta[::-1]*self.sinBeta[::-1] * self.dy/self.ds[self.np:][temp04] - 
                             (self.dy - self.ds[self.np:][temp04])/self.ds[self.np:][temp04]
                             ) * connectedEdges[sp.roll(temp04,self.nx)]
        diagXPNx = sp.zeros(self.np)
        diagXPNx[sp.roll(temp03,self.nx)] = ( self.sinBeta*self.sinBeta * self.dy/self.ds[self.np:][temp03] - 
                                              (self.dy - self.ds[self.np:][temp03])/self.ds[self.np:][temp03] 
                                              ) * connectedEdges[temp03]
        diagYMNxToX = sp.zeros(self.np)
        diagYMNxToX[temp04] = ( -self.sinBeta[::-1]*self.cosBeta[::-1] * self.dy/self.ds[self.np:][temp04]) * connectedEdges[sp.roll(temp04,self.nx)]
        diagYPNxToX = sp.zeros(self.np)
        diagYPNxToX[sp.roll(temp03,self.nx)] = ( -self.sinBeta*self.cosBeta * self.dy/self.ds[self.np:][temp03]) * connectedEdges[temp03]                     
        
             
        diagY0 = self.insidePoints*1.
        diagYM1 = sp.zeros(self.np)
        diagYM1[temp02] = ( self.sinAlpha**2 * self.dx/self.ds[temp02] - 
                            (self.dx - self.ds[temp02])/self.ds[temp02]
                            ) * connectedEdges[sp.roll(temp02,1)]
        diagYP1 = sp.zeros(self.np)
        diagYP1[sp.roll(temp01,1)] = ( self.sinAlpha**2 * self.dx/self.ds[temp01] - 
                                       (self.dx - self.ds[temp01])/self.ds[temp01] 
                                       ) * connectedEdges[temp01]
        diagXM1ToY = diagYM1ToX
        diagXP1ToY = diagYP1ToX

        
        diagYMNx = sp.zeros(self.np)
        diagYMNx[temp04] = ( self.cosBeta[::-1]**2 * self.dy/self.ds[self.np:][temp04] - 
                            (self.dy - self.ds[self.np:][temp04])/self.ds[self.np:][temp04]
                            )* connectedEdges[sp.roll(temp04,self.nx)]
        diagYPNx = sp.zeros(self.np)
        diagYPNx[sp.roll(temp03,self.nx)] = ( self.cosBeta**2 * self.dy/self.ds[self.np:][temp03] - 
                                              (self.dy - self.ds[self.np:][temp03])/self.ds[self.np:][temp03] 
                                              ) * connectedEdges[temp03]
        diagXMNxToY = diagYMNxToX
        diagXPNxToY = diagYPNxToX       
           
        self.edgeToNode = spsp.bmat([
                                     [spsp.dia_matrix(([diagX0, diagXM1, diagXP1, diagXMNx, diagXPNx], [0, -1, 1, -self.nx, self.nx]), shape=(self.np, self.np)),
                                      spsp.dia_matrix(([diagYM1ToX, diagYP1ToX, diagYMNxToX, diagYPNxToX], [-1, 1, -self.nx, self.nx]), shape=(self.np, self.np))],
                                     [spsp.dia_matrix(([diagXM1ToY, diagXP1ToY, diagXMNxToY, diagXPNxToY], [-1, 1, -self.nx, self.nx]), shape=(self.np, self.np)),
                                      spsp.dia_matrix(([diagY0, diagYM1, diagYP1, diagYMNx, diagYPNx], [0, -1, 1, -self.nx, self.nx]), shape=(self.np, self.np))]
                                     ]
                                    ).dot(self.edgeToNode)

        
        # Stuff.
        temp = 0.5*outsidePoints*sp.roll(outsidePoints,1)*sp.roll(outsidePoints,-1)*sp.roll(outsidePoints,self.nx)*sp.roll(outsidePoints,-self.nx)
        temp01 = temp * sp.tile(self.xMesh <= 0, self.ny)
        temp02 = temp * sp.tile(self.xMesh >= 0, self.ny)
        temp03 = temp * sp.repeat(self.yMesh <= 0, self.nx)
        temp04 = temp * sp.repeat(self.yMesh >= 0, self.nx)
        diag = 1.*(self.insidePoints + sp.roll(self.insidePoints,-1) + sp.roll(self.insidePoints,1) + 
                   sp.roll(self.insidePoints,-self.nx) + sp.roll(self.insidePoints,self.nx))
        self.edgeToNode = spsp.bmat([
                                     [spsp.dia_matrix(([diag, sp.roll(temp02,-1), sp.roll(temp01,1), sp.roll(temp04,-self.nx), sp.roll(temp03,self.nx)], [0, -1, 1, -self.nx, self.nx]), shape=(self.np, self.np)),
                                      None],
                                     [None,
                                      spsp.dia_matrix(([diag, sp.roll(temp02,-1), sp.roll(temp01,1), sp.roll(temp04,-self.nx), sp.roll(temp03,self.nx)], [0, -1, 1, -self.nx, self.nx]), shape=(self.np, self.np))]
                                     ]
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

    cpdef double getLx(self):
        return self.lx
    
    cpdef double getLy(self):
        return self.ly
    
    cpdef double getDx(self):
        return self.dx
    
    cpdef double getDy(self):
        return self.dy    
    
    cpdef unsigned int getNx(self):
        return self.nx 
    
    cpdef unsigned int getNy(self):
        return self.ny 
    
    cpdef unsigned int getNp(self):
        return self.np 
    
    def getInsidePoints(self):
        return self.insidePoints
    
    def getInsidePointsInd(self):
        return self.insidePointsInd
    
    def getInsideEdges(self):
        return self.insideEdges
    
    def getInsideEdgesInd(self):
        return self.insideEdgesInd
    
    def getInsideCells(self):
        return self.insideEdgesInd
    
    def getDs(self):
        return self.ds
    
    def getDsi(self):
        return self.dsi
    
    def getDa(self):
        return self.da
    
    def getDai(self):
        return self.dai

    def getDst(self):
        return self.dst
     
    def getDat(self):
        return self.dat
       
    def getDati(self):
        return self.dati
        
    def getGridBasics(self):
        return ( [self.nx, self.ny], [self.lx, self.ly], [self.dx, self.dy] )
    
    def getEdgeToNode(self):
        return self.edgeToNode 
 
    def getGridBoundaryObj(self):
        return self.gridBoundaryObj    


'''
Class to describe rectangular grid boundary.

This is just a helper class with some useful functions.
Not implemented yet. Will be done later to clean some stuff up.

'''    
class GridBoundaryRectangular:
    # 
    def __init__(self, gridObj):
        
        self.gridObj = gridObj
        self.nx = gridObj.nx; self.ny = gridObj.ny;
        self.lx = gridObj.lx; self.ly = gridObj.ly;
        self.dx = gridObj.dx; self.dy = gridObj.dy; 
        self.radius = gridObj.getRadius()
              
    def isInside(self, x, y):
        # Returns True if inside boundary, False if outside.
        pass
    
    def normalVectors(self, x, y):
        # Returns the normal vector at a given point.
        pass
    
    def CosNormalAngle(self, xOrYMesh):
        # Returns the cosine of the angle of a coordinate axis to the normal vector.
        pass
 
   
'''
Class to describe elliptical grid boundary.

This is just a helper class with some useful functions.

'''     
class GridBoundaryElliptical:
 
#     cdef:
#         object gridObj
#         double lx, ly, lxHalfSqInv, lyHalfSqInv

    def __init__(self, gridObj):
         
        self.gridObj = gridObj
        self.lx = gridObj.getLx()
        self.ly = gridObj.getLy()
     
    def CosSinNormalAngleX(self, y):#numpy.ndarray[numpy.double_t] yNumpy):
        
#         cdef:
#             double lxHalfSq = self.lx*self.lx/4.
#             double lxHalfSqInv = 1./lxHalfSq, lyHalfSqInv = 4./(self.ly*self.ly)
#             double lxHalfP4Inv = lxHalfSqInv*lxHalfSqInv, lyHalfP4Inv = lyHalfSqInv*lyHalfSqInv
#             
#             unsigned int yLength = y.shape[0]
#             numpy.ndarray[numpy.double_t] sinAlphaNumpy = numpy.empty(yLength, dtype=numpy.double)
#             double *sinAlpha = &sinAlphaNumpy[0], *y = &yNumpy[0]
#             double norm
#             unsigned int ii
#             
#         for ii in range(yLength):
#             norm = sqrt(((1.-y[ii]*y[ii]*lyHalfSqInv)*lxHalfSq)*lxHalfP4Inv + y[ii]*y[ii]*lyHalfP4Inv
        
        b = self.ly*.5
        a = self.lx*.5
        x = numpy.sqrt((1-y**2/b**2)*a**2)
        norm = numpy.sqrt(x**2/a**4+y**2/b**4)
        sinAlpha = y/b**2/norm
        cosAlpha = numpy.sqrt(1-sinAlpha**2)
        return ( cosAlpha, 
                 sinAlpha 
                 )
          
    def CosSinNormalAngleY(self, x):
        b = self.ly*.5
        a = self.lx*.5
        y = numpy.sqrt((1-x**2/a**2)*b**2)
        norm = numpy.sqrt(x**2/a**4+y**2/b**4)
        sinBeta = x/a**2/norm
        cosBeta = numpy.sqrt(1-sinBeta**2)
        return ( cosBeta, 
                 sinBeta
                 )
               
    def isInside(self, x, y):      
        return 1. >= x**2*(2./self.lx)**2 + y**2*(2./self.ly)**2


class GridBoundaryArbitrary:
 
    def __init__(self, gridObj, boundFunc):
         
        self.gridObj = gridObj
        self.isInside = boundFunc
        ( [self.nx, self.ny], [self.lx, self.ly], [self.dx, self.dy] ) = gridObj.getGridBasics()
     
    def CosSinNormalAngleX(self, y):
        pass
          
    def CosSinNormalAngleY(self, x):
        pass

    
