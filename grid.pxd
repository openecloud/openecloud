cimport numpy

cdef class Grid:

    cdef:
        unsigned int nx, ny, np, nxExt, nyExt, npExt
        double lx, ly, lxExt, lyExt, dx, dy, cutCellAcc, cutCellMinEdgeLength, scanAroundPoint
        double[:] xMesh, yMesh, cosAlpha, sinAlpha, cosBeta, sinBeta
        double[:] ds, dsi, da, dai, dst, dsti, dat, dati
        double[:,:] boundaryPoints
        unsigned int[:] boundaryPointsInd
        int[:,:] cutCellPointsInd
        double[:,:] cutCellCenter
        double[:,:] cutCellNormalVectors
        unsigned short[:] insidePoints, insideEdges, insideFaces 
        object boundFunc, edgeToNode
        
        void computeStaircaseGridGeom(Grid self)
        void computeCutCellGridGeom(Grid self)
    cpdef double getDx(Grid self)
    cpdef double getDy(Grid self)
    cpdef double getLx(Grid self)
    cpdef double getLy(Grid self)   
    cpdef unsigned int getNx(Grid self)
    cpdef unsigned int getNy(Grid self)
    cpdef unsigned int getNp(Grid self)
    cpdef double[:] getXMesh(Grid self)
    cpdef double[:] getYMesh(Grid self)
    cpdef double[:] getDs(Grid self)
    cpdef double[:] getDst(Grid self)
    cpdef double getLxExt(Grid self)
    cpdef double getLyExt(Grid self)
    cpdef double getDx(Grid self)
    cpdef double getDy(Grid self)
    cpdef unsigned int getNxExt(Grid self)
    cpdef unsigned int getNyExt(Grid self)
    cpdef unsigned int getNpExt(Grid self)
    cpdef double[:,:] getBoundaryPoints(Grid self)
    cpdef unsigned int[:] getBoundaryPointsInd(Grid self)
    cpdef int[:,:] getCutCellPointsInd(Grid self)
    cpdef object getBoundFunc(Grid self)
    cpdef unsigned short[:] getInsideFaces(Grid self)
    cpdef unsigned int[:] getInCell(Grid self)
    cpdef double[:,:] getCutCellCenter(Grid self)
    cpdef double[:,:] getCutCellNormalVectors(Grid self)
    
                   
