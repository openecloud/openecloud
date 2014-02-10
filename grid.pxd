cimport numpy

cdef class Grid:

    cdef:
        unsigned int nx, ny, np
        double lx, ly, dx, dy, cutCellAcc, cutCellMinEdgeLength, scanAroundPoint
        numpy.ndarray xMesh, yMesh, cosAlpha, sinAlpha, cosBeta, sinBeta 
        numpy.ndarray insidePoints, insidePointsInd, insideEdges, insideEdgesInd
        numpy.ndarray insideCells, insideCellsInd
        numpy.ndarray ds, dsi, da, dai, dst, dsti, dat, dati
        object gridBoundaryObj, edgeToNode
        
    cpdef double getDx(Grid self)
    cpdef double getDy(Grid self)
    cpdef double getLx(Grid self)
    cpdef double getLy(Grid self)   
    cpdef unsigned int getNx(Grid self)
    cpdef unsigned int getNy(Grid self)
    cpdef unsigned int getNp(Grid self)
    cpdef double[:] getXMesh(Grid self)
    cpdef double[:] getYMesh(Grid self)
