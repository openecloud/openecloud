cimport grid
cimport particles


cdef class ParticleBoundary:
    
    cdef:
        grid.Grid gridObj
        particles.Particles particlesObj
        unsigned int nx, ny, np, absorbedMacroParticleCount
        double lx, ly, dx, dy
        double[:,:] absorbedParticles, normalVectors
        double[:] remainingTimeStep
        
    cpdef object saveAbsorbed(ParticleBoundary self)
    cpdef double[:,:] getAbsorbedParticles(ParticleBoundary self)
    cpdef unsigned int getAbsorbedMacroParticleCount(ParticleBoundary self)
    cpdef double[:,:] getNormalVectors(ParticleBoundary self)
        
        
cdef class AbsorbRectangular(ParticleBoundary):
    
    cdef:
        unsigned short[:] particleLeftWhere  
        
    cpdef indexInside(AbsorbRectangular self)
    cpdef unsigned short isInside(AbsorbRectangular self, double x, double y)
    cpdef object calculateInteractionPoint(AbsorbRectangular self)
    cpdef object calculateNormalVectors(AbsorbRectangular self)
    
    
cdef class AbsorbElliptical(ParticleBoundary):

    cpdef object indexInside(AbsorbElliptical self)
    cpdef unsigned short isInside(AbsorbElliptical self, double x, double y)
    cpdef object calculateInteractionPoint(AbsorbElliptical self)
    cpdef object calculateNormalVectors(AbsorbElliptical self)
    
    
cdef class AbsorbArbitraryCutCell(ParticleBoundary):

    cpdef object indexInside(AbsorbArbitraryCutCell self)
    cpdef unsigned short isInside(AbsorbArbitraryCutCell self, double x, double y)
    cpdef object calculateInteractionPoint(AbsorbArbitraryCutCell self)
    cpdef object calculateNormalVectors(AbsorbArbitraryCutCell self)
