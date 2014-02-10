#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy
cimport numpy
cimport cython

# Import some C methods.
cdef extern from "math.h":
    double sqrt(double x) nogil
    double fmod(double x, double y) nogil


cdef class ParticleBoundary:
    
    cdef:
        object gridObj, particlesObj
        unsigned int nx, ny, absorbedMacroParticleCount
        double lx, ly, dx, dy
        numpy.ndarray absorbedParticles, remainingTimeStep, normalVectors

    def __init__(self, object gridObj, object particlesObj):
        
        self.gridObj = gridObj
        self.particlesObj = particlesObj
        ( [self.nx, self.ny], [self.lx, self.ly], [self.dx, self.dy] ) = gridObj.getGridBasics()
        
        self.absorbedParticles = numpy.empty((1,particlesObj.getNCoords()), dtype=numpy.double)
        self.normalVectors = numpy.empty((1,2), dtype=numpy.double)
        self.remainingTimeStep = numpy.empty(1, dtype=numpy.double)
        self.absorbedMacroParticleCount = 0
        
    cpdef saveAbsorbed(self):
        cdef:
            double[:,:] particleDataBuff = self.particlesObj.getParticleData()
            double *particleData = &particleDataBuff[0,0]
            unsigned int macroParticleCount = self.particlesObj.getMacroParticleCount()
            unsigned int nCoords = self.particlesObj.getNCoords()
            unsigned int absorbedMacroParticleCount = macroParticleCount, ii, jj, kk, ind1, ind2
            unsigned short[:] isInside = self.particlesObj.getIsInside()
            double[:,:] absorbedParticlesBuff
            double *absorbedParticles
            
        for ii in range(macroParticleCount):
            absorbedMacroParticleCount -= isInside[ii]
        
        if absorbedMacroParticleCount > self.absorbedParticles.shape[0]:
            self.absorbedParticles = numpy.empty((absorbedMacroParticleCount, nCoords), dtype = numpy.double)
        self.absorbedMacroParticleCount = absorbedMacroParticleCount
        
        absorbedParticlesBuff = self.absorbedParticles
        absorbedParticles = &absorbedParticlesBuff[0,0]
        
        ii = 0; jj = 0;
        while ii < absorbedMacroParticleCount:
            if isInside[jj] == 0:
                ind1 = nCoords*ii
                ind2 = nCoords*jj
                for kk in range(nCoords):
                    absorbedParticles[ind1+kk] = particleData[ind2+kk]
                ii += 1
            jj += 1
            

    cpdef getAbsorbedParticles(self):
        return self.absorbedParticles[:self.absorbedMacroParticleCount]
    
    cpdef getAbsorbedMacroParticleCount(self):
        return self.absorbedMacroParticleCount   
    
    cpdef getNormalVectors(self):
        return self.normalVectors[:self.absorbedMacroParticleCount]   
    
        
cdef class AbsorbRectangular(ParticleBoundary):
    
    cdef:
        numpy.ndarray particleLeftWhere  

    def __init__(self, object gridObj, object particlesObj):
        
        super().__init__(gridObj, particlesObj)
        self.particleLeftWhere = numpy.empty(1, dtype=numpy.uint16)
        
                
    cpdef indexInside(self):
        cdef:
            double[:,:] particleDataBuff = self.particlesObj.getParticleData()
            double *particleData = &particleDataBuff[0,0]
            unsigned int macroParticleCount = self.particlesObj.getMacroParticleCount()
            unsigned int nCoords = self.particlesObj.getNCoords()
            double lxHalf = self.lx*0.5, lyHalf = self.ly*0.5
            unsigned int ii
            unsigned short[:] isInsideBuff = self.particlesObj.getIsInside()
            unsigned short *isInside = &isInsideBuff[0]
        
        for ii in range(macroParticleCount):
            if (particleData[nCoords*ii]<-lxHalf or particleData[nCoords*ii]>lxHalf or
                particleData[nCoords*ii+1]<-lyHalf or particleData[nCoords*ii+1]>lyHalf):
                isInside[ii] = 0
            else:
                isInside[ii] = 1
                
    cpdef unsigned short isInside(self, double[:] point):
        cdef:
            double lxHalf = self.lx*0.5, lyHalf = self.ly*0.5
        if (point[0]<-lxHalf or point[0]>lxHalf or
            point[1]<-lyHalf or point[1]>lyHalf):
            return 0
        else:
            return 1        
                
    cpdef calculateInteractionPoint(self):       
        
        cdef:
            double[:,:] absorbedParticlesBuff = self.absorbedParticles
            double *absorbedParticles = &absorbedParticlesBuff[0,0]
            unsigned int absorbedMacroParticleCount = self.absorbedMacroParticleCount
            unsigned int nCoords = self.particlesObj.getNCoords()
            double xIsInside, yIsInside, tempRemainingTimeStepX, tempRemainingTimeStepY
            double epsFailsafe = 1e-6
            double lxHalf = self.lx*0.5-self.dx*epsFailsafe, lyHalf = self.ly*0.5-self.dy*epsFailsafe      
            unsigned int ii
            double[:] remainingTimeStepNumpy
            double *remainingTimeStep
            unsigned short[:] particleLeftWhereNumpy
            unsigned short *particleLeftWhere
        
        if self.remainingTimeStep.shape[0]<absorbedMacroParticleCount:    
            self.remainingTimeStep = numpy.empty(absorbedMacroParticleCount, dtype=numpy.double)
        if self.particleLeftWhere.shape[0]<absorbedMacroParticleCount:
            self.particleLeftWhere = numpy.empty(absorbedMacroParticleCount, dtype=numpy.uint16)
        
        particleLeftWhereNumpy = self.particleLeftWhere
        particleLeftWhere = &particleLeftWhereNumpy[0]
        remainingTimeStepNumpy = self.remainingTimeStep
        remainingTimeStep = &remainingTimeStepNumpy[0]
                    
        for ii in range(absorbedMacroParticleCount):
            xIsInside = 1
            yIsInside = 1
            if absorbedParticles[nCoords*ii]>lxHalf:
                xIsInside = 2
            elif absorbedParticles[nCoords*ii]<-lxHalf:
                xIsInside = 0
            if absorbedParticles[nCoords*ii+1]>lyHalf:
                yIsInside = 2
            elif absorbedParticles[nCoords*ii+1]<-lyHalf:
                yIsInside = 0
            
            if xIsInside == 0 and yIsInside == 1:
                remainingTimeStep[ii] = (absorbedParticles[nCoords*ii]+lxHalf)/absorbedParticles[nCoords*ii+2] 
                particleLeftWhere[ii] = 0  
            elif xIsInside == 2 and yIsInside == 1:
                remainingTimeStep[ii] = (absorbedParticles[nCoords*ii]-lxHalf)/absorbedParticles[nCoords*ii+2]
                particleLeftWhere[ii] = 1
            elif xIsInside == 1 and yIsInside == 0:
                remainingTimeStep[ii] = (absorbedParticles[nCoords*ii+1]+lyHalf)/absorbedParticles[nCoords*ii+3]   
                particleLeftWhere[ii] = 2 
            elif xIsInside == 1 and yIsInside == 2:
                remainingTimeStep[ii] = (absorbedParticles[nCoords*ii+1]-lyHalf)/absorbedParticles[nCoords*ii+3]
                particleLeftWhere[ii] = 3 
            elif xIsInside == 0 and yIsInside == 0:
                remainingTimeStepX = (absorbedParticles[nCoords*ii]+lxHalf)/absorbedParticles[nCoords*ii+2] 
                remainingTimeStepY = (absorbedParticles[nCoords*ii+1]+lyHalf)/absorbedParticles[nCoords*ii+3]   
                if remainingTimeStepX > remainingTimeStepY:
                    remainingTimeStep[ii] = remainingTimeStepX
                    particleLeftWhere[ii] = 0
                else:
                    remainingTimeStep[ii] = remainingTimeStepY
                    particleLeftWhere[ii] = 1
            elif xIsInside == 2 and yIsInside == 0:
                remainingTimeStepX = (absorbedParticles[nCoords*ii]-lxHalf)/absorbedParticles[nCoords*ii+2] 
                remainingTimeStepY = (absorbedParticles[nCoords*ii+1]+lyHalf)/absorbedParticles[nCoords*ii+3]   
                if remainingTimeStepX > remainingTimeStepY:
                    remainingTimeStep[ii] = remainingTimeStepX
                    particleLeftWhere[ii] = 2
                else:
                    remainingTimeStep[ii] = remainingTimeStepY
                    particleLeftWhere[ii] = 1
            elif xIsInside == 0 and yIsInside == 2:
                remainingTimeStepX = (absorbedParticles[nCoords*ii]+lxHalf)/absorbedParticles[nCoords*ii+2] 
                remainingTimeStepY = (absorbedParticles[nCoords*ii+1]-lyHalf)/absorbedParticles[nCoords*ii+3]   
                if remainingTimeStepX > remainingTimeStepY:
                    remainingTimeStep[ii] = remainingTimeStepX
                    particleLeftWhere[ii] = 0
                else:
                    remainingTimeStep[ii] = remainingTimeStepY
                    particleLeftWhere[ii] = 3
            else:
                remainingTimeStepX = (absorbedParticles[nCoords*ii]-lxHalf)/absorbedParticles[nCoords*ii+2] 
                remainingTimeStepY = (absorbedParticles[nCoords*ii+1]-lyHalf)/absorbedParticles[nCoords*ii+3]   
                if remainingTimeStepX > remainingTimeStepY:
                    remainingTimeStep[ii] = remainingTimeStepX
                    particleLeftWhere[ii] = 2
                else:
                    remainingTimeStep[ii] = remainingTimeStepY
                    particleLeftWhere[ii] = 3
            
            absorbedParticles[nCoords*ii] -= remainingTimeStep[ii]*absorbedParticles[nCoords*ii+2]
            absorbedParticles[nCoords*ii+1] -= remainingTimeStep[ii]*absorbedParticles[nCoords*ii+3]


    cpdef calculateNormalVectors(self):       
        cdef:
            numpy.ndarray[numpy.double_t, ndim=2] absorbedParticlesNumpy = self.absorbedParticles
            double *absorbedParticles = &absorbedParticlesNumpy[0,0]
            unsigned int absorbedMacroParticleCount = self.absorbedMacroParticleCount
            unsigned int nCoords = self.particlesObj.getNCoords()
            numpy.ndarray[numpy.uint16_t] particleLeftWhereNumpy = self.particleLeftWhere
            unsigned short *particleLeftWhere = &particleLeftWhereNumpy[0]
            unsigned int ii
            numpy.ndarray[numpy.double_t, ndim=2] normalVectorsNumpy
            double *normalVectors
        
        if self.normalVectors.shape[0]<absorbedMacroParticleCount:    
            self.normalVectors = numpy.empty((absorbedMacroParticleCount,2), dtype=numpy.double)
        
        normalVectorsNumpy = self.normalVectors
        normalVectors = &normalVectorsNumpy[0,0]
        
        for ii in range(absorbedMacroParticleCount):
            if particleLeftWhere[ii] == 0:
                normalVectors[2*ii] = 1.
                normalVectors[2*ii+1] = 0.
            elif particleLeftWhere[ii] == 1:
                normalVectors[2*ii] = -1.
                normalVectors[2*ii+1] = 0.
            elif particleLeftWhere[ii] == 2:
                normalVectors[2*ii] = 0.
                normalVectors[2*ii+1] = 1.
            elif particleLeftWhere[ii] == 3:
                normalVectors[2*ii] = 0.
                normalVectors[2*ii+1] = -1.

            
            
cdef class AbsorbElliptical(ParticleBoundary):
     
    cpdef indexInside(self):
 
        cdef:
            double[:,:] particleDataBuff = self.particlesObj.getParticleData()
            double *particleData = &particleDataBuff[0,0]
            unsigned int macroParticleCount = self.particlesObj.getMacroParticleCount()
            unsigned int nCoords = self.particlesObj.getNCoords()            
            unsigned int ii
            double lxHalfSqInv = 4./(self.lx*self.lx), lyHalfSqInv = 4./(self.ly*self.ly)
            unsigned short[:] isInside = self.particlesObj.getIsInside()
         
        for ii in range(macroParticleCount):
            if (particleData[nCoords*ii]*particleData[nCoords*ii]*lxHalfSqInv + 
                particleData[nCoords*ii+1]*particleData[nCoords*ii+1]*lyHalfSqInv) >= 1.:
                isInside[ii] = 0
            else:
                isInside[ii] = 1
        
    cpdef unsigned short isInside(self, double[:] point):
        cdef:
            double lxHalfSqInv = 4./self.lx**2, lyHalfSqInv = 4./self.ly**2
        if (point[0]*point[0]*lxHalfSqInv + point[1]*point[1]*lyHalfSqInv) >= 1.:
            return 0
        else:
            return 1 
                            
    cpdef calculateInteractionPoint(self):   
         
        cdef: 
            double[:,:] absorbedParticlesBuff = self.absorbedParticles
            double *absorbedParticles = &absorbedParticlesBuff[0,0]   
            double a, b, c
            unsigned int ii 
            unsigned int absorbedMacroParticleCount = self.absorbedMacroParticleCount
            unsigned int nCoords = self.particlesObj.getNCoords()
            double epsFailsafe = 1.e-6
            double lxHalfSqInv = 4./(self.lx-epsFailsafe*self.dx)**2, lyHalfSqInv = 4./(self.ly-epsFailsafe*self.dy)**2
            double[:] remainingTimeStepNumpy
            double *remainingTimeStep
         
        if absorbedMacroParticleCount > self.remainingTimeStep.shape[0]:
            self.remainingTimeStep = numpy.empty(absorbedMacroParticleCount, dtype=numpy.double)  
        remainingTimeStepNumpy = self.remainingTimeStep
        remainingTimeStep = &remainingTimeStepNumpy[0]           
         
        for ii in range(absorbedMacroParticleCount):
            c = absorbedParticles[nCoords*ii]*absorbedParticles[nCoords*ii]*lxHalfSqInv + absorbedParticles[nCoords*ii+1]*absorbedParticles[nCoords*ii+1]*lyHalfSqInv - 1.
            b = absorbedParticles[nCoords*ii]*absorbedParticles[nCoords*ii+2]*lxHalfSqInv + absorbedParticles[nCoords*ii+1]*absorbedParticles[nCoords*ii+3]*lyHalfSqInv
            a = absorbedParticles[nCoords*ii+2]*absorbedParticles[nCoords*ii+2]*lxHalfSqInv + absorbedParticles[nCoords*ii+3]*absorbedParticles[nCoords*ii+3]*lyHalfSqInv
            remainingTimeStep[ii] = (b - sqrt(b*b - a*c))/a
            absorbedParticles[nCoords*ii] -= remainingTimeStep[ii]*absorbedParticles[nCoords*ii+2]
            absorbedParticles[nCoords*ii+1] -= remainingTimeStep[ii]*absorbedParticles[nCoords*ii+3]
 
 
    cpdef calculateNormalVectors(self):      
    
        cdef: 
            double[:,:] absorbedParticlesBuff = self.absorbedParticles
            double *absorbedParticles = &absorbedParticlesBuff[0,0]   
            unsigned int absorbedMacroParticleCount = self.absorbedMacroParticleCount    
            unsigned int nCoords = self.particlesObj.getNCoords()       
            unsigned int ii
            double lxHalfSqInv = 4./(self.lx*self.lx), lyHalfSqInv = 4./(self.ly*self.ly)
            double lxHalfP4Inv = lxHalfSqInv*lxHalfSqInv, lyHalfP4Inv = lyHalfSqInv*lyHalfSqInv
            double[:,:] normalVectorsBuff
            double *normalVectors
            double normInv
                         
        if absorbedMacroParticleCount > self.normalVectors.shape[0]:
            self.normalVectors = numpy.empty((absorbedMacroParticleCount,2), dtype=numpy.double)
             
        normalVectorsBuff = self.normalVectors
        normalVectors = &normalVectorsBuff[0,0]    
         
        for ii in range(absorbedMacroParticleCount):
            normInv = 1./sqrt(absorbedParticles[nCoords*ii]*absorbedParticles[nCoords*ii]*lxHalfP4Inv + absorbedParticles[nCoords*ii+1]*absorbedParticles[nCoords*ii+1]*lyHalfP4Inv)
            normalVectors[2*ii] = -absorbedParticles[nCoords*ii]*lxHalfSqInv*normInv
            normalVectors[2*ii+1] = -absorbedParticles[nCoords*ii+1]*lyHalfSqInv*normInv
 



# cdef class AbsorbArbitrary(ParticleBoundary):
#      
#     cpdef indexInside(self):
#  
#         cdef:
#             numpy.ndarray[numpy.double_t, ndim=2] particleDataNumpy = self.particlesObj.getParticleData()
#             double *particleData = &particleDataNumpy[0,0]
#             numpy.ndarray[numpy.double_t] inCellNumpy = self.particlesObj.getInCell()
#             double *inCell = &inCellNumpy[0]
#             numpy.ndarray[numpy.uint16_t] cellTypeNumpy = self.gridObj.getCellType()
#             double *cellType = &cellTypeNumpy[0]
#             unsigned int macroParticleCount = self.particlesObj.getMacroParticleCount()
#             unsigned int nCoords = self.particlesObj.getNCoords()            
#             unsigned int ii
#             double lxHalfSqInv = 4./self.lx**2, lyHalfSqInv = 4./self.ly**2
#             numpy.ndarray[numpy.uint16_t] isInsideNumpy
#             unsigned int *isInside  
#             
#         if self.inside.shape[0]<macroParticleCount:
#             self.isInside = numpy.empty(macroParticleCount, dtype=numpy.uint16)
#         isInsideNumpy = self.isInside
#         isInside = &isInsideNumpy[0]
#         
#         for ii in range(macroParticleCount):
#             if cellType[inCell[ii]] == 2:
#                 isInside[ii] = 1
#             elif cellType[inCell[ii]] == 1:
#                 isInside[ii] = 0
#             else:
                
         
#     def calculateInteractionPoint(self):       
#         self.remainingTimeStep = numpy.empty(self.absorbedParticles.shape[0], order='C')
#         absorbCircularC.calculateInteractionPoint(self.absorbedParticles, self.remainingTimeStep, self.gridObj.getRadius(), self.epsFailsafe)
# 
#     def calculateNormalVectors(self):       
#         self.normalVectors = numpy.empty((self.absorbedParticles.shape[0],2), order='C')
#         absorbCircularC.calculateNormalVectors(self.absorbedParticles, self.normalVectors)
#         
#     def clipToDomain(self):
#         particleData = self.particlesObj.getParticleData()
#         particleDataRadiusClip = numpy.sqrt(particleData[:,0]**2 + particleData[:,1]**2)
#         particleDataRadiusClip = numpy.clip(particleDataRadiusClip,0,self.gridObj.getRadius())/particleDataRadiusClip
#         particleData[:,:2] *= particleDataRadiusClip[:,numpy.newaxis]

# Checks if point is in konvex polygon.
# http://demonstrations.wolfram.com/AnEfficientTestForAPointToBeInAConvexPolygon/
cdef unsigned int pointInPolygon(double* pointCoords, double* cornerCoords, unsigned int nCorners) nogil:
    cdef:
        unsigned int currentInd, ii, isInside = 1
        double[10] cornerCoordsTrans
        double ai
        
    for ii in range(nCorners):
        cornerCoordsTrans[2*ii] = cornerCoords[2*ii] - pointCoords[0]
        cornerCoordsTrans[2*ii+1] = cornerCoords[2*ii+1] - pointCoords[1]
    
    ai = cornerCoordsTrans[2]*cornerCoordsTrans[1] - cornerCoordsTrans[0]*cornerCoordsTrans[3]
    if ai>=0.:
        for ii in range(1, nCorners):
            currentInd = <unsigned int> (fmod(ii, nCorners-1) + 0.1)
            ai = cornerCoordsTrans[2*ii+2]*cornerCoordsTrans[2*ii+1] - \
                 cornerCoordsTrans[2*currentInd]*cornerCoordsTrans[2*currentInd+3]
            if ai<0:
                isInside = 0
                break
    else:
        for ii in range(1, nCorners):
            currentInd = <unsigned int> (fmod(ii, nCorners-1) + 0.1)
            ai = cornerCoordsTrans[2*ii+2]*cornerCoordsTrans[2*ii+1] - \
                 cornerCoordsTrans[2*currentInd]*cornerCoordsTrans[2*currentInd+3]
            if ai>0:
                isInside = 0
                break
    return isInside
            
            
