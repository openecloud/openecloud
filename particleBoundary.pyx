#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy
cimport numpy
cimport cython
cimport grid
cimport particles
from constants cimport *

# Import some C methods.
cdef extern from "math.h":
    double sqrt(double x) nogil
    double fmod(double x, double y) nogil


cdef class ParticleBoundary:

    def __init__(ParticleBoundary self, grid.Grid gridObj, particles.Particles particlesObj):
        
        self.gridObj = gridObj
        self.particlesObj = particlesObj
        self.nx = gridObj.getNxExt()
        self.ny = gridObj.getNyExt()
        self.np = gridObj.getNpExt()
        self.lx = gridObj.getLx()
        self.ly = gridObj.getLy()  
        self.dx = gridObj.getDx()
        self.dy = gridObj.getDy()
        
        self.absorbedParticles = numpy.empty((1,particlesObj.getNCoords()), dtype=numpy.double)
        self.normalVectors = numpy.empty((1,2), dtype=numpy.double)
        self.remainingTimeStep = numpy.empty(1, dtype=numpy.double)
        self.absorbedMacroParticleCount = 0
        
    cpdef object saveAbsorbed(ParticleBoundary self):
        cdef:
            double *particleData = &self.particlesObj.getFullParticleData()[0,0]
            unsigned int macroParticleCount = self.particlesObj.getMacroParticleCount()
            unsigned int nCoords = self.particlesObj.getNCoords()
            unsigned int absorbedMacroParticleCount, ii, jj, kk, ind1, ind2
            unsigned short* isInside = &self.particlesObj.getIsInside()[0]
            double *absorbedParticles
            
        absorbedMacroParticleCount = macroParticleCount
        for ii in range(macroParticleCount):
            absorbedMacroParticleCount -= isInside[ii]
        
        if absorbedMacroParticleCount > self.absorbedParticles.shape[0]:
            self.absorbedParticles = numpy.empty((<unsigned int> (absorbedMacroParticleCount*1.1), nCoords), dtype = numpy.double)
        self.absorbedMacroParticleCount = absorbedMacroParticleCount
        
        absorbedParticles = &self.absorbedParticles[0,0]
        
        ii = 0; jj = 0;
        while ii < absorbedMacroParticleCount:
            if isInside[jj] == 0:
                ind1 = nCoords*ii
                ind2 = nCoords*jj
                for kk in range(nCoords):
                    absorbedParticles[ind1+kk] = particleData[ind2+kk]
                ii += 1
            jj += 1
            
    cpdef double[:,:] getAbsorbedParticles(ParticleBoundary self):
        return self.absorbedParticles[:self.absorbedMacroParticleCount]
    
    cpdef unsigned int getAbsorbedMacroParticleCount(ParticleBoundary self):
        return self.absorbedMacroParticleCount   
    
    cpdef double[:,:] getNormalVectors(ParticleBoundary self):
        return self.normalVectors[:self.absorbedMacroParticleCount]   
    
        
cdef class AbsorbRectangular(ParticleBoundary):

    def __init__(AbsorbRectangular self, grid.Grid gridObj, particles.Particles particlesObj):
        
        super().__init__(gridObj, particlesObj)
        self.particleLeftWhere = numpy.empty(1, dtype=numpy.ushort)
                       
    cpdef indexInside(AbsorbRectangular self):
        cdef:
            double* particleData = &self.particlesObj.getFullParticleData()[0,0]
            unsigned int macroParticleCount = self.particlesObj.getMacroParticleCount()
            unsigned int nCoords = self.particlesObj.getNCoords()
            double lxHalf = self.lx*0.5, lyHalf = self.ly*0.5
            unsigned int ii
            unsigned short* isInside = &self.particlesObj.getIsInside()[0]
        
        for ii in range(macroParticleCount):
            if (particleData[nCoords*ii]<-lxHalf or particleData[nCoords*ii]>lxHalf or
                particleData[nCoords*ii+1]<-lyHalf or particleData[nCoords*ii+1]>lyHalf):
                isInside[ii] = 0
            else:
                isInside[ii] = 1
                
    cpdef unsigned short isInside(AbsorbRectangular self, double x, double y):
        cdef:
            double lxHalf = self.lx*0.5, lyHalf = self.ly*0.5
        if x<-lxHalf or x>lxHalf or y<-lyHalf or y>lyHalf:
            return 0
        else:
            return 1        
                
    cpdef object calculateInteractionPoint(AbsorbRectangular self):       
        
        cdef:
            double *absorbedParticles = &self.absorbedParticles[0,0]
            unsigned int absorbedMacroParticleCount = self.absorbedMacroParticleCount
            unsigned int nCoords = self.particlesObj.getNCoords()
            unsigned short xIsInside, yIsInside 
            double tempRemainingTimeStepX, tempRemainingTimeStepY
            double epsFailsafe = 1e-6
            double lxHalf = self.lx*0.5-self.dx*epsFailsafe, lyHalf = self.ly*0.5-self.dy*epsFailsafe      
            unsigned int ii
            double *remainingTimeStep
            unsigned short *particleLeftWhere
        
        if self.remainingTimeStep.shape[0]<absorbedMacroParticleCount:    
            self.remainingTimeStep = numpy.empty(<unsigned int> (absorbedMacroParticleCount*1.1), dtype=numpy.double)
        if self.particleLeftWhere.shape[0]<absorbedMacroParticleCount:
            self.particleLeftWhere = numpy.empty(<unsigned int> (absorbedMacroParticleCount*1.1), dtype=numpy.ushort)
        
        particleLeftWhere = &self.particleLeftWhere[0]
        remainingTimeStep = &self.remainingTimeStep[0]
                    
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


    cpdef object calculateNormalVectors(AbsorbRectangular self):       
        cdef:
            double *absorbedParticles = &self.absorbedParticles[0,0]
            unsigned int absorbedMacroParticleCount = self.absorbedMacroParticleCount
            unsigned int nCoords = self.particlesObj.getNCoords()
            unsigned short *particleLeftWhere = &self.particleLeftWhere[0]
            unsigned int ii
            double *normalVectors
        
        if self.normalVectors.shape[0]<absorbedMacroParticleCount:    
            self.normalVectors = numpy.empty((<unsigned int> (absorbedMacroParticleCount*1.1),2), dtype=numpy.double)
        
        normalVectors = &self.normalVectors[0,0]
        
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
     
    cpdef object indexInside(AbsorbElliptical self):
 
        cdef:
            double *particleData = &self.particlesObj.getFullParticleData()[0,0]
            unsigned int macroParticleCount = self.particlesObj.getMacroParticleCount()
            unsigned int nCoords = self.particlesObj.getNCoords()            
            unsigned int ii
            double lxHalfSqInv = 4./(self.lx*self.lx), lyHalfSqInv = 4./(self.ly*self.ly)
            unsigned short *isInside = &self.particlesObj.getIsInside()[0]
         
        for ii in range(macroParticleCount):
            if (particleData[nCoords*ii]*particleData[nCoords*ii]*lxHalfSqInv + 
                particleData[nCoords*ii+1]*particleData[nCoords*ii+1]*lyHalfSqInv) >= 1.:
                isInside[ii] = 0
            else:
                isInside[ii] = 1
        
    cpdef unsigned short isInside(AbsorbElliptical self, double x, double y):
        cdef:
            double lxHalfSqInv = 4./self.lx**2, lyHalfSqInv = 4./self.ly**2
        if (x**2*lxHalfSqInv + y**2*lyHalfSqInv) >= 1.:
            return 0
        else:
            return 1 
                            
    cpdef object calculateInteractionPoint(AbsorbElliptical self):   
         
        cdef: 
            double *absorbedParticles = &self.absorbedParticles[0,0]   
            double a, b, c
            unsigned int ii 
            unsigned int nCoords = self.particlesObj.getNCoords()
            double epsFailsafe = 1.e-6
            double lxHalfSqInv = 4./(self.lx-epsFailsafe*self.dx)**2, lyHalfSqInv = 4./(self.ly-epsFailsafe*self.dy)**2
            double *remainingTimeStep
         
        if self.absorbedMacroParticleCount > self.remainingTimeStep.shape[0]:
            self.remainingTimeStep = numpy.empty(<unsigned int> (self.absorbedMacroParticleCount*1.1), dtype=numpy.double)  
        remainingTimeStep = &self.remainingTimeStep[0]           
         
        for ii in range(self.absorbedMacroParticleCount):
            c = absorbedParticles[nCoords*ii]**2*lxHalfSqInv + absorbedParticles[nCoords*ii+1]**2*lyHalfSqInv - 1.
            b = absorbedParticles[nCoords*ii]*absorbedParticles[nCoords*ii+2]*lxHalfSqInv + \
                absorbedParticles[nCoords*ii+1]*absorbedParticles[nCoords*ii+3]*lyHalfSqInv
            a = absorbedParticles[nCoords*ii+2]**2*lxHalfSqInv + absorbedParticles[nCoords*ii+3]**2*lyHalfSqInv
            remainingTimeStep[ii] = (b - sqrt(b*b - a*c))/a
            absorbedParticles[nCoords*ii] -= remainingTimeStep[ii]*absorbedParticles[nCoords*ii+2]
            absorbedParticles[nCoords*ii+1] -= remainingTimeStep[ii]*absorbedParticles[nCoords*ii+3]
 
 
    cpdef object calculateNormalVectors(AbsorbElliptical self):      
    
        cdef: 
            double *absorbedParticles = &self.absorbedParticles[0,0]   
            unsigned int absorbedMacroParticleCount = self.absorbedMacroParticleCount    
            unsigned int nCoords = self.particlesObj.getNCoords()       
            unsigned int ii
            double lxHalfSqInv = 4./(self.lx*self.lx), lyHalfSqInv = 4./(self.ly*self.ly)
            double lxHalfP4Inv = lxHalfSqInv*lxHalfSqInv, lyHalfP4Inv = lyHalfSqInv*lyHalfSqInv
            double *normalVectors
            double normInv
                         
        if absorbedMacroParticleCount > self.normalVectors.shape[0]:
            self.normalVectors = numpy.empty((<unsigned int> (absorbedMacroParticleCount*1.1),2), dtype=numpy.double)
             
        normalVectors = &self.normalVectors[0,0]    
         
        for ii in range(absorbedMacroParticleCount):
            normInv = 1./sqrt(absorbedParticles[nCoords*ii]**2*lxHalfP4Inv + absorbedParticles[nCoords*ii+1]**2*lyHalfP4Inv)
            normalVectors[2*ii] = -absorbedParticles[nCoords*ii]*lxHalfSqInv*normInv
            normalVectors[2*ii+1] = -absorbedParticles[nCoords*ii+1]*lyHalfSqInv*normInv
 



cdef class AbsorbArbitraryCutCell(ParticleBoundary):
 
    cpdef unsigned short isInside(AbsorbArbitraryCutCell self, double x, double y):
        cdef:
            unsigned short *insideFaces = &self.gridObj.getInsideFaces()[0]
            double *boundaryPoints = &self.gridObj.getBoundaryPoints()[0,0]
            int *cutCellPointsInd = &self.gridObj.getCutCellPointsInd()[0,0]
            double lxHalf = 0.5*self.gridObj.getLxExt(), lyHalf = 0.5*self.gridObj.getLyExt()
            double dxi = 1./self.gridObj.getDx(), dyi = 1./self.gridObj.getDy() 
            unsigned int nx = self.gridObj.getNxExt()        
            unsigned int ii, indx, indy
            unsigned int inCell


        indx = <unsigned int> ( (x+lxHalf)*dxi )
        indy = <unsigned int> ( (y+lyHalf)*dyi )
        inCell = indx + indy*nx

        if insideFaces[inCell]==8:
            return 1
        elif insideFaces[inCell]==0:
            return 0
        # Point-in-polygon check with just the two cut-cell points.
        else:
            if ( (boundaryPoints[2*cutCellPointsInd[2*inCell]]-x) * (boundaryPoints[2*cutCellPointsInd[2*inCell+1]+1]-y) >
                 (boundaryPoints[2*cutCellPointsInd[2*inCell+1]]-x) * (boundaryPoints[2*cutCellPointsInd[2*inCell]+1]-y) ):
                return 1
            else:
                return 0   
 
                 
    cpdef object indexInside(AbsorbArbitraryCutCell self):
  
        cdef:
            double *particleData = &self.particlesObj.getParticleData()[0,0]
            unsigned short *isInside = &self.particlesObj.getIsInside()[0]
            unsigned short *insideFaces = &self.gridObj.getInsideFaces()[0]
            double *boundaryPoints = &self.gridObj.getBoundaryPoints()[0,0]
            int *cutCellPointsInd = &self.gridObj.getCutCellPointsInd()[0,0]
            unsigned int macroParticleCount = self.particlesObj.getMacroParticleCount()
            unsigned int nCoords = self.particlesObj.getNCoords()   
            double lxHalf = 0.5*self.gridObj.getLxExt(), lyHalf = 0.5*self.gridObj.getLyExt()
            double dxi = 1./self.gridObj.getDx(), dyi = 1./self.gridObj.getDy() 
            unsigned int nx = self.gridObj.getNxExt()        
            unsigned int ii, indx, indy
            unsigned int inCell

        # First clip particles to grid. This is a fail safe and should be avoided by small enough time steps!
        _clipParticleDataToGrid(particleData, macroParticleCount, nCoords, 
                                lxHalf - 1.e-6*self.gridObj.getDx(), lyHalf - 1.e-6*self.gridObj.getDy())
                                
        for ii in range(macroParticleCount):
            indx = <unsigned int> ( (particleData[nCoords*ii]+lxHalf)*dxi )
            indy = <unsigned int> ( (particleData[nCoords*ii+1]+lyHalf)*dyi )
            inCell = indx + indy*nx

            if insideFaces[inCell]==8:
                isInside[ii] = 1
            elif insideFaces[inCell]==0:
                isInside[ii] = 0
            # Point-in-polygon check with just the two cut-cell points.
            else:
                if ( (boundaryPoints[2*cutCellPointsInd[2*inCell]]-particleData[nCoords*ii]) * 
                     (boundaryPoints[2*cutCellPointsInd[2*inCell+1]+1]-particleData[nCoords*ii+1]) >
                     (boundaryPoints[2*cutCellPointsInd[2*inCell+1]]-particleData[nCoords*ii]) * 
                     (boundaryPoints[2*cutCellPointsInd[2*inCell]+1]-particleData[nCoords*ii+1]) ):
                    isInside[ii] = 1
                else:
                    isInside[ii] = 0   
    
    # Inspired by http://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect .
    cpdef object calculateInteractionPoint(AbsorbArbitraryCutCell self):       
        cdef:
            double *absorbedParticles = &self.absorbedParticles[0,0]
            double *boundaryPoints = &self.gridObj.getBoundaryPoints()[0,0]
            int *cutCellPointsInd = &self.gridObj.getCutCellPointsInd()[0,0]
            unsigned short *insideFaces = &self.gridObj.getInsideFaces()[0]
            unsigned int nCoords = self.particlesObj.getNCoords()       
            unsigned int nx = self.gridObj.getNxExt()
            double dt = self.particlesObj.getDt()
            double lxHalf = 0.5*self.gridObj.getLxExt(), lyHalf = 0.5*self.gridObj.getLyExt()
            double dxi = 1./self.gridObj.getDx(), dyi = 1./self.gridObj.getDy() 
            unsigned int ii, currentInCell, indx, indy, inCell, stopFlag
            int kk, mm
            double rx, ry, sx, sy, rCrossS, t, u, qmpx, qmpy
            int xDir[2]
            int yDir[2]
            double* remainingTimeStep
            double safetyEps = 1.e-6            # relative safety margin for some calculations
        
        if self.absorbedMacroParticleCount > self.remainingTimeStep.shape[0]:
            self.remainingTimeStep = numpy.empty(<unsigned int> (self.absorbedMacroParticleCount*1.1), dtype=numpy.double)  
        remainingTimeStep = &self.remainingTimeStep[0]             
         
        for ii in range(self.absorbedMacroParticleCount):
            indx = <unsigned int> ( (absorbedParticles[nCoords*ii]+lxHalf)*dxi )
            indy = <unsigned int> ( (absorbedParticles[nCoords*ii+1]+lyHalf)*dyi )
            inCell = indx + indy*nx
            
            rx = -dt*absorbedParticles[nCoords*ii+2]
            ry = -dt*absorbedParticles[nCoords*ii+3]
            
            xDir[0] = 0
            yDir[0] = 0
            if rx<0.:
                xDir[1] = -1
            else:
                xDir[1] = 1
            if ry<0.:
                yDir[1] = -1
            else:
                yDir[1] = 1
            stopFlag = 0
            for kk in xDir:
                for mm in yDir:
#                    print mm, kk, xDir, yDir
                    currentInCell = inCell + kk + mm*nx
                    if insideFaces[currentInCell] != 0 and insideFaces[currentInCell] != 8:
#                        sx = boundaryPoints[2*cutCellPointsInd[2*currentInCell+1]] - \
                        sx = boundaryPoints[2*cutCellPointsInd[2*currentInCell+1]] - \
                             boundaryPoints[2*cutCellPointsInd[2*currentInCell]]
#                        sy = boundaryPoints[2*cutCellPointsInd[2*currentInCell+1]+1] - \
                        sy = boundaryPoints[2*cutCellPointsInd[2*currentInCell+1]+1] - \
                             boundaryPoints[2*cutCellPointsInd[2*currentInCell]+1]
                        rCrossS = rx*sy - ry*sx
                        if rCrossS != 0.:
                            qmpx = boundaryPoints[2*cutCellPointsInd[2*currentInCell]] - absorbedParticles[nCoords*ii]
                            qmpy = boundaryPoints[2*cutCellPointsInd[2*currentInCell]+1] - absorbedParticles[nCoords*ii+1]
                            t = (qmpx*sy - qmpy*sx)/rCrossS
                            u = (qmpx*ry - qmpy*rx)/rCrossS
                            if -safetyEps <= t and t <= 1.+safetyEps and -safetyEps <= u and u <= 1.+safetyEps:
                                absorbedParticles[nCoords*ii] += t*rx*(1.+safetyEps)
                                absorbedParticles[nCoords*ii+1] += t*ry*(1.+safetyEps)
                                remainingTimeStep[ii] = t*(1.+safetyEps)*dt  
                                stopFlag = 1
                                break
                if stopFlag == 1:
                    break
            if stopFlag == 1:
                continue
            # Correct cell not found.
            print 'WARNING: Correct boundary not found. Reduce time step! Particles should not cross more than one cell. ' + \
                  'Location of particle in cell ' + str(inCell) + ' at: New (' + str(absorbedParticles[nCoords*ii]) + \
                  ',' + str(absorbedParticles[nCoords*ii+1]) + ') from old ' + \
                  '(' + str(absorbedParticles[nCoords*ii]+rx) + ',' + str(absorbedParticles[nCoords*ii+1]+ry) + ').'
#            print inCell, rx, ry 
#            print insideFaces[inCell], insideFaces[inCell+1], insideFaces[inCell-1],insideFaces[inCell+nx],insideFaces[inCell-nx], \
#                  insideFaces[inCell+1+nx], insideFaces[inCell-1-nx],insideFaces[inCell+nx-1],insideFaces[inCell-nx+1]
#            print boundaryPoints[2*cutCellPointsInd[2*(inCell+xDir[0]+nx*yDir[0])]], boundaryPoints[2*cutCellPointsInd[2*(inCell+xDir[0]+nx*yDir[0])+1]]
#            print boundaryPoints[2*cutCellPointsInd[2*(inCell+xDir[0]+nx*yDir[0])]+1], boundaryPoints[2*cutCellPointsInd[2*(inCell+xDir[0]+nx*yDir[0])+1]+1]
#            print boundaryPoints[2*cutCellPointsInd[2*(inCell+xDir[1]+nx*yDir[0])]], boundaryPoints[2*cutCellPointsInd[2*(inCell+xDir[1]+nx*yDir[0])+1]]
#            print boundaryPoints[2*cutCellPointsInd[2*(inCell+xDir[1]+nx*yDir[0])]+1], boundaryPoints[2*cutCellPointsInd[2*(inCell+xDir[1]+nx*yDir[0])+1]+1]
#            print boundaryPoints[2*cutCellPointsInd[2*(inCell+xDir[0]+nx*yDir[1])]], boundaryPoints[2*cutCellPointsInd[2*(inCell+xDir[0]+nx*yDir[1])+1]]
#            print boundaryPoints[2*cutCellPointsInd[2*(inCell+xDir[0]+nx*yDir[1])]+1], boundaryPoints[2*cutCellPointsInd[2*(inCell+xDir[0]+nx*yDir[1])+1]+1]
#            print boundaryPoints[2*cutCellPointsInd[2*(inCell+xDir[1]+nx*yDir[1])]], boundaryPoints[2*cutCellPointsInd[2*(inCell+xDir[1]+nx*yDir[1])+1]]
#            print boundaryPoints[2*cutCellPointsInd[2*(inCell+xDir[1]+nx*yDir[1])]+1], boundaryPoints[2*cutCellPointsInd[2*(inCell+xDir[1]+nx*yDir[1])+1]+1]
            
 
    cpdef object calculateNormalVectors(AbsorbArbitraryCutCell self):       
        cdef:
            double *absorbedParticles = &self.absorbedParticles[0,0]
            unsigned int nx = self.gridObj.getNxExt()
            double* cutCellNormalVectors = &self.gridObj.getCutCellNormalVectors()[0,0]
            double lxHalf = 0.5*self.gridObj.getLxExt(), lyHalf = 0.5*self.gridObj.getLyExt()
            double dxi = 1./self.gridObj.getDx(), dyi = 1./self.gridObj.getDy() 
            unsigned int nCoords = self.particlesObj.getNCoords()       
            double* normalVectors
            unsigned int indx, indy, inCell, ii
                   
        if self.absorbedMacroParticleCount > self.normalVectors.shape[0]:
            self.normalVectors = numpy.empty((<unsigned int> (self.absorbedMacroParticleCount*1.1),2), dtype=numpy.double)             
        normalVectors = &self.normalVectors[0,0]  
        
        for ii in range(self.absorbedMacroParticleCount):
            indx = <unsigned int> ( (absorbedParticles[nCoords*ii]+lxHalf)*dxi )
            indy = <unsigned int> ( (absorbedParticles[nCoords*ii+1]+lyHalf)*dyi )
            inCell = indx + indy*nx
            
            normalVectors[2*ii] = cutCellNormalVectors[2*inCell]
            normalVectors[2*ii+1] = cutCellNormalVectors[2*inCell+1]
            
            
            
cdef void _clipParticleDataToGrid(double *particleData, unsigned int macroParticleCount, 
                                  unsigned int nCoords, double lxHalf, double lyHalf) nogil:

    cdef:
        unsigned int ii
        unsigned short clipFlag = 0
    
    for ii in range(macroParticleCount):
        if particleData[nCoords*ii] > lxHalf:
            particleData[nCoords*ii] = lxHalf
            clipFlag = 1
        elif particleData[nCoords*ii] < -lxHalf:
            particleData[nCoords*ii] = -lxHalf
            clipFlag = 1
        if particleData[nCoords*ii+1] > lyHalf:
            particleData[nCoords*ii+1] = lyHalf
            clipFlag = 1
        elif particleData[nCoords*ii+1] < -lyHalf:
            particleData[nCoords*ii+1] = -lyHalf
            clipFlag = 1
    
    with gil:
        if clipFlag == 1:
            print 'WARNING: Some particles have been outside of the grid. This should not happen if time step is smal enough.'


        
