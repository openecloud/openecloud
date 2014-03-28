#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy
cimport numpy
cimport cython
from constants cimport *
cimport randomGen
cimport grid


cdef class Particles:
        
    def __init__(self, particleSpecies, particleType):
        
        self.macroParticleCount = 0 
        
        if particleSpecies == 'electrons':
            self.particleMass = m_e
            self.particleCharge = -elementary_charge
        elif particleSpecies == 'protons':
            self.particleMass = m_p
            self.particleCharge = elementary_charge
        else:
            raise NotImplementedError("Not yet implemented.")
        
        if particleType == 'variableWeight':
            self.nCoords = 6            
        else:
            raise NotImplementedError("Not yet implemented.")
        
        self.chargeOnGrid = numpy.empty(1, dtype=numpy.double)
        self.particleData = numpy.empty((0,self.nCoords), dtype=numpy.double)


     
    cpdef double getMeanWeight(Particles self):
        cdef:
            double[:,:] particleData = self.particleData
            double mean = 0
            unsigned int ii, macroParticleCount = self.macroParticleCount          
        for ii in range(macroParticleCount):
            mean+= particleData[ii,5]        
        mean/= macroParticleCount
        return mean

    cpdef object sortByColumn(Particles self, unsigned int sortByColumn):
        _quicksortByColumn(self.particleData[:self.macroParticleCount], 0, self.macroParticleCount-1, self.nCoords, sortByColumn)

    cpdef object calcGridWeights(Particles self, grid.Grid gridObj):
        _calcGridWeights(&self.particleData[0,0], &self.inCell[0], &self.weightLowXLowY[0], &self.weightUpXLowY[0],
                         &self.weightLowXUpY[0], &self.weightUpXUpY[0],
                         1./gridObj.getDx(), 1./gridObj.getDy(), gridObj.getLxExt()*0.5, gridObj.getLyExt()*0.5,
                         gridObj.getNxExt(), self.nCoords, self.macroParticleCount)

    cpdef object eFieldToParticles(Particles self, grid.Grid gridObj, double[:] eAtGridPoints):
        _fieldToParticles(&eAtGridPoints[0], &self.eAtParticles[0,0], &self.inCell[0], 
                          &self.weightLowXLowY[0], &self.weightUpXLowY[0],
                          &self.weightLowXUpY[0], &self.weightUpXUpY[0], gridObj.getNxExt(), 
                          gridObj.getNpExt(), self.macroParticleCount)
    
    cpdef object bFieldToParticles(Particles self, grid.Grid gridObj, double[:] bAtGridPoints):
        _fieldToParticles3Dims(&bAtGridPoints[0], &self.bAtParticles[0,0], &self.inCell[0], 
                               &self.weightLowXLowY[0], &self.weightUpXLowY[0],
                               &self.weightLowXUpY[0], &self.weightUpXUpY[0], 
                               gridObj.getNxExt(), gridObj.getNpExt(), self.macroParticleCount)
 
    cpdef object chargeToGrid(Particles self, grid.Grid gridObj):
        cdef:
            unsigned int np = gridObj.getNpExt()
        if self.chargeOnGrid.shape[0] < np: 
            self.chargeOnGrid = numpy.empty(np, dtype=numpy.double)
        _chargeToGrid(&self.particleData[0,0], &self.inCell[0], &self.weightLowXLowY[0], &self.weightUpXLowY[0],
                      &self.weightLowXUpY[0], &self.weightUpXUpY[0], &self.chargeOnGrid[0], self.particleCharge, 
                      gridObj.getNxExt(), gridObj.getNpExt(), self.nCoords, self.macroParticleCount)

    cpdef object borisPush(Particles self, double dt, unsigned short typeBField = 2): 
        self.dt = dt    
        # No magnetic field.
        if typeBField==0:
            _borisPushNoB(&self.particleData[0,0], &self.eAtParticles[0,0], dt, 
                          self.particleCharge/self.particleMass*dt, self.macroParticleCount, self.nCoords)
        # Constant (dipole/in space) magnetic field.
        elif typeBField==1:
            _borisPushConstB(&self.particleData[0,0], &self.eAtParticles[0,0], &self.bConst[0], dt, 
                             0.5*self.particleCharge/self.particleMass*dt, self.macroParticleCount, self.nCoords)
        # General magnetic field.
        else:
            _borisPush(&self.particleData[0,0], &self.eAtParticles[0,0], &self.bAtParticles[0,0], dt, 
                       0.5*self.particleCharge/self.particleMass*dt, self.macroParticleCount, self.nCoords)

    cpdef object addAndRemoveParticles(Particles self, double[:,:] addParticleData, unsigned short[:] keepParticles):
        cdef:
            unsigned int addParticleCount = addParticleData.shape[0]
            unsigned int keepParticleCount = numpy.sum(keepParticles)
            unsigned int newParticleCount = keepParticleCount + addParticleCount
            unsigned int oldParticleDataLen = self.particleData.shape[0]
            double[:,:] overwriteParticleData
        # Nothing to do here.
        if addParticleCount==0 and keepParticleCount==self.macroParticleCount:
            pass
        # Only remove particles.
        elif addParticleCount==0:
            _removeParticles(&self.particleData[0,0], &keepParticles[0],
                             self.macroParticleCount, self.nCoords, newParticleCount)
        # Add and remove particles. Memory for particleData might have to be increased.
        else:
            if newParticleCount>oldParticleDataLen:
                overwriteParticleData = numpy.empty((<unsigned int> (newParticleCount*1.1),self.nCoords),
                                                    dtype=numpy.double)      
                _addAndRemoveParticlesNewArray(&self.particleData[0,0], &addParticleData[0,0], &keepParticles[0], 
                                               addParticleCount, keepParticleCount, self.macroParticleCount, 
                                               self.nCoords, &overwriteParticleData[0,0])
                self.setParticleData(overwriteParticleData)          
            else:
                _addAndRemoveParticles(&self.particleData[0,0], &addParticleData[0,0], &keepParticles[0], 
                                       keepParticleCount, addParticleCount, self.macroParticleCount,
                                       self.nCoords)                                        
        self.macroParticleCount = newParticleCount       
        
    cpdef object setBConst(Particles self, double[:] bConst):
        self.bConst = bConst
               
    cpdef object setParticleData(Particles self, double[:,:] particleData):
            
        if particleData.is_c_contig():
            self.particleData = particleData
        else:
            self.particleData = particleData.copy()
        self.macroParticleCount = particleData.shape[0]
        self.inCell = numpy.empty(self.macroParticleCount, dtype=numpy.uintc)
        self.weightLowXLowY = numpy.empty(self.macroParticleCount, dtype=numpy.double)
        self.weightUpXLowY = numpy.empty(self.macroParticleCount, dtype=numpy.double)
        self.weightLowXUpY = numpy.empty(self.macroParticleCount, dtype=numpy.double)
        self.weightUpXUpY = numpy.empty(self.macroParticleCount, dtype=numpy.double)
        self.eAtParticles = numpy.empty((self.macroParticleCount,2), dtype=numpy.double)
        self.bAtParticles = numpy.empty((self.macroParticleCount,3), dtype=numpy.double) 
        self.isInside = numpy.empty(self.macroParticleCount, dtype=numpy.ushort) 

    cpdef double[:,:] getParticleData(Particles self):
        return numpy.asarray(self.particleData[:self.macroParticleCount])
    
    cpdef double[:,:] getFullParticleData(Particles self):
        return self.particleData
    
    cpdef unsigned int getNCoords(Particles self):
        return self.nCoords   
    
    cpdef unsigned int getMacroParticleCount(Particles self):
        return self.macroParticleCount
    
    cpdef object setMacroParticleCount(Particles self, unsigned int macroParticleCount):
        self.macroParticleCount = macroParticleCount
    
    cpdef double[:] getChargeOnGrid(Particles self):
        return numpy.asarray(self.chargeOnGrid)
    
    cpdef object setChargeOnGrid(Particles self, double[:] chargeOnGrid):
        self.chargeOnGrid = chargeOnGrid
    
    cpdef double getParticleCharge(Particles self):
        return self.particleCharge
    
    cpdef double getParticleMass(Particles self):
        return self.particleMass
    
    cpdef unsigned int[:] getInCell(Particles self):
        return self.inCell[:self.macroParticleCount]
    
    cpdef unsigned short[:] getFullIsInside(Particles self):
        return self.isInside
    
    cpdef unsigned short[:] getIsInside(Particles self):
        return self.isInside[:self.macroParticleCount]
    
    cpdef double[:] getWeightLowXUpY(Particles self):
        return self.weightLowXUpY[:self.macroParticleCount]
    
    cpdef double[:] getWeightLowXLowY(Particles self):
        return self.weightLowXLowY[:self.macroParticleCount]
    
    cpdef double[:] getWeightUpXUpY(Particles self):
        return self.weightUpXUpY[:self.macroParticleCount]
    
    cpdef double[:] getWeightUpXLowY(Particles self):
        return self.weightUpXLowY[:self.macroParticleCount]
    
    cpdef double[:,:] getEAtParticles(Particles self):
        return self.eAtParticles[:self.macroParticleCount]
    
    cpdef double[:,:] getBAtParticles(Particles self):
        return self.bAtParticles
        
    cpdef double getDt(Particles self):
        return self.dt
                               

        
cdef void _addAndRemoveParticlesNewArray(double* fullParticleData, double* addParticleData, unsigned short* keepParticles,
                                         unsigned int addParticleCount, unsigned int keepParticleCount, 
                                         unsigned int oldParticleCount, unsigned int nCoords, 
                                         double* overwriteParticleData) nogil:
    cdef: 
        unsigned int ii = 0, jj = 0, ll = 0, kk, ind1, ind2
    while jj < keepParticleCount:
        if keepParticles[ii] == 1:
            ind1 = nCoords*jj
            ind2 = nCoords*ii
            for kk in range(nCoords):
                overwriteParticleData[ind1+kk] = fullParticleData[ind2+kk]
            jj += 1 
        ii += 1   
    for ii in range(addParticleCount):
        ind1 = nCoords*(keepParticleCount + ii)
        ind2 = nCoords*ii
        for kk in range(nCoords):
            overwriteParticleData[ind1+kk] = addParticleData[ind2+kk]    
        
                                       
cdef void _addAndRemoveParticles(double* fullParticleData, double* addParticleData, unsigned short* keepParticles, 
                                 unsigned int keepParticleCount, unsigned int addParticleCount, 
                                 unsigned int oldParticleCount, unsigned int nCoords) nogil:
    cdef: 
        unsigned int ii = 0, jj = 0, ll = 0, kk, ind1, ind2
    while ii < addParticleCount and jj < oldParticleCount:
        if  keepParticles[jj] == 0:
            ind1 = nCoords*jj
            ind2 = nCoords*ii
            for kk in range(nCoords):
                fullParticleData[ind1+kk] = addParticleData[ind2+kk]
            ii += 1
        jj += 1
    if ii == addParticleCount:
        ii = jj - addParticleCount
        while ii < keepParticleCount:
            if keepParticles[jj] == 1:
                ind1 = nCoords*(ii+addParticleCount)
                ind2 = nCoords*jj
                for kk in range(nCoords):
                    fullParticleData[ind1+kk] = fullParticleData[ind2+kk]
                ii += 1          
            jj += 1         
    elif jj == oldParticleCount:
        while ii < addParticleCount:  
            ind1 = nCoords*jj
            ind2 = nCoords*ii
            for kk in range(nCoords):
                fullParticleData[ind1+kk] = addParticleData[ind2+kk]
            ii += 1
            jj += 1
            
cdef void _removeParticles(double* fullParticleData, unsigned short* keepParticles, unsigned int oldParticleCount, 
                           unsigned int nCoords, unsigned int newParticleCount) nogil:         
        cdef:
            unsigned int ll = 0, jj = 0, kk, ind1, ind2
        ll += oldParticleCount
        while jj < newParticleCount:
            if keepParticles[jj] == 0:
                while True:
                    ll -= 1
                    if keepParticles[ll] == 1:
                        break                 
                ind1 = nCoords*jj
                ind2 = nCoords*ll
                for kk in range(nCoords):
                    fullParticleData[ind1+kk] = fullParticleData[ind2+kk]
            jj += 1
                      
# General non-relativistic boris push for arbitrary magnetic field.
cdef void _borisPush(double* particleData, double* eAtParticles, double* bAtParticles, double dt, 
                     double chargeDtOverMassHalf, unsigned int macroParticleCount, unsigned int nCoords) nogil:
    cdef: 
        unsigned int ii     
        double tSqSumPlOneInvTimTwo, ts0, ts1, ts2, vs0, vs1, vs2  
        
    for ii in range(macroParticleCount):
        particleData[nCoords*ii+2] += chargeDtOverMassHalf*eAtParticles[2*ii]
        particleData[nCoords*ii+3] += chargeDtOverMassHalf*eAtParticles[2*ii+1]        
        ts0 = chargeDtOverMassHalf*bAtParticles[ii*3] 
        ts1 = chargeDtOverMassHalf*bAtParticles[ii*3+1] 
        ts2 = chargeDtOverMassHalf*bAtParticles[ii*3+2] 
        vs0 = particleData[nCoords*ii+2] + particleData[nCoords*ii+3]*ts2 - particleData[nCoords*ii+4]*ts1
        vs1 = particleData[nCoords*ii+3] + particleData[nCoords*ii+4]*ts0 - particleData[nCoords*ii+2]*ts2
        vs2 = particleData[nCoords*ii+4] + particleData[nCoords*ii+2]*ts1 - particleData[nCoords*ii+3]*ts0
        tSqSumPlOneInvTimTwo = 2./(1. + ts0*ts0 + ts1*ts1 + ts2*ts2)
        ts0 = ts0*tSqSumPlOneInvTimTwo    
        ts1 = ts1*tSqSumPlOneInvTimTwo    
        ts2 = ts2*tSqSumPlOneInvTimTwo
        particleData[nCoords*ii+2] += vs1*ts2 - vs2*ts1 + chargeDtOverMassHalf*eAtParticles[2*ii]
        particleData[nCoords*ii+3] += vs2*ts0 - vs0*ts2 + chargeDtOverMassHalf*eAtParticles[2*ii+1]
        particleData[nCoords*ii+4] += vs0*ts1 - vs1*ts0     
        particleData[nCoords*ii] += dt*particleData[nCoords*ii+2]
        particleData[nCoords*ii+1] += dt*particleData[nCoords*ii+3]
 
# Non-relativistic boris push with constant (dipole/in space) magnetic field.     
cdef void _borisPushConstB(double* particleData, double* eAtParticles, double* bAtParticles, double dt, 
                           double chargeDtOverMassHalf, unsigned int macroParticleCount, unsigned int nCoords) nogil:
    cdef: 
        unsigned int ii     
        double tSqSumPlOneInvTimTwo, s0, s1, s2, vs0, vs1, vs2
        double t0 = chargeDtOverMassHalf*bAtParticles[0] 
        double t1 = chargeDtOverMassHalf*bAtParticles[1] 
        double t2 = chargeDtOverMassHalf*bAtParticles[2]    
         
    for ii in range(macroParticleCount):
        particleData[nCoords*ii+2] += chargeDtOverMassHalf*eAtParticles[2*ii]
        particleData[nCoords*ii+3] += chargeDtOverMassHalf*eAtParticles[2*ii+1]
        vs0 = particleData[nCoords*ii+2] + particleData[nCoords*ii+3]*t2 - particleData[nCoords*ii+4]*t1
        vs1 = particleData[nCoords*ii+3] + particleData[nCoords*ii+4]*t0 - particleData[nCoords*ii+2]*t2
        vs2 = particleData[nCoords*ii+4] + particleData[nCoords*ii+2]*t1 - particleData[nCoords*ii+3]*t0
        tSqSumPlOneInvTimTwo = 2./(1. + t0*t0 + t1*t1 + t2*t2)
        s0 = t0*tSqSumPlOneInvTimTwo    
        s1 = t1*tSqSumPlOneInvTimTwo    
        s2 = t2*tSqSumPlOneInvTimTwo
        particleData[nCoords*ii+2] += vs1*s2 - vs2*s1 + chargeDtOverMassHalf*eAtParticles[2*ii]
        particleData[nCoords*ii+3] += vs2*s0 - vs0*s2 + chargeDtOverMassHalf*eAtParticles[2*ii+1]
        particleData[nCoords*ii+4] += vs0*s1 - vs1*s0     
        particleData[nCoords*ii] += dt*particleData[nCoords*ii+2]
        particleData[nCoords*ii+1] += dt*particleData[nCoords*ii+3]
                       
# Non-relativistic boris push without magnetic field.        
cdef void _borisPushNoB(double* particleData, double* eAtParticles, double dt, double chargeDtOverMass,
                        unsigned int macroParticleCount, unsigned int nCoords) nogil:
    cdef: 
        unsigned int ii
    for ii in range(macroParticleCount):
        particleData[nCoords*ii+2] += chargeDtOverMass*eAtParticles[2*ii]
        particleData[nCoords*ii+3] += chargeDtOverMass*eAtParticles[2*ii+1]
        
        particleData[nCoords*ii] += dt*particleData[nCoords*ii+2]
        particleData[nCoords*ii+1] += dt*particleData[nCoords*ii+3]

      
cdef void _chargeToGrid(double* particleData, unsigned int* inCell, double* weightLowXLowY, double* weightUpXLowY,
                        double* weightLowXUpY, double* weightUpXUpY, double* chargeOnGrid, double particleCharge,
                        unsigned int nx, unsigned int np, unsigned int nCoords, unsigned int macroParticleCount) nogil:     
    cdef: 
        unsigned int ii, ind
    for ii in range(np):
        chargeOnGrid[ii] = 0.    
    for ii in range(macroParticleCount):
        ind = inCell[ii]       
        chargeOnGrid[ind] += weightLowXLowY[ii]*particleData[nCoords*ii+5]*particleCharge   
        ind = ind + 1
        chargeOnGrid[ind] += weightUpXLowY[ii]*particleData[nCoords*ii+5]*particleCharge
        ind = ind - 1 + nx
        chargeOnGrid[ind] += weightLowXUpY[ii]*particleData[nCoords*ii+5]*particleCharge
        ind = ind + 1
        chargeOnGrid[ind] += weightUpXUpY[ii]*particleData[nCoords*ii+5]*particleCharge

    
cdef void _fieldToParticles(double* fieldAtGridPoints, double* fieldAtParticles, unsigned int* inCell, 
                            double* weightLowXLowY, double* weightUpXLowY,
                            double* weightLowXUpY, double* weightUpXUpY, unsigned int nx, 
                            unsigned int np, unsigned int macroParticleCount) nogil:                  
    cdef: 
        unsigned int ind, ii    
    for ii in range(macroParticleCount):
        ind = inCell[ii]
        fieldAtParticles[2*ii] =  weightLowXLowY[ii]*fieldAtGridPoints[ind]         # First one has to set (=, not +=),  
        fieldAtParticles[2*ii+1] =  weightLowXLowY[ii]*fieldAtGridPoints[ind+np]    # so values are overwritten.
        ind = ind + 1
        fieldAtParticles[2*ii] += weightUpXLowY[ii]*fieldAtGridPoints[ind]
        fieldAtParticles[2*ii+1] += weightUpXLowY[ii]*fieldAtGridPoints[ind+np]
        ind = ind - 1 + nx
        fieldAtParticles[2*ii] += weightLowXUpY[ii]*fieldAtGridPoints[ind]
        fieldAtParticles[2*ii+1] += weightLowXUpY[ii]*fieldAtGridPoints[ind+np]
        ind = ind + 1
        fieldAtParticles[2*ii] += weightUpXUpY[ii]*fieldAtGridPoints[ind]
        fieldAtParticles[2*ii+1] += weightUpXUpY[ii]*fieldAtGridPoints[ind+np]


cdef void _fieldToParticles3Dims(double* fieldAtGridPoints, double* fieldAtParticles, unsigned int* inCell, 
                                 double* weightLowXLowY, double* weightUpXLowY,
                                 double* weightLowXUpY, double* weightUpXUpY, unsigned int nx, 
                                 unsigned int np, unsigned int macroParticleCount) nogil:                  
    cdef: 
        unsigned int ind, ii    
    for ii in range(macroParticleCount):
        ind = inCell[ii]
        fieldAtParticles[3*ii] = weightLowXLowY[ii]*fieldAtGridPoints[ind]          # First one has to set (=, not +=),  
        fieldAtParticles[3*ii+1] = weightLowXLowY[ii]*fieldAtGridPoints[ind+np]     # so values are overwritten.
        fieldAtParticles[3*ii+2] = weightLowXLowY[ii]*fieldAtGridPoints[ind+2*np]
        ind = ind + 1
        fieldAtParticles[3*ii] += weightUpXLowY[ii]*fieldAtGridPoints[ind]
        fieldAtParticles[3*ii+1] += weightUpXLowY[ii]*fieldAtGridPoints[ind+np]
        fieldAtParticles[3*ii+2] += weightUpXLowY[ii]*fieldAtGridPoints[ind+2*np]
        ind = ind - 1 + nx
        fieldAtParticles[3*ii] += weightLowXUpY[ii]*fieldAtGridPoints[ind]
        fieldAtParticles[3*ii+1] += weightLowXUpY[ii]*fieldAtGridPoints[ind+np]
        fieldAtParticles[3*ii+2] += weightLowXUpY[ii]*fieldAtGridPoints[ind+2*np]
        ind = ind + 1
        fieldAtParticles[3*ii] += weightUpXUpY[ii]*fieldAtGridPoints[ind]
        fieldAtParticles[3*ii+1] += weightUpXUpY[ii]*fieldAtGridPoints[ind+np]
        fieldAtParticles[3*ii+2] += weightUpXUpY[ii]*fieldAtGridPoints[ind+2*np]
        
        
cdef void _calcGridWeights(double* particleData, unsigned int* inCell, double* weightLowXLowY, double* weightUpXLowY,
                           double* weightLowXUpY, double* weightUpXUpY, double dxi, double dyi, double lxHalf, double lyHalf, 
                           unsigned int nx, unsigned int nCoords, unsigned int macroParticleCount) nogil:
    cdef: 
        double inCellCoordsNormX, inCellCoordsNormY
        unsigned int indx, indy
        unsigned int ii    
    for ii in range(macroParticleCount):
        inCellCoordsNormX = (particleData[nCoords*ii]+lxHalf)*dxi
        inCellCoordsNormY = (particleData[nCoords*ii+1]+lyHalf)*dyi
        indx = <unsigned int> inCellCoordsNormX
        indy = <unsigned int> inCellCoordsNormY
        inCell[ii] = indx + indy*nx
        inCellCoordsNormX = inCellCoordsNormX - indx
        inCellCoordsNormY = inCellCoordsNormY - indy
        weightLowXLowY[ii] = (1-inCellCoordsNormX)*(1-inCellCoordsNormY)
        weightUpXLowY[ii] = inCellCoordsNormX*(1-inCellCoordsNormY)
        weightLowXUpY[ii] = (1-inCellCoordsNormX)*inCellCoordsNormY
        weightUpXUpY[ii] = inCellCoordsNormX*inCellCoordsNormY
        

# Quicksort to sort particle data by a specified column.
# Inspired by http://en.wikipedia.org/wiki/Quicksort.
# This is only faster than using e.g. numpy because
# "sort whole matrix by one column" isn't implemented
# in numpy (and one has to use indirect sorting). 
# Otherwise I can't compete with the speed of
# other implementations, but I really can't tell why.
# Sorts in-place, so small memory overhead.
# Random pivot element prevents worst-case runtimes.
# Catches elements equal to pivot element. 
cdef void _quicksortByColumn(double[:,:] array, int left, int right,
                             unsigned int nColumns, unsigned int sortByColumn) nogil:
    cdef:    
        double pivotValue, temp
        unsigned int pivotIndex, pivotNewIndexLeft, pivotNewIndexRight = left
        unsigned int ii, jj   
    if left<right:         
        pivotIndex = left + randomGen.randi(right-left)
        pivotValue = array[pivotIndex,sortByColumn]
        for jj in range(nColumns):
            temp = array[pivotIndex,jj]
            array[pivotIndex,jj] = array[right,jj]
            array[right,jj] = temp
        for ii in range(left, right):
            if array[ii,sortByColumn]<=pivotValue:
                for jj in range(nColumns):
                    temp = array[ii,jj]
                    array[ii,jj] = array[pivotNewIndexRight,jj]
                    array[pivotNewIndexRight,jj] = temp
                pivotNewIndexRight += 1
        for jj in range(nColumns):
            temp = array[pivotNewIndexRight,jj]
            array[pivotNewIndexRight,jj] = array[right,jj]
            array[right,jj] = temp
        pivotNewIndexLeft = pivotNewIndexRight
        while pivotNewIndexLeft>left and array[pivotNewIndexLeft-1,sortByColumn]==pivotValue:
            pivotNewIndexLeft-= 1
        _quicksortByColumn(array, left, pivotNewIndexLeft - 1, nColumns, sortByColumn)
        _quicksortByColumn(array, pivotNewIndexRight + 1, right, nColumns, sortByColumn)

 
   


    
    
