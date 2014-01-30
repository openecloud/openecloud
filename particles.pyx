#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy
cimport numpy
cimport cython
from constants cimport *
import time



class Particles:
      
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


    def setBAtParticles(self, bAtParticles):
        self.bAtParticles = bAtParticles
               
    def setParticleData(self, particleData):
            
        if particleData.flags['C_CONTIGUOUS']:
            self.particleData = particleData
        else:
            self.particleData = particleData.copy()
        self.macroParticleCount = particleData.shape[0]
        self.inCell = numpy.empty(self.macroParticleCount, dtype=numpy.uint32)
        self.weightLowXLowY = numpy.empty(self.macroParticleCount, dtype=numpy.double)
        self.weightUpXLowY = numpy.empty(self.macroParticleCount, dtype=numpy.double)
        self.weightLowXUpY = numpy.empty(self.macroParticleCount, dtype=numpy.double)
        self.weightUpXUpY = numpy.empty(self.macroParticleCount, dtype=numpy.double)
        self.eAtParticles = numpy.empty((self.macroParticleCount,2), dtype=numpy.double) 
        self.isInside = numpy.empty(self.macroParticleCount, dtype=numpy.uint16) 

    def getParticleData(self):
        return self.particleData[:self.macroParticleCount]
    
    def getFullParticleData(self):
        return self.particleData
    
    def getNCoords(self):
        return self.nCoords   
    
    def getMacroParticleCount(self):
        return self.macroParticleCount
    
    def setMacroParticleCount(self, macroParticleCount):
        self.macroParticleCount = macroParticleCount
    
    def getChargeOnGrid(self):
        return self.chargeOnGrid
    
    def setChargeOnGrid(self, chargeOnGrid):
        self.chargeOnGrid = chargeOnGrid
    
    def getParticleCharge(self):
        return self.particleCharge
    
    def getParticleMass(self):
        return self.particleMass
    
    def getInCell(self):
        return self.inCell[:self.macroParticleCount]
    
    def getFullIsInside(self):
        return self.isInside
    
    def getIsInside(self):
        return self.isInside[:self.macroParticleCount]
    
#     def setIsInside(self, isInside):
#         self.isInside = isInside
    
    def getWeightLowXUpY(self):
        return self.weightLowXUpY[:self.macroParticleCount]
    
    def getWeightLowXLowY(self):
        return self.weightLowXLowY[:self.macroParticleCount]
    
    def getWeightUpXUpY(self):
        return self.weightUpXUpY[:self.macroParticleCount]
    
    def getWeightUpXLowY(self):
        return self.weightUpXLowY[:self.macroParticleCount]
    
    def getEAtParticles(self):
        return self.eAtParticles[:self.macroParticleCount]
    
    def getBAtParticles(self):
        return self.bAtParticles


cpdef calcGridWeights(object particlesObj, object gridObj):  

    # Variable declarations. Some repeatedly used numbers.
    cdef: 
        double dxi = 1./gridObj.getDx()
        double dyi = 1./gridObj.getDy()
        double lxHalf = gridObj.getLx()*0.5
        double lyHalf = gridObj.getLy()*0.5
        unsigned int nx = gridObj.getNx()           
        unsigned int nCoords = particlesObj.getNCoords()
        unsigned int macroParticleCount = particlesObj.getMacroParticleCount()
        double inCellCoordsNormX, inCellCoordsNormY
        unsigned int indx, indy
        unsigned int ii

    # Numpy and pointers to the numpy arrays.
    cdef: 
        numpy.ndarray[numpy.uint32_t] inCellNumpy = particlesObj.getInCell()
        unsigned int* inCell = &inCellNumpy[0]
        numpy.ndarray[numpy.double_t, ndim = 2] particleDataNumpy = particlesObj.getParticleData()
        double* particleData = &particleDataNumpy[0,0]
        numpy.ndarray[numpy.double_t] weightLowXLowYNumpy = particlesObj.getWeightLowXLowY()
        double* weightLowXLowY = &weightLowXLowYNumpy[0]
        numpy.ndarray[numpy.double_t] weightUpXLowYNumpy = particlesObj.getWeightUpXLowY()
        double* weightUpXLowY = &weightUpXLowYNumpy[0]
        numpy.ndarray[numpy.double_t] weightLowXUpYNumpy = particlesObj.getWeightLowXUpY()
        double* weightLowXUpY = &weightLowXUpYNumpy[0]
        numpy.ndarray[numpy.double_t] weightUpXUpYNumpy = particlesObj.getWeightUpXUpY()
        double* weightUpXUpY = &weightUpXUpYNumpy[0]
    
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
        
cpdef eFieldToParticles(particlesObj, gridObj, numpy.ndarray[numpy.double_t] eAtGridPointsNumpy):   
    
    # Variable declarations.
    cdef: 
        unsigned int nx = gridObj.getNx(), np = gridObj.getNp()
        unsigned int macroParticleCount = particlesObj.getMacroParticleCount()      
        unsigned int ind
        unsigned int ii

    # Numpy and pointers to the numpy arrays.
    cdef: 
        numpy.ndarray[numpy.uint32_t] inCellNumpy = particlesObj.getInCell()
        unsigned int* inCell = &inCellNumpy[0]                
        numpy.ndarray[numpy.double_t] weightLowXLowYNumpy = particlesObj.getWeightLowXLowY()
        double *weightLowXLowY = &weightLowXLowYNumpy[0]
        numpy.ndarray[numpy.double_t] weightUpXLowYNumpy = particlesObj.getWeightUpXLowY()
        double *weightUpXLowY = &weightUpXLowYNumpy[0]
        numpy.ndarray[numpy.double_t] weightLowXUpYNumpy = particlesObj.getWeightLowXUpY()
        double *weightLowXUpY = &weightLowXUpYNumpy[0]
        numpy.ndarray[numpy.double_t] weightUpXUpYNumpy = particlesObj.getWeightUpXUpY()
        double *weightUpXUpY = &weightUpXUpYNumpy[0]
        numpy.ndarray[numpy.double_t, ndim = 2] eAtParticlesNumpy = particlesObj.getEAtParticles()
        double *eAtParticles = &eAtParticlesNumpy[0,0]
        double *eAtGridPoints = &eAtGridPointsNumpy[0]
    
    for ii in range(macroParticleCount):
        ind = inCell[ii]
        eAtParticles[2*ii] =  weightLowXLowY[ii]*eAtGridPoints[ind]         # First one has to set (=, not +=),  
        eAtParticles[2*ii+1] =  weightLowXLowY[ii]*eAtGridPoints[ind+np]    # so values are overwritten.
        ind += 1
        eAtParticles[2*ii] += weightUpXLowY[ii]*eAtGridPoints[ind]
        eAtParticles[2*ii+1] += weightUpXLowY[ii]*eAtGridPoints[ind+np]
        ind += -1 + nx
        eAtParticles[2*ii] += weightLowXUpY[ii]*eAtGridPoints[ind]
        eAtParticles[2*ii+1] += weightLowXUpY[ii]*eAtGridPoints[ind+np]
        ind += 1
        eAtParticles[2*ii] += weightUpXUpY[ii]*eAtGridPoints[ind]
        eAtParticles[2*ii+1] += weightUpXUpY[ii]*eAtGridPoints[ind+np]


cpdef chargeToGrid(particlesObj, gridObj):   
    
    cdef: 
        unsigned int ii
        unsigned int ind       
        unsigned int nx = gridObj.getNx(), np = gridObj.getNp()
        unsigned int nCoords = particlesObj.getNCoords()                                                   
        unsigned int macroParticleCount = particlesObj.getMacroParticleCount()              
        double particleCharge = particlesObj.getParticleCharge()     
        numpy.ndarray[numpy.uint32_t] inCellNumpy = particlesObj.getInCell()
        unsigned int* inCell = &inCellNumpy[0]                
        numpy.ndarray[numpy.double_t] weightLowXLowYNumpy = particlesObj.getWeightLowXLowY()
        double *weightLowXLowY = &weightLowXLowYNumpy[0]
        numpy.ndarray[numpy.double_t] weightUpXLowYNumpy = particlesObj.getWeightUpXLowY()
        double *weightUpXLowY = &weightUpXLowYNumpy[0]
        numpy.ndarray[numpy.double_t] weightLowXUpYNumpy = particlesObj.getWeightLowXUpY()
        double *weightLowXUpY = &weightLowXUpYNumpy[0]
        numpy.ndarray[numpy.double_t] weightUpXUpYNumpy = particlesObj.getWeightUpXUpY()
        double *weightUpXUpY = &weightUpXUpYNumpy[0]                                  
        numpy.ndarray[numpy.double_t, ndim = 2] particleDataNumpy = particlesObj.getParticleData()
        double* particleData = &particleDataNumpy[0,0]         
        numpy.ndarray[numpy.double_t] chargeOnGridNumpy
        double *chargeOnGrid
     
    if particlesObj.getChargeOnGrid().shape[0] < np: 
        particlesObj.setChargeOnGrid(numpy.empty(np, dtype=numpy.double))
    chargeOnGridNumpy = particlesObj.getChargeOnGrid()
    chargeOnGrid = &chargeOnGridNumpy[0]
    for ii in range(np):
        chargeOnGrid[ii] = 0.
    
    for ii in range(macroParticleCount):
        ind = inCell[ii]       
        chargeOnGrid[ind] += weightLowXLowY[ii]*particleData[nCoords*ii+5]*particleCharge   
        ind += 1
        chargeOnGrid[ind] += weightUpXLowY[ii]*particleData[nCoords*ii+5]*particleCharge
        ind += -1 + nx
        chargeOnGrid[ind] += weightLowXUpY[ii]*particleData[nCoords*ii+5]*particleCharge
        ind += 1
        chargeOnGrid[ind] += weightUpXUpY[ii]*particleData[nCoords*ii+5]*particleCharge
    
    particlesObj.setChargeOnGrid(chargeOnGridNumpy)
 

cpdef borisPushNonRelNoB(particlesObj, double dt):

    cdef: 
        unsigned int ii
        unsigned int nCoords = particlesObj.nCoords
        unsigned int macroParticleCount = particlesObj.getMacroParticleCount()                
        double chargeDtOverMass = particlesObj.getParticleCharge()/particlesObj.getParticleMass()*dt
                                   
        numpy.ndarray[numpy.double_t, ndim = 2] particleDataNumpy = particlesObj.getParticleData()
        double* particleData = &particleDataNumpy[0,0]    
        numpy.ndarray[numpy.double_t, ndim = 2] eAtParticlesNumpy = particlesObj.getEAtParticles()
        double* eAtParticles = &eAtParticlesNumpy[0,0]

    for ii in range(macroParticleCount):
        particleData[nCoords*ii+2] += chargeDtOverMass*eAtParticles[2*ii]
        particleData[nCoords*ii+3] += chargeDtOverMass*eAtParticles[2*ii+1]
        
        particleData[nCoords*ii] += dt*particleData[nCoords*ii+2]
        particleData[nCoords*ii+1] += dt*particleData[nCoords*ii+3]

    
cpdef borisPushNonRel(particlesObj, double dt):

    cdef: 
        unsigned int ii  
        unsigned int nCoords = particlesObj.getNCoords()                                                               
        unsigned int macroParticleCount = particlesObj.getMacroParticleCount()                
        double chargeDtOverMassHalf = 0.5*particlesObj.getParticleCharge()/particlesObj.getParticleMass()*dt      
        double tSqSumPlOneInvTimTwo
                                  
        numpy.ndarray[numpy.double_t, ndim = 2] particleDataNumpy = particlesObj.getParticleData()
        double* particleData = &particleDataNumpy[0,0]                                     
        numpy.ndarray[numpy.double_t, ndim = 2] buff01 = particlesObj.getEAtParticles()      
        double* eAtParticles = <double*> buff01.data                                    
        numpy.ndarray[numpy.double_t] buff02 = particlesObj.getBAtParticles()
        double* bAtParticles = <double*> buff02.data
        double ts0,ts1,ts2,vs0,vs1,vs2
    
    for ii in range(macroParticleCount):
        particleData[nCoords*ii+2] += chargeDtOverMassHalf*eAtParticles[2*ii]
        particleData[nCoords*ii+3] += chargeDtOverMassHalf*eAtParticles[2*ii+1]
        ts0 = chargeDtOverMassHalf*bAtParticles[0]
        ts1 = chargeDtOverMassHalf*bAtParticles[1]
        ts2 = chargeDtOverMassHalf*bAtParticles[2]
        vs0 = particleData[nCoords*ii+2] + particleData[nCoords*ii+3]*ts2 - particleData[nCoords*ii+4]*ts1
        vs1 = particleData[nCoords*ii+3] + particleData[nCoords*ii+4]*ts0 - particleData[nCoords*ii+2]*ts2
        vs2 = particleData[nCoords*ii+4] + particleData[nCoords*ii+2]*ts1 - particleData[nCoords*ii+3]*ts0
        tSqSumPlOneInvTimTwo = 2./(1. + ts0*ts0 + ts1*ts1 + ts2*ts2)
        ts0 *= tSqSumPlOneInvTimTwo    
        ts1 *= tSqSumPlOneInvTimTwo    
        ts2 *= tSqSumPlOneInvTimTwo
        particleData[nCoords*ii+2] += vs1*ts2 - vs2*ts1 + chargeDtOverMassHalf*eAtParticles[2*ii]
        particleData[nCoords*ii+3] += vs2*ts0 - vs0*ts2 + chargeDtOverMassHalf*eAtParticles[2*ii+1]
        particleData[nCoords*ii+4] += vs0*ts1 - vs1*ts0
        
        particleData[nCoords*ii] += dt*particleData[nCoords*ii+2]
        particleData[nCoords*ii+1] += dt*particleData[nCoords*ii+3]
#TRACKING IN MULTIPOLES UP TO SEXTUPOLES
cpdef borisPushNonRelMultipole(particlesObj, double dt, double order, double strength):

    cdef: 
        unsigned int ii  
        unsigned int nCoords = particlesObj.getNCoords()                                                               
        unsigned int macroParticleCount = particlesObj.getMacroParticleCount()                
        double chargeDtOverMassHalf = 0.5*particlesObj.getParticleCharge()/particlesObj.getParticleMass()*dt      
        double tSqSumPlOneInvTimTwo
                                  
        numpy.ndarray[numpy.double_t, ndim = 2] particleDataNumpy = particlesObj.getParticleData()
        double* particleData = &particleDataNumpy[0,0]                                     
        numpy.ndarray[numpy.double_t, ndim = 2] buff01 = particlesObj.getEAtParticles()      
        double* eAtParticles = <double*> buff01.data                                    
        numpy.ndarray[numpy.double_t] buff02 = particlesObj.getBAtParticles()
        double* bAtParticles = <double*> buff02.data
        double ts0,ts1,ts2,vs0,vs1,vs2
        double b0atparticle=0.0
        double b1atparticle=0.0
    
    for ii in range(macroParticleCount):
        particleData[nCoords*ii+2] += chargeDtOverMassHalf*eAtParticles[2*ii]
        particleData[nCoords*ii+3] += chargeDtOverMassHalf*eAtParticles[2*ii+1]
        if order==1:
         b1atparticle=strength
        if order==2:
         b0atparticle=-strength*particleData[nCoords*ii+1]
         b1atparticle=-strength*particleData[nCoords*ii]
        if order==3:
         b0atparticle=-6.0*strength*particleData[nCoords*ii+1]*particleData[nCoords*ii]
         b1atparticle=-3.0*strength*(particleData[nCoords*ii]*particleData[nCoords*ii]-particleData[nCoords*ii+1]*particleData[nCoords*ii+1])
        ts0 = chargeDtOverMassHalf*b0atparticle
        ts1 = chargeDtOverMassHalf*b1atparticle
        ts2 = 0.0
        vs0 = particleData[nCoords*ii+2] + particleData[nCoords*ii+3]*ts2 - particleData[nCoords*ii+4]*ts1
        vs1 = particleData[nCoords*ii+3] + particleData[nCoords*ii+4]*ts0 - particleData[nCoords*ii+2]*ts2
        vs2 = particleData[nCoords*ii+4] + particleData[nCoords*ii+2]*ts1 - particleData[nCoords*ii+3]*ts0
        tSqSumPlOneInvTimTwo = 2./(1. + ts0*ts0 + ts1*ts1 + ts2*ts2)
        ts0 *= tSqSumPlOneInvTimTwo    
        ts1 *= tSqSumPlOneInvTimTwo    
        ts2 *= tSqSumPlOneInvTimTwo
        particleData[nCoords*ii+2] += vs1*ts2 - vs2*ts1 + chargeDtOverMassHalf*eAtParticles[2*ii]
        particleData[nCoords*ii+3] += vs2*ts0 - vs0*ts2 + chargeDtOverMassHalf*eAtParticles[2*ii+1]
        particleData[nCoords*ii+4] += vs0*ts1 - vs1*ts0
        
        particleData[nCoords*ii] += dt*particleData[nCoords*ii+2]
        particleData[nCoords*ii+1] += dt*particleData[nCoords*ii+3]

cpdef addAndRemoveParticles(particlesObj, numpy.ndarray[numpy.double_t, ndim = 2] addParticleDataNumpy, numpy.ndarray[numpy.uint16_t] keepParticlesNumpy):

    cdef: 
        unsigned int nCoords = particlesObj.getNCoords()  
        unsigned int keepParticleCount = numpy.sum(keepParticlesNumpy)
        unsigned int addParticleCount = addParticleDataNumpy.shape[0]
        unsigned int newParticleCount = keepParticleCount + addParticleCount
        unsigned int ii = 0, jj = 0, ll = 0, kk, ind1, ind2
        numpy.ndarray[numpy.double_t, ndim = 2] fullParticleDataNumpy = particlesObj.getFullParticleData()      
        double* fullParticleData = &fullParticleDataNumpy[0,0]
        unsigned int oldParticleCount = particlesObj.getMacroParticleCount()
        unsigned int oldParticleDataLen = fullParticleDataNumpy.shape[0]
        numpy.ndarray[numpy.double_t, ndim = 2] overwriteParticleDataNumpy
        double *overwriteParticleData         
        unsigned short *keepParticles = &keepParticlesNumpy[0]
        double *addParticleData
    
    if addParticleCount == 0:
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
                ii += 1
            jj += 1
        particlesObj.setMacroParticleCount(newParticleCount)  
    else:   
        addParticleData = &addParticleDataNumpy[0,0]
        if newParticleCount <= oldParticleDataLen:
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
            particlesObj.setMacroParticleCount(newParticleCount)
    
        elif newParticleCount > oldParticleDataLen:
            overwriteParticleDataNumpy = numpy.empty((newParticleCount*1.05,nCoords), dtype=numpy.double)
            overwriteParticleData = &overwriteParticleDataNumpy[0,0]
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
            particlesObj.setParticleData(overwriteParticleDataNumpy)
            particlesObj.setMacroParticleCount(newParticleCount)
    


# Quicksort to sort particle data by a specified column.
# Inspired by http://en.wikipedia.org/wiki/Qicksort
# Sorts 2D numpy double array by specified column.

cdef void quicksortByColumn(numpy.ndarray[numpy.double_t, ndim=2] numpyArray, unsigned int sortByColumn):

    _quicksortByColumn(&numpyArray[0,0], 0, numpyArray.shape[0]-1, numpyArray.shape[1], sortByColumn)
        
        
cdef void _quicksortByColumn(double* array, int left, int right,
                             unsigned int nColumns, unsigned int sortByColumn):

    cdef:
        int pivotIndex, pivotNewIndex
        
    if left<right:
        pivotIndex = (right+left)/2
        pivotNewIndex = _partitionColumn(array, left, right, pivotIndex, nColumns, sortByColumn)
        _quicksortByColumn(array, left, pivotNewIndex - 1, nColumns, sortByColumn)
        _quicksortByColumn(array, pivotNewIndex + 1, right, nColumns, sortByColumn)


cdef unsigned int _partitionColumn(double* array, int left, int right, int pivotIndex, 
                                   unsigned int nColumns, unsigned int sortByColumn):

    cdef:
        double pivotValue
        unsigned int storeIndex = left, ii
        
    pivotValue = array[nColumns*pivotIndex+sortByColumn]
    _swap(&array[nColumns*pivotIndex], &array[nColumns*right], nColumns)
    for ii in range(left, right):
        if array[nColumns*ii+sortByColumn]<=pivotValue:
            _swap(&array[nColumns*ii], &array[nColumns*storeIndex], nColumns)
            storeIndex += 1
    _swap(&array[nColumns*storeIndex], &array[nColumns*right], nColumns)
    return storeIndex


cdef inline void _swap(double* a, double* b, unsigned int nColumns):

    cdef:
        double temp
        unsigned int ii
        
    for ii in range(nColumns):
        temp = a[ii]
        a[ii] = b[ii]
        b[ii] = temp
 
   


    
    
