#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy
cimport numpy
cimport cython
# cimport openmp
# from cython.parallel cimport prange
import scipy as sp
import scipy.spatial as sps
import scipy.constants as spc
import particles

class ParticleManagement:
    
    def __init__(self, weightOrphans = -1):
        
        pass
    
    def categorize(self):
    
        pass
    

class SimpleManagement:
    
    def __init__(self, gridObj, particlesObj, particleBoundaryObj, mergeCriteriaWeight = 1.e-1, splitCriteriaWeight = 2.):
        
        self.gridObj = gridObj
        self.particlesObj = particlesObj
        ( [self.nx, self.ny], [self.lx, self.ly], [self.dx, self.dy] ) = gridObj.getGridBasics()
        self.mergeCriteriaWeight = mergeCriteriaWeight
        self.splitCriteriaWeight = splitCriteriaWeight
        self.particleBoundaryObj = particleBoundaryObj
        self.sigmaRandSplitPosition = 0.05
    
    def mergeAndSplit(self):
        
        oldParticleData =  self.particlesObj.getParticleData()[self.particlesObj.getParticleData()[:,5].argsort()]
        newParticleData = sp.empty(oldParticleData.shape)
        # 100ms 
        meanWeight = sp.mean(oldParticleData[:,5])
         
        toMerge = (oldParticleData[:,5] > self.mergeCriteriaWeight*meanWeight).searchsorted(True)
        toSplit = oldParticleData.shape[0]-(oldParticleData[:,5] > self.splitCriteriaWeight*meanWeight).searchsorted(True)
         
        toManage = sp.clip(sp.amax([toMerge,toSplit]),0,sp.int0(oldParticleData.shape[0]/4))
          
        if toManage > 0:
            # Splitting
            newParticleData[-toManage:] = oldParticleData[-toManage:]
            newParticleData[-toManage:,5] *= 0.5
            newParticleData[-2*toManage:-toManage] = newParticleData[-toManage:]
            newParticleData[-2*toManage:,:2] += sp.randn(2*toManage,2)*(sp.array([[self.dx,self.dy]])*self.sigmaRandSplitPosition)
            oldParticleData = oldParticleData[:-toManage]
             
            toMerge = toManage
            currentIndex = 0
            while toMerge > 0:
                kdTree = sps.cKDTree(sp.hstack((oldParticleData[toMerge:,:2], sp.array([[self.dx/spc.c, self.dy/spc.c, sp.sqrt(self.dx**2+self.dy**2)/spc.c]])*oldParticleData[toMerge:,2:5])))            
                firstIndex = kdTree.query(sp.hstack((oldParticleData[:toMerge,:2], 
                                                     sp.array([[self.lx/spc.c, self.ly/spc.c, sp.sqrt(self.dx**2+self.dy**2)/spc.c]])*oldParticleData[:toMerge,2:5])),
                                          )[1]+toMerge
                secondIndex = sp.arange(toMerge)
                 
                uniqueIndex = sp.unique(sp.concatenate((secondIndex, firstIndex)), return_index = True)[1]
                uniqueFirstIndex = uniqueIndex[uniqueIndex >= toMerge]-toMerge
                uniqueSecondIndex = uniqueIndex[uniqueIndex < toMerge]
                notDuplicate = sp.zeros(toMerge, dtype=bool)               
                notDuplicate[uniqueFirstIndex] = True 
                temp = sp.zeros(toMerge, dtype=bool)  
                temp[uniqueSecondIndex] = True
                notDuplicate *= temp
                 
                firstIndex = firstIndex[notDuplicate]
                secondIndex = secondIndex[notDuplicate]
                numberNotDuplicate = firstIndex.shape[0]
 
                weightsFirst = oldParticleData[firstIndex,5]
                weightsSecond = oldParticleData[secondIndex,5]
                selectFirst = sp.rand(numberNotDuplicate) < (weightsFirst/(weightsFirst+weightsSecond))
                selectSecond = sp.logical_not(selectFirst)
                numberFirst = sp.sum(selectFirst)
                numberSecond = numberNotDuplicate - numberFirst
                if numberFirst > 0:
                    newParticleData[currentIndex:currentIndex+numberFirst] = oldParticleData[firstIndex[selectFirst]]
                    newParticleData[currentIndex:currentIndex+numberFirst,5] += oldParticleData[secondIndex[selectFirst],5]
                if numberSecond > 0:
                    newParticleData[currentIndex+numberFirst:currentIndex+numberFirst+numberSecond] = oldParticleData[secondIndex[selectSecond]]
                    newParticleData[currentIndex+numberFirst:currentIndex+numberFirst+numberSecond,5] += oldParticleData[firstIndex[selectSecond],5]
                 
                oldParticleData = sp.delete(oldParticleData,sp.concatenate((firstIndex, secondIndex)),0)
                toMerge -= numberNotDuplicate
                currentIndex += numberNotDuplicate
             
            newParticleData[toManage:-2*toManage] = oldParticleData
            self.particlesObj.setParticleData(newParticleData)
            self.particleBoundaryObj.clipToDomain()

        
        
     
class SimpleManagement02:
    
    def __init__(self, gridObj, particlesObj, targeMacroParticleCount, mergeCriteriaWeight = 1.e-3, splitCriteriaWeight = 1.e+1):
        
        self.gridObj = gridObj
        self.particlesObj = particlesObj
        self.mergeCriteriaWeight = mergeCriteriaWeight
        self.splitCriteriaWeight = splitCriteriaWeight
        self.targeMacroParticleCount = targeMacroParticleCount
        
        ( [nx, ny], [lx, ly], [dx, dy] ) = gridObj.getGridBasics()  # @UnusedVariable
        self.maxDistance = sp.amin([dx, dy])/2.
    
    def mergeAndSplit(self):
        particleData = self.particlesObj.getParticleData()
        particleData =  particleData[particleData[:,5].argsort()]
        particleData = particleData[::-1]
        meanWeight = sp.mean(particleData[:,5])
        
        toMerge = particleData.shape[0]-(particleData[:,5] < self.mergeCriteriaWeight*meanWeight).searchsorted(True)
        toSplit = (particleData[:,5] < self.splitCriteriaWeight*meanWeight).searchsorted(True)
        
        # Force merge very light particles. This is just globally charge conserving, but as the merged particles
        # are just very light this doesn't introduce any measurable noise. Just add weight to closest particle.
        if toMerge > 0:    
            kdTree = sps.cKDTree(particleData[:-toMerge,:2])
            addToIndex = kdTree.query(particleData[-toMerge:,:2])[1]
            particleData[:-toMerge,5] += sp.bincount(addToIndex, weights = particleData[-toMerge:,5], minlength = particleData.shape[0] - toMerge)
            particleData[-toMerge:] = particleData[:toMerge]
            
        if toSplit > 0:
            kdTree = sps.cKDTree(particleData[:-toSplit-toMerge,:2])    
            ( distances, addToIndex ) = kdTree.query(particleData[-toSplit-toMerge:-toMerge,:2], p = sp.infty, distance_upper_bound = self.maxDistance)  
            haveNeighbours = distances < sp.infty
            toSplitPossible = sp.sum(haveNeighbours)
            
            if toSplitPossible > 0:   
                addToIndex = addToIndex[haveNeighbours]   
                haveNeighbours = sp.nonzero(haveNeighbours)[0] + toSplit + toMerge
                particleData[:-toMerge, 5] += sp.bincount(addToIndex, weights = particleData[haveNeighbours,5], minlength = particleData.shape[0] - toMerge)
                particleData[:toSplitPossible+toMerge,5] *= 0.5              
                particleData[haveNeighbours] = particleData[toMerge:toSplitPossible+toMerge]
 
        self.particlesObj.setParticleData(particleData)

  
cpdef globalRandom(gridObj, particlesObj, unsigned int targetMacroParticleCount):

    cdef: 
        unsigned int macroParticleCount = particlesObj.getMacroParticleCount()
        numpy.ndarray[numpy.uint16_t] keepParticlesNumpy = numpy.ones(macroParticleCount, dtype=numpy.uint16)
        unsigned short* keepParticles = <unsigned short*> keepParticlesNumpy.data
        unsigned int nCoords = particlesObj.getNCoords()
        unsigned int ii
#         int diffParticleCount = numpy.amax(numpy.array([0,numpy.int32(macroParticleCount)-numpy.int32(targetMacroParticleCount)]))
        int diffParticleCount = numpy.int32(macroParticleCount)-numpy.int32(targetMacroParticleCount)
        double factorWeight = (<double> macroParticleCount)/(<double> targetMacroParticleCount)
        
        unsigned int keepParticlesInd = numpy.random.randint(macroParticleCount)

        numpy.ndarray[numpy.double_t, ndim = 2] particleDataNumpy = particlesObj.getParticleData()
        double* particleData = <double*> particleDataNumpy.data
    
    if diffParticleCount>0:
        for ii in range(macroParticleCount):
            particleData[nCoords*ii+5] *= factorWeight
        for ii in range(diffParticleCount):
            while keepParticles[keepParticlesInd] == 0:
                keepParticlesInd = numpy.random.randint(macroParticleCount)
            keepParticles[keepParticlesInd] = 0
      
        particles.addAndRemoveParticles(particlesObj, numpy.zeros((0,6), dtype=numpy.double), keepParticlesNumpy)
    
    elif numpy.double(targetMacroParticleCount)*0.5>numpy.double(macroParticleCount):   
        for ii in range(macroParticleCount):
            particleData[nCoords*ii+5] *= .5
        particles.addAndRemoveParticles(particlesObj, particleDataNumpy, numpy.ones((macroParticleCount), dtype=numpy.uint16))
            
        
    
        
         
