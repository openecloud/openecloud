#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy
cimport numpy
cimport cython
cimport particles
cimport grid
cimport kdTree
cimport randomGen
from constants cimport *


cdef inline double min(double a, double b) nogil:
    if a<=b:
        return a
    else:
        return b 
        
cdef inline double max(double a, double b) nogil:
    if a>=b:
        return a
    else:
        return b          
        
cdef extern from "math.h":
    double sqrt(double x) nogil
    
# Global random particle management. Sometimes called russian roulette method.
# For now only meant for same weight particles.
# If too many particles selects a particle randomly, kills it and adds its weight evenly to all other particles.
# If initial particle number less than half splits all particles in half 
cpdef globalRandom(grid.Grid gridObj, particles.Particles particlesObj, unsigned int targetMacroParticleCount):

    cdef: 
        unsigned int ii, jj
        unsigned int macroParticleCount = particlesObj.getMacroParticleCount()
        unsigned int nCoords = particlesObj.getNCoords()
        int diffParticleCount = (<int> macroParticleCount) - (<int> targetMacroParticleCount)
        double factorWeight = (<double> macroParticleCount)/(<double> targetMacroParticleCount)     
        unsigned int randInd = randomGen.randi(macroParticleCount)
        
        unsigned short[:] keepParticles = numpy.ones(macroParticleCount, dtype=numpy.ushort)
        double[:,:] particleData = particlesObj.getParticleData()
        double[:,:] addParticleData
    
    if diffParticleCount > 0:
        for ii in range(macroParticleCount):
            particleData[ii,5] *= factorWeight
        for ii in range(diffParticleCount):
            while keepParticles[randInd] == 0:
                randInd = randomGen.randi(macroParticleCount)
            keepParticles[randInd] = 0
      
        particlesObj.addAndRemoveParticles(numpy.zeros((0,6), dtype=numpy.double), keepParticles)
    
    elif targetMacroParticleCount > macroParticleCount:   
        addParticleData = numpy.empty((-diffParticleCount,nCoords), dtype=numpy.double)
        for ii in range(macroParticleCount):
            particleData[ii,5] *= factorWeight
        for ii in range(-diffParticleCount):
            randInd = randomGen.randi(macroParticleCount)
            for jj in range(nCoords):
                addParticleData[ii,jj] = particleData[randInd,jj]
        particlesObj.addAndRemoveParticles(addParticleData, numpy.ones((macroParticleCount), dtype=numpy.ushort))
            

# Uses kdTree to merge nearest neighbor (full phase space) particles, thus locally
# conserving momentum, energy and velocity distribution. Splits with a slightly
# modified Lapenta-rule, which conserves linear grid moments.
# This method aims for globally similar weight particles.
cpdef localRandom(grid.Grid gridObj, particles.Particles particlesObj, particleBoundaryObj, 
                  unsigned int targetMacroParticleCount, double[:] phaseSpaceWeights = numpy.empty(0, dtype=numpy.double),
                  double thresholdFactor = 2., double lambdavFactor = 1., double factorDisLimit = 1.,
                  unsigned int mergeScheme = 0):
    cdef:
        unsigned int ii, jj
        unsigned int macroParticleCount = particlesObj.getMacroParticleCount()
        unsigned int nCoords = particlesObj.getNCoords()     
        unsigned int toSplitCount, toMergeCount
        double meanWeight = particlesObj.getMeanWeight()
        double goalWeight = meanWeight*macroParticleCount/targetMacroParticleCount
        double maxThreshold = thresholdFactor*goalWeight 
        double minThreshold = goalWeight/thresholdFactor  
        double factorLimitSplitMerge, randomTemp, foundDist
        unsigned short[:] keepParticles = numpy.ones(macroParticleCount, dtype=numpy.ushort)
        double newCoords0Buff[2]
        double newCoords1Buff[2]
        double[:] newCoords0 = newCoords0Buff, newCoords1 = newCoords1Buff
        double[:,:] particleData = particlesObj.getParticleData()
        double[:,:] addParticleData
        double dxi = 1./gridObj.getDx(), dyi = 1./gridObj.getDy(), ds = min(gridObj.getDx(), gridObj.getDy())
        double lxHalf = gridObj.getLx()*0.5, lyHalf = gridObj.getLy()*0.5
        unsigned int inCellCoordsNormX, inCellCoordsNormY, foundInd
        double vTypical = 0., factorVScale
        kdTree.KDTree tree
        
    # Sort particles by weight.
    particlesObj.sortByColumn(5)
    
    # Number of operations determined by weight tresholds.
    toMergeCount = _searchSorted(particleData[:,5], minThreshold, macroParticleCount)
    toSplitCount = macroParticleCount-_searchSorted(particleData[:,5], maxThreshold, macroParticleCount)
    meanInd = _searchSorted(particleData[:,5], meanWeight, macroParticleCount)
    # Limit maximum number of operations.
    if 2*toMergeCount+toSplitCount>=macroParticleCount:
        factorLimitSplitMerge = 0.95*macroParticleCount/(2*toMergeCount+toSplitCount)
        toMergeCount = <unsigned int> (toMergeCount*factorLimitSplitMerge)
        toSplitCount = <unsigned int> (toSplitCount*factorLimitSplitMerge)
    # Number of operations given by total particle number.
    newMacroParticleCount = macroParticleCount+toSplitCount-toMergeCount
    if newMacroParticleCount>targetMacroParticleCount:       
        toMergeCount+= newMacroParticleCount-targetMacroParticleCount
    elif newMacroParticleCount<targetMacroParticleCount:
        toSplitCount+= targetMacroParticleCount-newMacroParticleCount
    # Limit maximum number of operations (again).
    if 2*toMergeCount+toSplitCount>=macroParticleCount:
        factorLimitSplitMerge = 0.95*macroParticleCount/(2*toMergeCount+toSplitCount)
        toMergeCount = <unsigned int> (toMergeCount*factorLimitSplitMerge)
        toSplitCount = <unsigned int> (toSplitCount*factorLimitSplitMerge)
    
    # Calculate typical velocity for later before modifying particle data.
    for ii in range(macroParticleCount):
        vTypical += (particleData[ii,2]**2+particleData[ii,3]**2+particleData[ii,4]**2)*particleData[ii,5]**2
    vTypical = sqrt(vTypical)/(macroParticleCount*meanWeight)
    vTypical = max(vTypical,1.e-12*c)    # Lower limit to prevent unrealistic weighting of velocity
    
    # Split particles.
    # Lapenta-like split, but with equal weights of splitted particles.
    addParticleData = numpy.empty((toSplitCount,nCoords), dtype=numpy.double)
    jj = 0
    for ii in range(macroParticleCount-toSplitCount,macroParticleCount):
        particleData[ii,5] *= 0.5
        addParticleData[jj,2] = particleData[ii,2]
        addParticleData[jj,3] = particleData[ii,3] 
        addParticleData[jj,4] = particleData[ii,4]       
        addParticleData[jj,5] = particleData[ii,5]      
        inCellCoordsNormX = <unsigned int> ((particleData[ii,0]+lxHalf)*dxi)
        inCellCoordsNormY = <unsigned int> ((particleData[ii,1]+lyHalf)*dyi)
        if inCellCoordsNormX<0.5:
            randomTemp = randomGen.rand()*min(inCellCoordsNormX, 0.5-inCellCoordsNormX)
        else:
            randomTemp = randomGen.rand()*min(inCellCoordsNormX-0.5, 1.-inCellCoordsNormX)
        newCoords0[0] = inCellCoordsNormX+randomTemp
        newCoords1[0] = inCellCoordsNormX-randomTemp
        if inCellCoordsNormY<0.5:
            randomTemp = randomGen.rand()*min(inCellCoordsNormY, 0.5-inCellCoordsNormY)
        else:
            randomTemp = randomGen.rand()*min(inCellCoordsNormY-0.5, 1.-inCellCoordsNormY)
        newCoords0[1] = inCellCoordsNormY+randomTemp
        newCoords1[1] = inCellCoordsNormY-randomTemp      
        if particleBoundaryObj.isInside(newCoords0[0],newCoords0[1])==0 or \
           particleBoundaryObj.isInside(newCoords1[0],newCoords1[1])==0:
            addParticleData[jj,0] = particleData[ii,0]
            addParticleData[jj,1] = particleData[ii,1]
        else:
            particleData[ii,0] = newCoords0[0]
            particleData[ii,1] = newCoords0[1]
            addParticleData[jj,0] = newCoords1[0]
            addParticleData[jj,1] = newCoords1[1]
        jj += 1
        
    # Merge particles.
    if phaseSpaceWeights.shape[0]==0:
        phaseSpaceWeights = numpy.ones(nCoords-1, dtype=numpy.double)
        lambdav = lambdavFactor*ds/vTypical
        phaseSpaceWeights[2] *= lambdav
        phaseSpaceWeights[3] *= lambdav
        phaseSpaceWeights[4] *= lambdav
    tree = kdTree.KDTree(particleData[:macroParticleCount-toSplitCount,:5], weights=phaseSpaceWeights)
    ii = 0
    jj = 0
    # All merge schemes take the position of the merged particle from one of the particles
    # at random for now.
    # Randomly take velocity of one of the particles.
    if mergeScheme == 0:
        while jj<toMergeCount and ii<(macroParticleCount-toSplitCount-1):
            if keepParticles[ii] == 0:
                ii += 1    
            else: 
                if tree.remove(ii) == 1:
                    tree.query(particleData[ii,:5], &foundInd, &foundDist)
                    if foundDist < factorDisLimit*ds:              
                        if randomGen.rand() < (particleData[ii,5]/(particleData[foundInd,5]+particleData[ii,5])):
                            particleData[ii,5] += particleData[foundInd,5]
                            keepParticles[foundInd] = 0
                        else:
                            particleData[foundInd,5] += particleData[ii,5]
                            keepParticles[ii] = 0
                        tree.remove(foundInd)
                        ii += 1
                        jj += 1
                    else:
                        ii += 1
                else:
                    ii += 1
    # Randomly take velocity of one of the particles but scale to conserve energy.
    elif mergeScheme == 1:
        while jj < toMergeCount and ii < (macroParticleCount-toSplitCount-1):
            if keepParticles[ii] == 0:
                ii += 1    
            else: 
                if tree.remove(ii) == 1:
                    tree.query(particleData[ii,:5], &foundInd, &foundDist)
                    if foundDist < factorDisLimit*ds:                                     
                        if randomGen.rand() < (particleData[ii,5]/(particleData[foundInd,5]+particleData[ii,5])):     
                            factorVScale = particleData[ii,5]/(particleData[foundInd,5]+particleData[ii,5]) + \
                                           (particleData[foundInd,2]**2 + particleData[foundInd,3]**2 + 
                                            particleData[foundInd,4]**2)/(particleData[ii,2]**2 + 
                                            particleData[ii,3]**2 + particleData[ii,4]**2) * \
                                            particleData[foundInd,5]/(particleData[foundInd,5]+particleData[ii,5])
                            particleData[ii,2] *= factorVScale
                            particleData[ii,3] *= factorVScale
                            particleData[ii,4] *= factorVScale
                            particleData[ii,5] += particleData[foundInd,5]
                            keepParticles[foundInd] = 0
                        else:
                            factorVScale = particleData[foundInd,5]/(particleData[foundInd,5]+particleData[ii,5]) + \
                                           (particleData[ii,2]**2 + particleData[ii,3]**2 + 
                                            particleData[ii,4]**2)/(particleData[foundInd,2]**2 + 
                                            particleData[foundInd,3]**2 + particleData[foundInd,4]**2) * \
                                            particleData[ii,5]/(particleData[foundInd,5]+particleData[ii,5])
                            particleData[foundInd,2] *= factorVScale
                            particleData[foundInd,3] *= factorVScale
                            particleData[foundInd,4] *= factorVScale
                            particleData[foundInd,5] += particleData[ii,5]
                            keepParticles[ii] = 0
                        tree.remove(foundInd)
                        ii += 1
                        jj += 1
                    else:
                        ii += 1
                else:
                    ii += 1
    # Finally concatenate.
    particlesObj.addAndRemoveParticles(addParticleData, keepParticles)

        
    
        
# Returns first index which is larger than value.       
cdef unsigned int _searchSorted(double[:] array, double value, unsigned int arrayLen) nogil:      
    cdef:
        unsigned int ind0, ind1, ind2
    if array[arrayLen-1]<=value:
        return arrayLen
    elif array[0]>=value:
        return 0
    else:
        ind0 = 0
        ind2 = arrayLen-1
        ind1 = <unsigned int> ((ind2+ind0)*0.5)
        while ind1 != ind0:
            if array[ind1]>value:
                ind2 = ind1
            else:
                ind0 = ind1
            ind1 = <unsigned int> ((ind2+ind0)*0.5)
        return ind1+1
        
