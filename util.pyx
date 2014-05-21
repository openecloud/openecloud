import numpy

cimport particles
from constants cimport *
cimport numpy


cpdef averageEnergy(particles.Particles partObj):
    cdef:
        double [:,:] particleData = partObj.getParticleData()
        unsigned int ii
        double energy = 0, sumWeight = 0
        
    for ii in range(partObj.getMacroParticleCount()):
        energy += (particleData[ii,2]**2 + particleData[ii,3]**2 + particleData[ii,4]**2)*particleData[ii,5]
        sumWeight += particleData[ii,5]
    
    energy *= partObj.getParticleMass()*0.5/elementary_charge/sumWeight
    
    return energy     
