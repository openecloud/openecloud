#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy
cimport numpy
cimport cython
import particles
 
 
cpdef globalRandom(gridObj, particlesObj, unsigned int targetMacroParticleCount):

    cdef: 
        unsigned int macroParticleCount = particlesObj.getMacroParticleCount()
        numpy.ndarray[numpy.uint16_t] keepParticlesNumpy = numpy.ones(macroParticleCount, dtype=numpy.uint16)
        unsigned short* keepParticles = <unsigned short*> keepParticlesNumpy.data
        unsigned int nCoords = particlesObj.getNCoords()
        unsigned int ii
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
            
        
    
        
         
