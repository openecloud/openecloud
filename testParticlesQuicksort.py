import particles
from grid import Grid
from particleEmitter import homoLoader
import numpy
import scipy
import timeit

nx = 8 
ny = 8   
lx = 0.04
ly = 0.04
nParticles = 10
weightInit = 1.e9/nParticles

gridObj = Grid([nx,ny],[lx,ly],'elliptical')
particlesObj = particles.Particles('electrons','variableWeight')
homoLoader(gridObj, particlesObj, nParticles, weightInit)

print particlesObj.getParticleData()[:,:2]

particles.quicksortByColumn(particlesObj.getParticleData(), 0)
print particlesObj.getParticleData()[:,:2]

particles.quicksortByColumn(particlesObj.getParticleData(), 1)
print particlesObj.getParticleData()[:,:2]


nParticles = 300000
homoLoader(gridObj, particlesObj, nParticles, weightInit)
print timeit.timeit('particles.quicksortByColumn(particlesObj.getParticleData(), 0)','from __main__ import numpy, particles, particlesObj', number=1)*1000
homoLoader(gridObj, particlesObj, nParticles, weightInit)
print timeit.timeit('particlesObj.getParticleData()[:,0].sort()','from __main__ import numpy, particlesObj', number=1)*1000
