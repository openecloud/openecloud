import particles
from grid import Grid
from particleEmitter import homoLoader
import numpy
import scipy
import kdTree

nx = 8 
ny = 8   
lx = 0.04
ly = 0.04
nParticles = 4
weightInit = 1.e9/nParticles

gridObj = Grid([nx,ny],[lx,ly],'elliptical')
particlesObj = particles.Particles('electrons','variableWeight')
homoLoader(gridObj, particlesObj, nParticles, weightInit)
inds = numpy.empty((nParticles,3),dtype=numpy.uint32)
print particlesObj.getParticleData()

kdTree.quicksortInd(particlesObj.getParticleData(),inds[:,:2])
print inds


