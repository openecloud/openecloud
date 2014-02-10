import particles
from grid import Grid
from particleEmitter import homoLoader
import numpy
import scipy
import scipy.spatial
import kdTree
import timeit
#import cProfile

nx = 8 
ny = 8   
lx = 0.04
ly = 0.04
nParticles = 300001
nDims = 5
weightInit = 1.e9/nParticles

gridObj = Grid([nx,ny],[lx,ly],'elliptical')
particlesObj = particles.Particles('electrons','variableWeight')
homoLoader(gridObj, particlesObj, nParticles, weightInit)
data = particlesObj.getParticleData()[:,:nDims]

#print '-----------------------'
#print 'build kdtree', timeit.timeit('kdTree.KDTree(data)','from __main__ import kdTree, data',number=1)*1000

#tree = kdTree.KDTree(particlesObj.getParticleData()[:,:nDims])
#print 'query kdtree', timeit.timeit('tree.query(numpy.zeros(nDims))','from __main__ import tree, data, numpy, nDims',number=30000)*1000
#result = tree.query(numpy.zeros(nDims))
#print 'nearest neighbor', result

#print 'remove kdtree', timeit.timeit('for ii in range(30000): tree.remove(ii)','from __main__ import tree, data, numpy, nDims',number=1)*1000

tree = kdTree.KDTree(data)
for ii in range(200000):
    if tree.remove(ii)!=1:
        print ii
        print data[ii]
#print '-----------------------'
#print 'build kdtree scipy', timeit.timeit('scipy.spatial.cKDTree(data)','from __main__ import scipy, data',number=1)*1000

#tree = scipy.spatial.cKDTree(data)
#print 'query kdtree scipy', timeit.timeit('tree.query(numpy.zeros(nDims))','from __main__ import tree, data, numpy, nDims',number=30000)*1000
#result = tree.query(numpy.zeros(nDims))[1]
#print 'nearest neighbor', result


