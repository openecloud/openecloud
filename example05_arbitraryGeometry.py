'''
This example is a low-resolution simulation of the electron cloud buildup in
a LHC-like scenario. Dynamic graphical output is used to illustrate the
development of the electron cloud and the behavior of the code. The graphical
output takes a lot of time, so this is not recommended for normal use.
'''

# openecloud imports
from grid import Grid
from poissonSolver import PoissonSolver
import particles
from particleBoundary import AbsorbArbitraryCutCell
from particleEmitter import FurmanEmitter, homoLoader
import particleManagement
from beam import LHCBeam
import plot

# Some standard library imports
import numpy
import datetime
import matplotlib.pyplot as mpl

# Several simulation parameters
nx = 44      # Number of cells in x
ny = 36      # and y
lx = 0.044   # Domains size in x
ly = 0.036   # and y
nParticles = nx*ny*20           # Number of macro particles
weightInit = 1.e9/nParticles    # Initial weight of macro particles
nMaxParticles = nParticles      # Maximum number of macro particles
dt = 2.155172413795e-11         # Time step
nTimeSteps = 10000              # Number of time steps of the simulation
nAnimate = 50                   # Number of time steps between graphical output
nHistoryOut = 10000             # Number of time steps between history output

# Boundary function
def LHCBound(x,y):
    if x**2+y**2<=(0.5*lx)**2 and y**2<0.5*ly:
        return 1
    else:
        return 0
        
        
# Generate all objects      
gridObj = Grid(nx,ny,lx,ly,3, boundFunc=LHCBound)                   # Arbitrary boundary with constant radius is circular
beamObj = LHCBeam(gridObj,dt)                                       # LHC beam type
particlesObj = particles.Particles('electrons','variableWeight')    # Particles types and kind
particleBoundaryObj = AbsorbArbitraryCutCell(gridObj,particlesObj)  # Particle boundary fitting to grid/field boundary
secElecEmitObj = FurmanEmitter(particleBoundaryObj,particlesObj)    # Secondary emission at particle boundary
poissonSolverObj = PoissonSolver(gridObj)                           # Poisson solver for electric field calculation

# Some setup
homoLoader(gridObj, particlesObj, particleBoundaryObj, nParticles, weightInit)          # Loads a homogeneous particle distribution
#mpl.plot(particlesObj.getParticleData()[:,0],particlesObj.getParticleData()[:,1],'.')
#print numpy.asarray(gridObj.getInsideFaces())
#mpl.show()

physicalParticleCount = numpy.zeros(nTimeSteps, dtype=numpy.uint0)                      # History of physical particle count
macroParticleCount = numpy.zeros(nTimeSteps, dtype=numpy.uint0)                         # History of macro particle count
fig0 = mpl.figure(1,figsize=(13,12))                                                    # Plot for dynamic graphical output
mpl.show(block=False)
            
# Main loop
for ii in range(nTimeSteps):

    print 'Step: ' + str(ii+1) + ' at ' + str(datetime.datetime.utcnow())   # Nice status update in command line
    
    physicalParticleCount[ii] = numpy.sum(particlesObj.getParticleData()[:,5]) # Calculate and save particle count 
    macroParticleCount[ii] = particlesObj.getMacroParticleCount()              # Save macro particle count
    
    # Particle management only if necessary and at the correct time step.
    # This method (globalRandom) is done only rarely as it introduces noise. For better methods see other examples.
    if ii>0 and macroParticleCount[ii]>=1.2*nMaxParticles:                   
        particleManagement.globalRandom(gridObj, particlesObj, nMaxParticles) 

    # Calculate the grid weights of the particles and scatter the charge to the grid
    particlesObj.calcGridWeights(gridObj)    
    particlesObj.chargeToGrid(gridObj)       

    # Graphical output (Not recommended during runtime in general, because slow. Just for example.)
    if ii>0 and numpy.mod(ii, nAnimate) == 0:       
        plot.plotAllAtRuntime(gridObj, particlesObj, poissonSolverObj, physicalParticleCount, 
                              macroParticleCount, ii, figObj = fig0)
        mpl.draw()
    
    # Save data at specified time steps
    if ii>0 and numpy.mod(ii, nHistoryOut) == 0:  
        numpy.save('physicalParticleCount.npy', physicalParticleCount[:ii])
        numpy.save('particleData.npy', particlesObj.getParticleData())

    # Solve Poisson problem with electron charge on grid and imprinted beam charge
    poissonSolverObj.solve(numpy.asarray(particlesObj.getChargeOnGrid())+beamObj.getCharge(ii*dt))     

    # Interpolate electric field to particle position
    particlesObj.eFieldToParticles(gridObj, poissonSolverObj.getEAtGridPoints())

    # Push particles (without magnetic field. This special function is faster than the general one.)
    particlesObj.borisPush(dt, typeBField=0)

    # Check which particles are inside and absorb particles outside
    particleBoundaryObj.indexInside()    
    particleBoundaryObj.saveAbsorbed()

    # If particles were absorbed at the particle boundary do secondary emission
    if particleBoundaryObj.getAbsorbedMacroParticleCount() > 0:
        particleBoundaryObj.calculateInteractionPoint()
        particleBoundaryObj.calculateNormalVectors()
        particlesObj.addAndRemoveParticles(secElecEmitObj.generateSecondaries(), particlesObj.getIsInside())
       
