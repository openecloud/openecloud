'''
This example is based on example03, but with added quadrupole field with octupole
focusing errors.
'''

# openecloud imports
from grid import Grid
from poissonSolver import PoissonSolver
import particles
from particleBoundary import AbsorbElliptical
from particleEmitter import FurmanEmitter, homoLoader
import magneticField
import particleManagement
from beam import LHCBeam
import plot

# Some standard library imports
import numpy
import datetime
import matplotlib.pyplot as mpl

# Several simulation parameters
nx = 32     # Number of cells in x
ny = 32     # and y
lx = 0.04   # Domains size in x
ly = 0.04   # and y
nParticles = 100000             # Number of macro particles
weightInit = 1.e9/nParticles    # Initial weight of macro particles
nMaxParticles = nParticles      # Maximum number of macro particles
dt = 2.155172413795e-11         # Time step
nTimeSteps = 10000              # Number of time steps of the simulation
nAnimate = 20                   # Number of time steps between graphical output
nTimeStepsPartMan = 50          # Number of time steps between particle management
nHistoryOut = 1000              # Number of time steps between history output

# Generate all objects
gridObj = Grid(nx,ny,lx,ly,1)                                       # Elliptical boundary with constant radius is circular
beamObj = LHCBeam(gridObj,dt)                                       # LHC beam type
particlesObj = particles.Particles('electrons','variableWeight')    # Particles types and kind
particleBoundaryObj = AbsorbElliptical(gridObj,particlesObj)        # Particle boundary fitting to grid/field boundary
secElecEmitObj = FurmanEmitter(particleBoundaryObj,particlesObj)    # Secondary emission at particle boundary
poissonSolverObj = PoissonSolver(gridObj)                           # Poisson solver for electric field calculation

# Some setup
homoLoader(gridObj, particlesObj, particleBoundaryObj, nParticles, weightInit)      # Loads a homogeneous particle distribution
bAtGridPoints = magneticField.multipoleExpansion(gridObj, [0., 5., 0., 0.01])       # Prepare quadrupole field (with error).

physicalParticleCount = numpy.zeros(nTimeSteps, dtype=numpy.uint0) 			        # History of physical particle count
macroParticleCount = numpy.zeros(nTimeSteps, dtype=numpy.uint0)     			    # History of macro particle count
fig0 = mpl.figure(1,figsize=(13,12))                                                # Plot for dynamic graphical output
mpl.show(block=False)
            
# Main loop
for ii in range(nTimeSteps):
    
    print 'Step: ' + str(ii+1) + ' at ' + str(datetime.datetime.utcnow())   # Nice status update in command line
    
    physicalParticleCount[ii] = numpy.sum(particlesObj.getParticleData()[:,5]) # Calculate and save particle count 
    macroParticleCount[ii] = particlesObj.getMacroParticleCount()              # Save macro particle count

    # Particle management only if necessary and at the correct time step
    if ii>0 and (macroParticleCount[ii]>nMaxParticles*1.05 or macroParticleCount[ii]<nMaxParticles*0.95) \
            and numpy.mod(ii, nTimeStepsPartMan) == 0:                
        particleManagement.localRandom(gridObj, particlesObj, particleBoundaryObj, nMaxParticles)

    # Callculate the grid weights of the particles and scatter the charge to the grid
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
    particlesObj.bFieldToParticles(gridObj, bAtGridPoints)  

    # Push particles with magnetic field.
    particlesObj.borisPush(dt, typeBField=2)

    # Check which particles are inside and absorb particles outside
    particleBoundaryObj.indexInside()    
    particleBoundaryObj.saveAbsorbed()

    # If particles were absorbed at the particle boundary do secondary emission
    if particleBoundaryObj.getAbsorbedMacroParticleCount() > 0:
        particleBoundaryObj.calculateInteractionPoint()
        particleBoundaryObj.calculateNormalVectors()        
        secondaries = secElecEmitObj.generateSecondaries()
        particlesObj.addAndRemoveParticles(secondaries, particlesObj.getIsInside())
       

