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
from particleBoundary import AbsorbElliptical
from particleEmitter import FurmanEmitter, homoLoader
import particleManagement
from beam import LHCBeam
import plot

# Some standard library imports
import numpy
import datetime
import matplotlib.pyplot as mpl

# Several simulation parameters
nx = 256    # Number of cells in x
ny = 128    # and y
lx = 0.08   # Domains size in x
ly = 0.04   # and y
strength=1.0 #multipole strength
order=2 #multipole order 0 1 2 3
nParticles = 100000              # Number of macro particles
weightInit = 1.e9/nParticles    # Initial weight of macro particles
nMaxParticles = nParticles      # Maximum number of macro particles
dt = 1.155172413795e-11         # Time step
nTimeSteps = 10000              # Number of time steps of the simulation
nAnimate = 160                   # Number of time steps between graphical output
nTimeStepsPartMan = 1160        # Number of time steps between particle management (should be once right before next beam bunch)
nHistoryOut = 1000              # Number of time steps between history output

# Generate all objects
gridObj = Grid([nx,ny],[lx,ly],'elliptical')                        # Elliptical boundary with constant radius is circular
beamObj = LHCBeam(gridObj,dt)                                       # LHC beam type
particlesObj = particles.Particles('electrons','variableWeight')    # Particles types and kind
particleBoundaryObj = AbsorbElliptical(gridObj,particlesObj)        # Particle boundary fitting to grid/field boundary
secElecEmitObj = FurmanEmitter(particleBoundaryObj,particlesObj)    # Secondary emission at particle boundary
poissonSolverObj = PoissonSolver(gridObj)                           # Poisson solver for electric field calculation
particlesObj.setBAtParticles(numpy.array([0.0,0.1,0.0]))	    #SETTING MAGNETIC FIELD OF 0.1  TESLA
# Some setup
homoLoader(gridObj, particlesObj, nParticles, weightInit)           # Loads a homogeneous particle distribution
physicalParticleCount = numpy.zeros(nTimeSteps, dtype=numpy.uint0)  # History of physical particle count
macroParticleCount = numpy.zeros(nTimeSteps, dtype=numpy.uint0)     # History of macro particle count
fig0 = mpl.figure(1,figsize=(13,12))                                # Plot for dynamic graphical output
mpl.show(block=False)
            
# Main loop
for ii in range(nTimeSteps):
    
    print 'Step: ' + str(ii+1) + ' at ' + str(datetime.datetime.utcnow())   # Nice status update in command line
    
    physicalParticleCount[ii] = numpy.sum(particlesObj.getParticleData()[:,5]) # Calculate and save particle count 
    macroParticleCount[ii] = particlesObj.getMacroParticleCount()              # Save macro particle count
    
    # Particle management only if necessary and at the correct time step
    if ii>0 and numpy.mod(ii,nTimeStepsPartMan) == 0:                   
        particleManagement.globalRandom(gridObj, particlesObj, nMaxParticles) 

    # Callculate the grid weights of the particles and scatter the charge to the grid
    particles.calcGridWeights(particlesObj, gridObj)    
    particles.chargeToGrid(particlesObj, gridObj)       

    # Graphical output (Not recommended during runtime in general, because slow. Just for example.)
    if ii>0 and numpy.mod(ii, nAnimate) == 0:       
        #plot.plotMacroParticles(gridObj, particlesObj, figObj = fig0)
        #plot.plotParticleCount(physicalParticleCount, dt, ii, figObj = fig0)
        plot.plotAllAtRuntime(gridObj, particlesObj, poissonSolverObj, physicalParticleCount, 
                              macroParticleCount, ii, figObj = fig0)
        mpl.draw()
    
    # Save data at specified time steps
    if ii>0 and numpy.mod(ii, nHistoryOut) == 0:  
        numpy.save('physicalParticleCount.npy', physicalParticleCount[:ii])
        numpy.save('particleData.npy', particlesObj.getParticleData())

    # Solve Poisson problem with electron charge on grid and imprinted beam charge
    poissonSolverObj.solve(particlesObj.getChargeOnGrid()+beamObj.getCharge(ii*dt))     

    # Interpolate electric field to particle position
    particles.eFieldToParticles(particlesObj, gridObj, poissonSolverObj.getEAtGridPoints())
    
    # Push particles (without magnetic field. This special function is faster than the general one.)
    #particles.borisPushNonRelQuadrupole(particlesObj, dt, strength)
    particles.borisPushNonRelMultipole(particlesObj, dt, order, strength)
    # Check which particles are inside and absorb particles outside
    particleBoundaryObj.indexInside()    
    particleBoundaryObj.saveAbsorbed()

    # If particles were absorbed at the particle boundary do secondary emission
    if particleBoundaryObj.getAbsorbedMacroParticleCount() > 0:
        particleBoundaryObj.calculateInteractionPoint()
        particleBoundaryObj.calculateNormalVectors()        
        secondaries = secElecEmitObj.generateSecondaries()
        particles.addAndRemoveParticles(particlesObj, secondaries, particlesObj.getIsInside())





