'''
Created on May 24, 2013

@author: ohaas
'''
import scipy as sp
import numpy
import matplotlib.pyplot as mpl
from scipy.constants import *


def plotPotential(gridObj, poissonSolverObj, figObj = None, subPlot = None, show = False):
    phi = poissonSolverObj.getPhi()
    ( [nx, ny], [lx, ly], [dx, dy] ) = gridObj.getGridBasics()  # @UnusedVariable

    if figObj == None:
        figObj = mpl.figure()
    
    if subPlot == None:
        figObj.clf()
        subPlot = figObj.add_subplot(111, aspect=1)
    else:
        pass
        
    img = subPlot.imshow(sp.reshape(phi,(ny,nx)), extent=[-lx/2., lx/2., -ly/2., ly/2.], origin='lower')
    mpl.colorbar(img)
    subPlot.set_xlabel('x in m')
    subPlot.set_ylabel('y in m')
    subPlot.set_title('Potential')
    subPlot.set_ylim([-ly/2.,ly/2.])
    subPlot.set_xlim([-lx/2.,lx/2.])
    if show:
        mpl.show(block=False)


def plotMacroParticles(gridObj, particlesObj, figObj = None, subPlot = None, colorWeight = False, show = False):
    particleData = particlesObj.getParticleData()
    ( [nx, ny], [lx, ly], [dx, dy] ) = gridObj.getGridBasics()  # @UnusedVariable

    if figObj == None:
        figObj = mpl.figure()
    
    if subPlot == None:
        figObj.clf()
        subPlot = figObj.add_subplot(111, aspect=1)
    else:
        pass
   
    if colorWeight:
        subPlot.scatter(particleData[:,0],particleData[:,1], c=particleData[:,5])
    else:
        subPlot.plot(particleData[:,0],particleData[:,1],'.', markersize=4)
     
    subPlot.set_xlabel('x in m')
    subPlot.set_ylabel('y in m')
    subPlot.set_title('Particle Position')
    subPlot.set_ylim([-ly/2.,ly/2.])
    subPlot.set_xlim([-lx/2.,lx/2.])
    if show:
        mpl.show(block=False)
  
    
def plotChargeDensity(gridObj, particlesObj, figObj = None, subPlotObj = None, logScale = True, show = False):

    ( [nx, ny], [lx, ly], [dx, dy] ) = gridObj.getGridBasics()  # @UnusedVariable
    q = sp.reshape(particlesObj.getChargeOnGrid()*gridObj.dati,(ny,nx))
    
    if figObj == None:
        figObj = mpl.figure()
    
    if subPlotObj == None:
        figObj.clf()
        subPlotObj = figObj.add_subplot(111)
    else:
        pass
    
    extent = [-.5, .5, -.5, .5]   
    if logScale:
        meanQ = sp.mean(q)
        img = subPlotObj.imshow(sp.log10(q/meanQ+1), extent=extent, origin='lower', aspect=ly/lx)
    else:
        img = subPlotObj.imshow(q, extent=extent, origin='lower')
    cbar = mpl.colorbar(img, shrink=ly/lx)
    cbar.set_label(r'$\log \left( \varrho/\varrho_0 + 1 \right) $', fontsize=20, rotation=90)
    subPlotObj.set_xlabel(r'$x/w_x$', fontsize=20)
    subPlotObj.set_ylabel(r'$y/w_y$', fontsize=20)
    subPlotObj.set_ylim([-.5,.5])
    subPlotObj.set_xlim([-.5,.5])
    if show:
        mpl.show(block=False)

    
def plotMacroParticleDensity(gridObj, particlesObj, figObj = None, subPlot = None, logScale = True, show = False):

    ( [nx, ny], [lx, ly], [dx, dy] ) = gridObj.getGridBasics()  # @UnusedVariable
    macroParticleDensity = sp.reshape(sp.bincount(particlesObj.getInCell(), minlength = nx*ny),(ny,nx))[:-1,:-1]
    
    if figObj == None:
        figObj = mpl.figure()
    
    if subPlot == None:
        figObj.clf()
        subPlot = figObj.add_subplot(111, aspect=ly/lx)
    
    extent=[-.5, .5, -.5, .5]    
    if logScale:
        meanMacroParticleDensity = sp.mean(macroParticleDensity)
        img = subPlot.imshow(sp.log10(macroParticleDensity/meanMacroParticleDensity+1), extent=extent, origin='lower', aspect=ly/lx)
    else:
        img = subPlot.imshow(macroParticleDensity, extent=extent, origin='lower', aspect=ly/lx)
    cbar = mpl.colorbar(img, shrink=ly/lx)
    cbar.set_label(r'$\log \left( \rho/\rho_0 + 1 \right) $', fontsize=20, rotation=90)
    subPlot.set_xlabel(r'$x/w_x$', fontsize=20)
    subPlot.set_ylabel(r'$y/w_y$', fontsize=20)
    subPlot.set_ylim([-.5,.5])
    subPlot.set_xlim([-.5,.5])
    if show:
        mpl.show(block=False)
    
def plotAllAtRuntime(gridObj, particlesObj, poissonSolverObj, particleCounts, macroCounts, currentStep, figObj = None):

    ( [nx, ny], [lx, ly], [dx, dy] ) = gridObj.getGridBasics()  # @UnusedVariable
    
    if figObj == None:
        figObj = mpl.figure()
    
    figObj.clf()
    chargeDensityPlot = figObj.add_subplot(321, aspect=ly/lx)
    potentialPlot = figObj.add_subplot(323, aspect=ly/lx)
    particleCountPlot = figObj.add_subplot(324)
    macroDensityPlot = figObj.add_subplot(322, aspect=ly/lx)
    macroPlot = figObj.add_subplot(326, aspect = 'equal')
    macroCountPlot = figObj.add_subplot(325)
    
    extent=[-.5, .5, -.5, .5] 
    macroParticleDensity = sp.reshape(sp.bincount(particlesObj.getInCell(), minlength = nx*ny),(ny,nx))[:-1,:-1]
    meanMacroParticleDensity = sp.mean(macroParticleDensity)
    macroDensityPlot.imshow(sp.log10(macroParticleDensity/meanMacroParticleDensity+1), extent=extent, 
                            origin='lower', aspect=ly/lx, interpolation='nearest')
    macroDensityPlot.set_title('Macro Density')
    
    q = sp.reshape(particlesObj.chargeOnGrid*gridObj.dati,(ny,nx))
    meanQ = sp.mean(q)
    chargeDensityPlot.imshow(sp.log10(q/meanQ+1), extent=extent, origin='lower', aspect=ly/lx)
    chargeDensityPlot.set_title('Charge Density')

    phi = poissonSolverObj.getPhi()        
    potentialPlot.imshow(sp.reshape(phi,(ny,nx)), extent=[-.5, .5, -.5, .5], origin='lower', aspect=ly/lx)
    potentialPlot.set_title('Potential')
    
    particleCountPlot.plot(particleCounts[:currentStep])
    particleCountPlot.set_title('Particle Count')
    
    macroCountPlot.plot(macroCounts[:currentStep])
    macroCountPlot.set_title('Macro Particle Count')
    
    macroPlot.plot(particlesObj.getParticleData()[:,0],particlesObj.getParticleData()[:,1], '.')
    macroPlot.set_title('Macro Particles')
    macroPlot.set_ylim([-ly/2.,ly/2.])
    macroPlot.set_xlim([-lx/2.,lx/2.])
        
    mpl.tight_layout()
    mpl.show(block=False)
    mpl.draw()


def plotParticleCount(particleCounts, dt, currentStep, figObj = None, subPlot = None, show = False):

    if figObj == None:
        figObj = mpl.figure()
    
    if subPlot == None:
        figObj.clf()
        subPlot = figObj.add_subplot(111)
    if currentStep > 0:
        tmax = dt*currentStep
        exp = numpy.int0(-numpy.log10(tmax)+1)
        lambdamax = numpy.amax(particleCounts[:currentStep])
        expl = numpy.int0(numpy.log10(lambdamax))
        mpl.plot(numpy.arange(currentStep)*dt*10**exp,particleCounts[:currentStep]*10**(-expl))
    else:
        mpl.plot(numpy.array([0]))
        exp = 0
        expl = 0
    subPlot.set_xlabel(r'$t\,\mathrm{in}\,10^{-'+str(exp)+'}\mathrm{s}$', fontsize=20)
    subPlot.set_ylabel(r'$\lambda\,\mathrm{in}\,10^{'+str(expl)+'}\mathrm{m}^{-1}$', fontsize=20)
    if show:
        mpl.show(block=False)
        
        
def plotHeatLoad(heatLoad, dt, currentStep, figObj = None, subPlot = None, show = False):

    if figObj == None:
        figObj = mpl.figure()
    
    if subPlot == None:
        figObj.clf()
        subPlot = figObj.add_subplot(111)
    if currentStep > 0:
        tmax = dt*currentStep
        exp = numpy.int0(-numpy.log10(tmax)+1)
        lambdamax = numpy.amax(heatLoad[:currentStep])
        expl = numpy.int0(numpy.log10(lambdamax))
        mpl.plot(numpy.arange(currentStep)*dt*10**exp,heatLoad[:currentStep]*10**(-expl))
    else:
        mpl.plot(numpy.array([0]))
        exp = 0
        expl = 0
    subPlot.set_xlabel(r'$t\,\mathrm{in}\,10^{-'+str(exp)+'}\mathrm{s}$', fontsize=20)
    subPlot.set_ylabel(r'$heatLoad\,\mathrm{in}\,10^{'+str(expl)+'}\mathrm{eV}$', fontsize=20)
    if show:
        mpl.show(block=False)
        
def plotEnergyLocal(gridObj, particlesObj, figObj = None, subPlot = None, show = False):

    ( [nx, ny], [lx, ly], [dx, dy] ) = gridObj.getGridBasics()  # @UnusedVariable
    inCell = particlesObj.getInCell()
    particleData = particlesObj.getParticleData()
    energies = 0.5*particlesObj.getParticleMass()/elementary_charge*(numpy.sum(particleData[:,2:5]**2, axis=1)*particleData[:,5])
    energyLocal = numpy.bincount(inCell, weight=energies, minLength=nx*ny)
    countsPerCell = numpy.bincount(inCell, minLength=nx*ny)
    energyLocal[countsPerCell>0] /= countsPerCell[countsPerCell>0]
    
    energyLocal = numpy.reshape(energyLocal, (ny,nx))
    if figObj == None:
        figObj = mpl.figure()
    
    if subPlot == None:
        figObj.clf()
        subPlot = figObj.add_subplot(111)
    else:
        pass
    
    extent = [-.5, .5, -.5, .5]   
#     if logScale:
#         meanQ = sp.mean(q)
#         img = subPlot.imshow(sp.log10(q/meanQ+1), extent=extent, origin='lower', aspect=ly/lx)
#     else:
#         img = subPlot.imshow(q, extent=extent, origin='lower')
    img = subPlot.imshow(energyLocal, extent=extent, origin='lower')
    cbar = mpl.colorbar(img, shrink=ly/lx)
    cbar.set_label(r'$\log \left( \varrho/\varrho_0 + 1 \right) $', fontsize=20, rotation=90)
    subPlot.set_xlabel(r'$x/w_x$', fontsize=20)
    subPlot.set_ylabel(r'$y/w_y$', fontsize=20)
    subPlot.set_ylim([-.5,.5])
    subPlot.set_xlim([-.5,.5])
    if show:
        mpl.show(block=False)
