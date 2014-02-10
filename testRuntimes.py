if True:
    
    import scipy as sp
    import poissonSolver
    import particles
    import particleBoundary
    import particleEmitter
    import grid
    import timeit
    
    nx = 128; ny = 128; lx = 1.; ly = 1.; nParticles = 3e+5; dt = 1.e-10;

    print 'Solver electric field on grid (rectangular)'
    gridObj = grid.Grid([nx,ny],[lx,ly],'rectangular')
    solver = poissonSolver.PoissonSolver(gridObj)
    q = sp.ones(nx*ny)
    print timeit.timeit('solver.solve(q)','from __main__ import solver; from __main__ import q; import scipy as sp', number=1)/1*1000
     
    print 'Solver electric field on grid (circular)'
    gridObj = grid.Grid([nx,ny],[lx,ly],'elliptical')
    solver = poissonSolver.PoissonSolver(gridObj)
    q = sp.ones(nx*ny)
    print timeit.timeit('solver.solve(q)','from __main__ import solver; from __main__ import q; import scipy as sp', number=1)/1*1000
        
    print 'Calculate grid weights of particles'
    gridObj = grid.Grid([nx,ny],[lx,ly],'rectangular')
    part = particles.Particles('electrons','variableWeight')
    part.setParticleData((sp.rand(nParticles,6)-0.5)*0.999)
    print timeit.timeit('part.calcGridWeights(gridObj)','from __main__ import part;from __main__ import particles; from __main__ import gridObj', number = 100)/100*1000    
 
    print 'Interpolate charge to grid'
    gridObj = grid.Grid([nx,ny],[lx,ly],'rectangular')
    part = particles.Particles('electrons','variableWeight')
    part.setParticleData((sp.rand(nParticles,6)-0.5)*0.999)
    part.calcGridWeights(gridObj)
    print timeit.timeit('part.chargeToGrid(gridObj)','from __main__ import part;from __main__ import gridObj;from __main__ import particles', number = 1)/1*1000
 
    print 'Interpolate field for particles'
    gridObj = grid.Grid([nx,ny],[lx,ly],'rectangular')
    solver = poissonSolver.PoissonSolver(gridObj)
    q = sp.ones(nx*ny)
    eAtGridPoints = solver.solve(q)
    part = particles.Particles('electrons','variableWeight')
    part.setParticleData((sp.rand(nParticles,6)-0.5)*0.999)
    part.calcGridWeights(gridObj)
    print timeit.timeit('part.eFieldToParticles(gridObj, eAtGridPoints)',
                        'from __main__ import part; from __main__ import eAtGridPoints; from __main__ import particles; from __main__ import gridObj', number = 1)/1*1000
   
   
    print 'Boris nonRel no magnetic field'
    gridObj = grid.Grid([nx,ny],[lx,ly],'rectangular')
    solver = poissonSolver.PoissonSolver(gridObj)
    q = sp.ones(nx*ny)
    eAtGridPoints = solver.solve(q)
    part = particles.Particles('electrons','variableWeight')
    part.setParticleData((sp.rand(nParticles,6)-0.5)*0.999)
    part.calcGridWeights(gridObj)
    part.eFieldToParticles(gridObj, eAtGridPoints)   
    print timeit.timeit('part.borisPush(dt, typeBField=0)','from __main__ import part; from __main__ import particles; from __main__ import dt', number = 1)/1*1000
     

    print 'Boris nonRel with magnetic field'
    gridObj = grid.Grid([nx,ny],[lx,ly],'rectangular')
    solver = poissonSolver.PoissonSolver(gridObj)
    q = sp.ones(nx*ny)
    eAtGridPoints = solver.solve(q)
    part = particles.Particles('electrons','variableWeight')
    part.setParticleData((sp.rand(nParticles,6)-0.5)*0.999)
    part.calcGridWeights(gridObj)
    part.eFieldToParticles(gridObj, eAtGridPoints)   
    part.setBAtParticles((sp.arange(3)*1.)[:,sp.newaxis])
    print timeit.timeit('part.borisPush(dt, typeBField=1)',
                        'from __main__ import part;from __main__ import particles; from __main__ import dt', number = 100)/100*1000

               
    print 'Boundary check if particles are inside (rectangular)'
    gridObj = grid.Grid([nx,ny],[lx,ly],'rectangular')
    part = particles.Particles('electrons','variableWeight')
    part.setParticleData(sp.vstack( ( (sp.rand(nParticles-1000,6)-0.5)*0.999,1.1*sp.ones((1000,6))) ) )
    particleBoundaryObj = particleBoundary.AbsorbRectangular(gridObj,part)  
    print timeit.timeit('particleBoundaryObj.indexInside()','from __main__ import particleBoundaryObj', number = 1)/1*1000
          
    print 'Boundary calculate interaction point for 1000 particles (rectangular)'
    gridObj = grid.Grid([nx,ny],[lx,ly],'rectangular')
    part = particles.Particles('electrons','variableWeight')
    part.setParticleData(sp.vstack( ( (sp.rand(nParticles-1000,6)-0.5)*0.999,1.1*sp.ones((1000,6))) ) )
    particleBoundaryObj = particleBoundary.AbsorbRectangular(gridObj,part)  
    particleBoundaryObj.indexInside()
    particleBoundaryObj.saveAbsorbed()
    print timeit.timeit('particleBoundaryObj.calculateInteractionPoint()','from __main__ import particleBoundaryObj', number = 1)/1*1000
  
    print 'Boundary calculate normal vectors for 1000 particles (rectangular)'
    gridObj = grid.Grid([nx,ny],[lx,ly],'rectangular')
    part = particles.Particles('electrons','variableWeight')
    part.setParticleData(sp.vstack( ( (sp.rand(nParticles-1000,6)-0.5)*0.999,1.1*sp.ones((1000,6))) ) )
    particleBoundaryObj = particleBoundary.AbsorbRectangular(gridObj,part)  
    particleBoundaryObj.indexInside()
    particleBoundaryObj.saveAbsorbed()
    particleBoundaryObj.calculateInteractionPoint()
    print timeit.timeit('particleBoundaryObj.calculateNormalVectors()','from __main__ import particleBoundaryObj', number = 1)/1*1000  

    print 'Boundary check if particles are inside (elliptical)'
    gridObj = grid.Grid([nx,ny],[2*lx,ly],'elliptical')
    part = particles.Particles('electrons','variableWeight')
    part.setParticleData(sp.vstack( ( (sp.rand(nParticles-1000,6)-0.5)*0.999,2.1*sp.ones((1000,6))) ) )
    particleBoundaryObj = particleBoundary.AbsorbElliptical(gridObj,part)  
    print timeit.timeit('particleBoundaryObj.indexInside()','from __main__ import particleBoundaryObj', number = 100)/100*1000
          
    print 'Boundary calculate interaction point for 1000 particles (elliptical)'
    gridObj = grid.Grid([nx,ny],[2*lx,ly],'elliptical')
    part = particles.Particles('electrons','variableWeight')
    part.setParticleData(sp.vstack( ( (sp.rand(nParticles-1000,6)-0.5)*0.999,2.1*sp.ones((1000,6))) ) )
    particleBoundaryObj = particleBoundary.AbsorbElliptical(gridObj,part)  
    particleBoundaryObj.indexInside()
    particleBoundaryObj.saveAbsorbed()
    print timeit.timeit('particleBoundaryObj.calculateInteractionPoint()','from __main__ import particleBoundaryObj', number = 100)/100*1000
  
    print 'Boundary calculate normal vectors for 1000 particles (elliptical)'
    gridObj = grid.Grid([nx,ny],[2*lx,ly],'elliptical')
    part = particles.Particles('electrons','variableWeight')
    part.setParticleData(sp.vstack( ( (sp.rand(nParticles-1000,6)-0.5)*0.999,2.1*sp.ones((1000,6))) ) )
    particleBoundaryObj = particleBoundary.AbsorbElliptical(gridObj,part)  
    particleBoundaryObj.indexInside()
    particleBoundaryObj.saveAbsorbed()
    particleBoundaryObj.calculateInteractionPoint()
    print timeit.timeit('particleBoundaryObj.calculateNormalVectors()','from __main__ import particleBoundaryObj', number = 100)/100*1000  
     
    print 'Generate secondaries for 1000 particles (Furman)'
    gridObj = grid.Grid([nx,ny],[lx,ly],'rectangular')
    part = particles.Particles('electrons','variableWeight')
    part.setParticleData(sp.vstack( ( (sp.rand(nParticles-1000,6)-0.5)*0.999,1.01*sp.ones((1000,6))) ) )
    particleBoundaryObj = particleBoundary.AbsorbRectangular(gridObj,part)  
    particleBoundaryObj.indexInside()
    particleBoundaryObj.saveAbsorbed()
    particleBoundaryObj.calculateInteractionPoint()
    particleBoundaryObj.calculateNormalVectors()
    secElecEmitterObj = particleEmitter.FurmanEmitter(particleBoundaryObj,part)
    print timeit.timeit('secElecEmitterObj.generateSecondaries()','from __main__ import secElecEmitterObj', number = 1)/1*1000   
 
    print 'Generate secondaries for 1000 particles'
    gridObj = grid.Grid([nx,ny],[lx,ly],'rectangular')
    part = particles.Particles('electrons','variableWeight')
    part.setParticleData(sp.vstack( ( (sp.rand(nParticles-1000,6)-0.5)*0.999,1.01*sp.ones((1000,6))) ) )
    particleBoundaryObj = particleBoundary.AbsorbRectangular(gridObj,part)  
    particleBoundaryObj.indexInside()
    particleBoundaryObj.saveAbsorbed()
    particleBoundaryObj.calculateInteractionPoint()
    particleBoundaryObj.calculateNormalVectors()
    secElecEmitterObj = particleEmitter.SecElecEmitter(particleBoundaryObj,part)
    print timeit.timeit('secElecEmitterObj.generateSecondaries()','from __main__ import secElecEmitterObj', number = 1)/1*1000 
       
    print 'Add and remove 1000 particles'
    gridObj = grid.Grid([nx,ny],[lx,ly],'rectangular')
    part = particles.Particles('electrons','variableWeight')
    part.setParticleData(sp.vstack( ( (sp.rand(nParticles-1000,6)-0.5)*0.999,sp.ones((1000,6))) ) )
    particleBoundaryObj = particleBoundary.AbsorbRectangular(gridObj,part)  
    particleBoundaryObj.indexInside()
    particleBoundaryObj.saveAbsorbed()
    particleBoundaryObj.calculateInteractionPoint()
    particleBoundaryObj.calculateNormalVectors()
    isInside = part.getIsInside()
    secElecEmitterObj = particleEmitter.FurmanEmitter(particleBoundaryObj,part)
    secondaries = secElecEmitterObj.generateSecondaries()
    print timeit.timeit('part.addAndRemoveParticles(secondaries, isInside)','from __main__ import part;from __main__ import isInside;from __main__ import secondaries; from __main__ import particles', number = 1)/1*1000     

