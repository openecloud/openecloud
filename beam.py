'''
Created on Jun 7, 2013

@author: ohaas
'''


import scipy as sp
import scipy.constants as spc
import scipy.stats as sps


class LHCBeam:
    
    def __init__(self, gridObj, dt, nParticles=1.e11, tBunchSpacing=25.e-9):
        
        self.gridObj = gridObj
        self.nx = gridObj.getNxExt()
        self.ny = gridObj.getNyExt()
        self.np = gridObj.getNpExt()
        self.lx = gridObj.getLxExt()
        self.ly = gridObj.getLyExt()  
        self.dx = gridObj.getDx()
        self.dy = gridObj.getDy()
        self.dt = dt
        
        self.nParticles = nParticles
        self.charge = spc.elementary_charge
        self.beamVelocity = spc.c
        self.circumference = 6900
        self.tRevolution = self.circumference/self.beamVelocity
        self.radiusSigma = 0.001
        self.radiusLimitSigma = 5
        self.xBeamCenter = 0.
        self.yBeamCenter = 0.    
        self.tBunchSpacing = 25.e-9
        self.bunchLengthSigma = 0.1
        self.tBunchLengthSigma = self.bunchLengthSigma/self.beamVelocity
        self.bunchLengthLimitSigma = 5
        self.nBatches = 4
        self.nBunchesPerBatch = 72
        self.tBatchLength = self.nBunchesPerBatch*self.tBunchSpacing
        self.nEmptyBunches = 8
        self.tBatchSpacing = self.nEmptyBunches*self.tBunchSpacing
        self.tTrainLength = (self.nBatches*self.nBunchesPerBatch + (self.nBatches-1)*self.nEmptyBunches) * self.tBunchSpacing
        
        self.qTransversalProfile = sp.zeros(self.np)
    
        xMesh = self.gridObj.getXMesh()
        yMesh = self.gridObj.getYMesh()
        xCoords = sp.tile(xMesh,self.ny)
        yCoords = sp.reshape(sp.tile(yMesh,(self.nx,1)).transpose(),self.np)
        beamPoints = ((self.radiusLimitSigma*self.radiusSigma+max([self.dx,self.dy]))**2 - xCoords**2 - yCoords**2) > 0
        self.qTransversalProfile[beamPoints] = ( (
                                            sps.norm.cdf(sp.clip((xCoords-self.xBeamCenter+self.dx/2.)/self.radiusSigma,-self.radiusLimitSigma,self.radiusLimitSigma))-
                                            sps.norm.cdf(sp.clip((xCoords-self.xBeamCenter-self.dx/2.)/self.radiusSigma,-self.radiusLimitSigma,self.radiusLimitSigma))
                                            )*
                                           (
                                            sps.norm.cdf(sp.clip((yCoords-self.yBeamCenter+self.dy/2.)/self.radiusSigma,-self.radiusLimitSigma,self.radiusLimitSigma))-
                                            sps.norm.cdf(sp.clip((yCoords-self.yBeamCenter-self.dy/2.)/self.radiusSigma,-self.radiusLimitSigma,self.radiusLimitSigma))
                                            )
                                           )[beamPoints]
        self.qTransversalProfile /= sp.sum(self.qTransversalProfile)   
        
    def getTransversalProfile(self):
        return self.qTransversalProfile    
    
    def getCharge(self, t):
        conditions = (
                      sp.mod(t,self.tBunchSpacing)<2*self.bunchLengthLimitSigma*self.tBunchLengthSigma and
                      sp.mod(t,self.tBatchLength+self.tBatchSpacing)<self.tBatchLength and
                      sp.mod(t,self.tRevolution)<self.tTrainLength
                      )
        if conditions:
            tRed = sp.mod(sp.mod(t,self.tRevolution),self.tBunchSpacing)
            temp = sps.norm.cdf(sp.clip((tRed+self.dt/2.)/self.tBunchLengthSigma-self.bunchLengthLimitSigma,-self.bunchLengthLimitSigma,self.bunchLengthLimitSigma))-sps.norm.cdf(sp.clip((tRed-self.dt/2.)/self.tBunchLengthSigma-self.bunchLengthLimitSigma,-self.bunchLengthLimitSigma,self.bunchLengthLimitSigma))              
            return temp*self.nParticles/self.dt/self.beamVelocity*self.charge*self.qTransversalProfile
        else:
            return 0
        
class LHCBeamEndless:
    
    def __init__(self, gridObj, dt, nParticles=1.e11, tBunchSpacing=25.e-9):
        
        self.gridObj = gridObj
        self.nx = gridObj.getNxExt()
        self.ny = gridObj.getNyExt()
        self.np = gridObj.getNpExt()
        self.lx = gridObj.getLxExt()
        self.ly = gridObj.getLyExt()  
        self.dx = gridObj.getDx()
        self.dy = gridObj.getDy()
        self.dt = dt
        
        self.nParticles = nParticles
        self.charge = spc.elementary_charge
        self.beamVelocity = spc.c
        self.circumference = 6900
        self.radiusSigma = 0.002
        self.radiusLimitSigma = 5
        self.xBeamCenter = 0.
        self.yBeamCenter = 0.    
        self.tBunchSpacing = tBunchSpacing
        self.bunchLengthSigma = 0.1
        self.tBunchLengthSigma = self.bunchLengthSigma/self.beamVelocity
        self.bunchLengthLimitSigma = 5
        
        self.qTransversalProfile = sp.zeros(self.np)
    
        xMesh = self.gridObj.getXMesh()
        yMesh = self.gridObj.getYMesh()
        xCoords = sp.tile(xMesh,self.ny)
        yCoords = sp.reshape(sp.tile(yMesh,(self.nx,1)).transpose(),self.np)
        beamPoints = ((self.radiusLimitSigma*self.radiusSigma+max([self.dx,self.dy]))**2 - xCoords**2 - yCoords**2) > 0
        self.qTransversalProfile[beamPoints] = ( (
                                            sps.norm.cdf(sp.clip((xCoords-self.xBeamCenter+self.dx/2.)/self.radiusSigma,-self.radiusLimitSigma,self.radiusLimitSigma))-
                                            sps.norm.cdf(sp.clip((xCoords-self.xBeamCenter-self.dx/2.)/self.radiusSigma,-self.radiusLimitSigma,self.radiusLimitSigma))
                                            )*
                                           (
                                            sps.norm.cdf(sp.clip((yCoords-self.yBeamCenter+self.dy/2.)/self.radiusSigma,-self.radiusLimitSigma,self.radiusLimitSigma))-
                                            sps.norm.cdf(sp.clip((yCoords-self.yBeamCenter-self.dy/2.)/self.radiusSigma,-self.radiusLimitSigma,self.radiusLimitSigma))
                                            )
                                           )[beamPoints]
        self.qTransversalProfile /= sp.sum(self.qTransversalProfile)   
        
    def getTransversalProfile(self):
        return self.qTransversalProfile    
    
    def getCharge(self, t):
        conditions = sp.mod(t,self.tBunchSpacing)<2*self.bunchLengthLimitSigma*self.tBunchLengthSigma
        if conditions:
            tRed = sp.mod(t,self.tBunchSpacing)
            temp = sps.norm.cdf(sp.clip((tRed+self.dt/2.)/self.tBunchLengthSigma-self.bunchLengthLimitSigma,-self.bunchLengthLimitSigma,self.bunchLengthLimitSigma))-sps.norm.cdf(sp.clip((tRed-self.dt/2.)/self.tBunchLengthSigma-self.bunchLengthLimitSigma,-self.bunchLengthLimitSigma,self.bunchLengthLimitSigma))              
            return temp*self.nParticles/self.dt/self.beamVelocity*self.charge*self.qTransversalProfile
        else:
            return 0
