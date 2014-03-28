#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy
cimport numpy
cimport cython
cimport specFun
cimport randomGen
cimport grid
cimport particles
cimport particleBoundary
from constants cimport *
from libc.stdlib cimport malloc, free


# Import some C methods.
cdef extern from "math.h":
    double sin(double x) nogil
    double cos(double x) nogil
    double exp(double x) nogil
    double sqrt(double x) nogil
    double asin(double x) nogil
    double acos(double x) nogil

cdef inline double abs(double x) nogil: 
    if x>=0.:
        return x
    else:
        return -x    
        
cdef inline double clip(double x, double xmin, double xmax)  nogil: 
    if x<=xmax and x>=xmin:
        return x
    elif x<xmin:
        return xmin
    else:
        return xmax

cpdef homoLoader(grid.Grid gridObj, particles.Particles particlesObj, particleBoundary.ParticleBoundary partBoundObj, 
                 unsigned int macroParticleCount, double weightInit, double vsigma = 10000.*doubleMinVal, 
                 unsigned int randomizeWeight = 0):
     
    cdef:
        unsigned int nCoords = particlesObj.getNCoords(), nCoordsm1 = nCoords-1
        unsigned int ii = 0
        double x, y
        double[:,:] particleData = numpy.zeros((macroParticleCount,nCoords), dtype=numpy.double)
        double lx = gridObj.getLx(), ly = gridObj.getLy()
 
    while ii < macroParticleCount:
        x = (randomGen.rand()-0.5)*lx
        y = (randomGen.rand()-0.5)*ly
        while not partBoundObj.isInside(x, y):                                 # Replace by particleBoundary!.
            x = (randomGen.rand()-0.5)*lx
            y = (randomGen.rand()-0.5)*ly 
        particleData[ii,0] = x
        particleData[ii,1] = y 
        particleData[ii,2] = randomGen.randn()*vsigma                   
        particleData[ii,3] = randomGen.randn()*vsigma 
        particleData[ii,4] = randomGen.randn()*vsigma    
        ii += 1
    
    if randomizeWeight == 0: 
        for ii in range(macroParticleCount):
            particleData[ii,nCoordsm1] = weightInit  
    else:
        for ii in range(macroParticleCount):
            particleData[ii,nCoordsm1] = weightInit*(0.5 + randomGen.rand())       
         
    particlesObj.setParticleData(particleData)
    


            
cdef class FurmanEmitter:
        
    cdef: 
        double p1EInf, p1Ehat, eEHat, w, p, e1, e2, p1RInf, eR, alpha, q, sigmaE
        double r, r1, r2, eHat0, t1, t2, t3, t4, s, deltaTSHat, particleMass
        unsigned int m, scaleSigmaE
        double[:] epsN, pSmallN      
        double[:,:] secondaries
        particles.Particles particlesObj
        particleBoundary.ParticleBoundary particleBoundaryObj
        double deltaERmax, theta0max
        unsigned int warningFlag00, warningFlag01
              
    def __init__(FurmanEmitter self, particleBoundaryObj, particlesObj, 
                 material='copper', double seyMax=-1., double reflec=-1., unsigned short scaleSigmaE = 0):
        
        self.particlesObj = particlesObj
        self.particleBoundaryObj = particleBoundaryObj
        if material=='stainless':
            self.stainlessMod(seyMax, reflec)    
        elif material=='copper':
            self.copperMod(seyMax, reflec)   
        else:
            raise ValueError('Material ' + '"' + str(material) + '" not implemented.')
            
        self.particleMass = self.particlesObj.getParticleMass()
        self.warningFlag00 = 0
        self.warningFlag01 = 0
        self.scaleSigmaE = scaleSigmaE
        
    def stainlessMod(FurmanEmitter self, double seyMax, double reflec):    
        # Model parameters stainless steel 
        self.p1EInf = 0.07; self.p1Ehat = 0.5; self.eEHat = 0.;
        self.w = 100.; self.p = 0.9; self.e1 = 0.26;  
        self.e2 = 2.; self.q = 0.4; self.sigmaE = 1.9;       
        self.p1RInf = 0.74; self.eR = 40.; self.r = 1.;
        self.r1 = 0.26 ; self.r2 = 2.; self.deltaTSHat = 1.22;
        self.s = 1.813; self.eHat0 = 310.; self.t3 = 0.7;   
        self.t4 = 1.; self.t1 = 0.66; self.t2 = 0.8;
        self.theta0max = 84.*pi/180;    self.deltaERmax = 0.99;
        
        self.m = 10         # m can't be changed just by setting here, as epsN and pSmallN are limited to 10.
        self.alpha = 1.     # alpha can't be changed just by setting here, as it is "hard coded" in some stuff below for speed.
        self.epsN = numpy.array([3.9, 6.2, 13., 8.8, 6.25, 2.25, 9.2, 5.3, 17.8, 10.], dtype=numpy.double)
        self.pSmallN = numpy.array([1.6, 2., 1.8, 4.7, 1.8, 2.4, 1.8, 1.8, 2.3, 1.8], dtype=numpy.double)
        
        if seyMax >= 0.5:  
            self.deltaTSHat = seyMax-self.p1RInf-self.p1EInf

        if reflec >= 0.:
            self.p1Ehat = reflec


    def copperMod(FurmanEmitter self, double seyMax, double reflec):    
        # Model parameters copper 
        self.p1EInf = 0.02; self.p1Ehat = 0.496; self.eEHat = 0.
        self.w = 60.86; self.p = 1.; self.e1 = 0.26  
        self.e2 = 2.; self.sigmaE = 2.;
        
        self.p1RInf = 0.2; self.eR = 0.041; self.r = 0.104
        self.q = 0.5; self.r1 = 0.26; self.r2 = 2.
        
        self.deltaTSHat = 1.8848; self.s = 1.54; self.eHat0 = 276.8
        self.t3 = 0.7; self.t4 = 1.; self.t1 = 0.66; self.t2 = 0.8
        
        self.theta0max = 84.*pi/180;    self.deltaERmax = 0.99;
        self.m = 10         # m can't be changed just by setting here, as epsN and pSmallN are limited to 10.
        self.alpha = 1.     # alpha can't be changed just by setting here, as it is "hard coded" in some stuff below for speed.
        self.epsN = numpy.array([2.5, 3.3, 2.5, 2.5, 2.8, 1.3, 1.5, 1.5, 1.5, 1.5], dtype=numpy.double)
        self.pSmallN = numpy.array([1.5, 1.75, 1., 3.75, 8.5, 11.5, 2.5, 3., 2.5, 3.], dtype=numpy.double)

        if seyMax >= 0.5:  
            self.deltaTSHat = seyMax-self.p1RInf-self.p1EInf

        if reflec >= 0.:
            self.p1Ehat = reflec
            
    def generateSecondaries(self):

        cdef: 
            double *absorbedParticles = &self.particleBoundaryObj.getAbsorbedParticles()[0,0]
            double *normalVectors = &self.particleBoundaryObj.getNormalVectors()[0,0]
            double deltaE0, deltaR0, eHat, deltaTS0, deltaTS, deltaTSPrime
            double pr, pNCumSum, theta0, u, aE, aR, p0, y, vSec, v0
            double thetaSec, cosThetaSec, sinThetaSec, phiSec, cosPhiSec, sinPhiSec
            double eFactor = 0.5*self.particleMass/elementary_charge
            double eFactorInv = 1./eFactor
            double deltaE, deltaR, temp, temp01, temp02
            unsigned int jj, ii, kk, currentSec, nSecSum
            unsigned int n0 = self.particleBoundaryObj.getAbsorbedMacroParticleCount()
            unsigned int nCoords = self.particlesObj.getNCoords()
                       
            double[::1] e0 = numpy.empty(n0, dtype=numpy.double)
            double[::1] eSec = numpy.empty(self.m, dtype=numpy.double)
            double[::1] thetaK = numpy.empty(self.m, dtype=numpy.double)
            double[::1] yK = numpy.empty(self.m, dtype=numpy.double)
            double[::1] pN = numpy.empty(self.m+1, dtype=numpy.double)         
            unsigned int[::1] nSec = numpy.empty(n0, dtype=numpy.uintc)
            double *secondaries


        for jj in range(n0):
            v0 = sqrt(absorbedParticles[nCoords*jj+2]*absorbedParticles[nCoords*jj+2] + 
                      absorbedParticles[nCoords*jj+3]*absorbedParticles[nCoords*jj+3] + 
                      absorbedParticles[nCoords*jj+4]*absorbedParticles[nCoords*jj+4])
            e0[jj] = eFactor*v0*v0
            theta0 = acos( (absorbedParticles[nCoords*jj+2]*normalVectors[2*jj] + 
                            absorbedParticles[nCoords*jj+3]*normalVectors[2*jj+1])/v0 )
            if theta0 > 0.5*pi:
                theta0 = pi - theta0
            theta0 = clip(theta0, 0., self.theta0max)
            
            deltaE0 = self.p1EInf + (self.p1Ehat-self.p1EInf)*exp(-((abs(e0[jj]-self.eEHat)/self.w)**self.p)/self.p)
            deltaE = deltaE0*(1 + self.e1*(1 - cos(theta0)**self.e2))
                         
            deltaR0 = self.p1RInf*(1-exp(-(e0[jj]/self.eR)**self.r))   
            deltaR = deltaR0*(1 + self.r1*(1 - cos(theta0)**self.r2)) 
                
            eHat = self.eHat0*(1 + self.t3*(1 - cos(theta0)**self.t4))       
            deltaTS0 = self.deltaTSHat*self.s*e0[jj]/eHat/(self.s-1+(e0[jj]/eHat)**self.s)  
            deltaTS = deltaTS0*(1 + self.t1*(1 - cos(theta0)**self.t2))
            
            temp = deltaE+deltaR
            if temp>=self.deltaERmax:
                deltaE = deltaE/temp*self.deltaERmax
                deltaR = deltaR/temp*self.deltaERmax
                if self.warningFlag00==0:
                    print 'WARNING: delta_E + delta_R >= delta_ERMax (~1). Will be set to delta_ERMax. \
                           This warning is suppressed from now on.'
                    self.warningFlag00 = 1
                     
            deltaTSPrime = deltaTS/(1-deltaE-deltaR)        
            if deltaTSPrime>=self.m:
                deltaTSPrime = self.m
                if self.warningFlag01==0:
                    print 'WARNING: delta_TS^Prime >= m (most likely m=10). Will be set to m. \
                           This warning is suppressed from now on.'
                    self.warningFlag01 = 1  
            elif deltaTSPrime<0:
                deltaTSPrime = 0
            
            pr = deltaTSPrime/self.m
            for ii in range(self.m+1):
                pN[ii] = specFun.binom(ii,pr,self.m)*(1-deltaE-deltaR)
            pN[1] += deltaE + deltaR
            pNCumSum = 0
            nSec[jj] = self.m
            u = randomGen.rand()
            for ii in range(self.m):
                pNCumSum += pN[ii]
                if u<=pNCumSum:
                    nSec[jj] = ii
                    break

        nSecSum = 0
        for ii in range(n0):
            nSecSum += nSec[ii]
     
    
        if nSecSum > 0:              
            self.secondaries = numpy.empty((nSecSum, self.particlesObj.getNCoords()), dtype=numpy.double)
            secondaries = &self.secondaries[0,0]
              
            currentSec = 0
            for jj in range(n0):
                if nSec[jj] == 0:
                    pass
                elif nSec[jj] == 1:
                    aE = deltaE/pN[1]
                    aR = deltaR/pN[1]
                      
                    u = randomGen.rand()

                    if u<aE:
                        if self.scaleSigmaE==0:
                            eSec[0] = e0[jj] - abs(self.sigmaE*randomGen.randn())
                        else: 
                            eSec[0] = e0[jj] - abs(self.sigmaE*sqrt(e0[jj]/300)*randomGen.randn())
                        if eSec[0]<0.:
                            eSec[0] = randomGen.rand()*e0[jj] 
                    elif u < aE + aR:
                        eSec[0] = e0[jj]*randomGen.rand()**(1./(1.+self.q))
                    else:              
                        eSec[0] = self.epsN[0]*specFun.gammaincinv(self.pSmallN[0],randomGen.rand()*
                                                                   specFun.gammainc(self.pSmallN[0],e0[jj]/self.epsN[0]))
                else:   
                    p0 = specFun.gammainc(nSec[jj]*self.pSmallN[nSec[jj]-1],e0[jj]/self.epsN[nSec[jj]-1])
                       
                    for ii in range(nSec[jj]-1):
                        thetaK[ii] = asin(sqrt(specFun.betaincinv(self.pSmallN[nSec[jj]-1]*
                                              (nSec[jj]-ii-1),self.pSmallN[nSec[jj]-1],randomGen.rand())))
                    y = sqrt(specFun.gammaincinv(nSec[jj]*self.pSmallN[nSec[jj]-1],randomGen.rand()*p0))     
                    yK[0] = y*cos(thetaK[0])
                    yK[nSec[jj]-1] = y
                    for ii in range(nSec[jj]-1):
                        yK[nSec[jj]-1] *= sin(thetaK[ii])
                    for ii in range(1,nSec[jj]-1):
                        yK[ii] = y*cos(thetaK[ii])
                        for kk in range(ii):
                            yK[ii] *= sin(thetaK[kk]) 
                    for ii in range(nSec[jj]):
                        eSec[ii] = self.epsN[nSec[jj]-1]*(yK[ii]*yK[ii])                    
                
                eSum = 0    
                for ii in range(nSec[jj]):
                    eSum+=eSec[ii]
                for ii in range(nSec[jj]):
                    kk = nCoords*currentSec 
                    secondaries[kk] = absorbedParticles[nCoords*jj]
                    secondaries[kk+1] = absorbedParticles[nCoords*jj+1]
                    secondaries[kk+5] = absorbedParticles[nCoords*jj+5]
        
                    vSec = sqrt(eFactorInv*eSec[ii])           
                    thetaSec = asin(randomGen.rand())
                    cosThetaSec = cos(thetaSec)
                    sinThetaSec = sin(thetaSec)
                    phiSec = randomGen.rand()*pi*2.
                    cosPhiSec = cos(phiSec)
                    sinPhiSec = sin(phiSec)
                    temp01 = cosThetaSec*normalVectors[2*jj] - sinThetaSec*normalVectors[2*jj+1]
                    temp02 = sinThetaSec*normalVectors[2*jj] + cosThetaSec*normalVectors[2*jj+1]
                    secondaries[kk+2] = vSec*( (cosPhiSec + normalVectors[2*jj]*normalVectors[2*jj]*(1-cosPhiSec))*temp01 +
                                                normalVectors[2*jj]*normalVectors[2*jj+1]*(1-cosPhiSec)*temp02 )
                    secondaries[kk+3] = vSec*( (cosPhiSec + normalVectors[2*jj+1]*normalVectors[2*jj+1]*(1-cosPhiSec))*temp02 +
                                                normalVectors[2*jj]*normalVectors[2*jj+1]*(1-cosPhiSec)*temp01 )
                    secondaries[kk+4] = vSec*( -sinPhiSec*normalVectors[2*jj+1]*temp01 + sinPhiSec*normalVectors[2*jj]*temp02 )
                    currentSec += 1    
                
            return self.secondaries
          
        else:
            return numpy.empty((0,nCoords), dtype=numpy.double)
        
    def getEmittedParticles(self):
        return self.secondaries

    
    
    
cdef class SecElecEmitter:
          
    # Import instance variables.
    cdef: 
        double p1EInf, p1Ehat, eEHat, w, p, e1, e2, p1RInf, eR, q, sigmaETS, muETS
        double r, r1, r2, eHat0, t1, t2, t3, t4, s, deltaTSHat, particleMass
        double theta0max, deltaERmax
        particles.Particles particlesObj
        particleBoundary.ParticleBoundary particleBoundaryObj
                
    def __init__(self, particleBoundaryObj, particlesObj, double seyMax=-1., double reflec=-1.):
        
        self.particlesObj = particlesObj
        self.particleBoundaryObj = particleBoundaryObj
        self.stainlessMod(seyMax, reflec)    
         
        self.particleMass = self.particlesObj.getParticleMass()
         
    def stainlessMod(self, double seyMax, double reflec):    
        # Model parameters stainless steel 
        self.p1EInf = 0.07; self.p1Ehat = 0.5; self.eEHat = 0.;
        self.w = 100.; self.p = 0.9; self.e1 = 0.26;  
        self.e2 = 2.; self.q = 0.4;      
        self.p1RInf = 0.74; self.eR = 40.; self.r = 1.
        self.r1 = 0.26 ; self.r2 = 2.; self.deltaTSHat = 1.22 
        self.s = 1.813; self.eHat0 = 310.; self.t3 = 0.7   
        self.t4 = 1.; self.t1 = 0.66; self.t2 = 0.8
        self.sigmaETS = 1.041;  self.muETS = 2.456
        self.theta0max = 84.*pi/180.;   self.deltaERmax = 0.99;
 
         
        if seyMax >= 0.5:  
            self.deltaTSHat = seyMax-self.p1RInf-self.p1EInf
            if self.deltaTSHat<0.:
                self.deltaTSHat = 1.e-6
                print 'WARNING: Maximum SEY too small. Setting true SEY to zero.'
 
        if reflec >= 0.:
            self.p1Ehat = reflec
    
  
    def generateSecondaries(self):
    
        # Some local variables
        cdef: 
            double *absorbedParticles = &self.particleBoundaryObj.getAbsorbedParticles()[0,0]
            double *normalVectors = &self.particleBoundaryObj.getNormalVectors()[0,0]
            
            double deltaE0, deltaR0, eHat, deltaTS0, deltaTS, deltaE, deltaR
            double theta0, u, aE, aR, p0, vSec, v0, nSecSum = 0, vFac
            double thetaSec, cosThetaSec, sinThetaSec, phiSec, cosPhiSec, sinPh
            double eFactor = 0.5*self.particleMass/elementary_charge
            double eFactorInv = 1./eFactor
            unsigned int jj, ii, kk, currentSec, deltaTSFloor
            unsigned int n0 = self.particleBoundaryObj.getAbsorbedMacroParticleCount()
            unsigned int nCoords = self.particlesObj.getNCoords()
                
            double[:] e0 = numpy.empty(n0, dtype=numpy.double)           
            unsigned int[:] nSecE = numpy.zeros(n0, dtype=numpy.uintc)
            unsigned int[:] nSecR = numpy.zeros(n0, dtype=numpy.uintc)
            unsigned int[:] nSecTS = numpy.empty(n0, dtype=numpy.uintc)
            
            double[:] eSec
            double[:,:] secondariesBuff
            double *secondaries
    
        
        for jj in range(n0):
                
            v0 = sqrt(absorbedParticles[nCoords*jj+2]*absorbedParticles[nCoords*jj+2] + 
                      absorbedParticles[nCoords*jj+3]*absorbedParticles[nCoords*jj+3] + 
                      absorbedParticles[nCoords*jj+4]*absorbedParticles[nCoords*jj+4])
            e0[jj] = eFactor*v0*v0
            theta0 = acos( (absorbedParticles[nCoords*jj+2]*normalVectors[2*jj] + 
                            absorbedParticles[nCoords*jj+3]*normalVectors[2*jj+1])/v0 )
            if theta0 > 0.5*pi:
                theta0 = pi - theta0
            theta0 = clip(theta0, 0., self.theta0max)
    
            deltaE0 = self.p1EInf + (self.p1Ehat-self.p1EInf)*exp(-(abs(e0[jj]-self.eEHat)/self.w)**self.p/self.p)
            deltaE = deltaE0*(1 + self.e1*(1 - cos(theta0)**self.e2))
                         
            deltaR0 = self.p1RInf*(1-exp(-(e0[jj]/self.eR)**self.r))   
            deltaR = deltaR0*(1 + self.r1*(1 - cos(theta0)**self.r2)) 
                     
            eHat = self.eHat0*(1 + self.t3*(1 - cos(theta0)**self.t4))            
            deltaTS0 = self.deltaTSHat*self.s*e0[jj]/eHat/(self.s-1+(e0[jj]/eHat)**self.s)  
            deltaTS = deltaTS0*(1 + self.t1*(1 - cos(theta0)**self.t2))

            temp = deltaE+deltaR
            if temp>=self.deltaERmax:
                deltaE = deltaE/temp*self.deltaERmax
                deltaR = deltaR/temp*self.deltaERmax
                                           
            if randomGen.rand()<=deltaE:
                nSecE[jj] = 1
            if randomGen.rand()<=deltaR:
                nSecR[jj] = 1
            deltaTSFloor = <unsigned int> deltaTS
            if randomGen.rand()<=(deltaTS-deltaTSFloor):
                nSecTS[jj] = deltaTSFloor + 1
            else:
                nSecTS[jj] = deltaTSFloor

        for ii in range(n0):
            nSecSum += nSecE[ii] + nSecR[ii] + nSecTS[ii]
              
        if nSecSum > 0:              
            secondariesBuff = numpy.empty((nSecSum, nCoords), dtype=numpy.double)
            secondaries = &secondariesBuff[0,0]
            eSec = numpy.empty(numpy.amax(nSecTS), dtype=numpy.double) 
              
            currentSec = 0
            for jj in range(n0):
                if nSecE[jj] == 1:
                    kk = nCoords*currentSec
                    secondaries[kk] = absorbedParticles[nCoords*jj]
                    secondaries[kk+1] = absorbedParticles[nCoords*jj+1]
                    secondaries[kk+5] = absorbedParticles[nCoords*jj+5]
                    secondaries[kk+2] = -absorbedParticles[nCoords*jj+2]
                    secondaries[kk+3] = -absorbedParticles[nCoords*jj+3]
                    secondaries[kk+4] = -absorbedParticles[nCoords*jj+4]
                    currentSec += 1    
                if nSecR[jj] == 1:
                    vFac = sqrt(randomGen.rand()**(1./(1.+self.q)))
                    kk = nCoords*currentSec
                    secondaries[kk] = absorbedParticles[nCoords*jj]
                    secondaries[kk+1] = absorbedParticles[nCoords*jj+1]
                    secondaries[kk+5] = absorbedParticles[nCoords*jj+5]
                    secondaries[kk+2] = -absorbedParticles[nCoords*jj+2]*vFac
                    secondaries[kk+3] = -absorbedParticles[nCoords*jj+3]*vFac
                    secondaries[kk+4] = -absorbedParticles[nCoords*jj+4]*vFac
                    currentSec += 1      
                if nSecTS[jj] > 0:
                    eSum = 0.
                    for ii in range(nSecTS[jj]):
                        eSec[ii] = exp(self.sigmaETS*randomGen.randn()+self.muETS)
                        eSum += eSec[ii]   
                    
                    if eSum>e0[jj]:               
                        for ii in range(nSecTS[jj]):
                            eSec[ii] = randomGen.rand()*e0[jj]/nSecTS[jj]
                    for ii in range(nSecTS[jj]):
                        kk = nCoords*currentSec
                        secondaries[kk] = absorbedParticles[nCoords*jj]
                        secondaries[kk+1] = absorbedParticles[nCoords*jj+1]
                        secondaries[kk+5] = absorbedParticles[nCoords*jj+5]
                        vSec = sqrt(eFactorInv*eSec[ii])           
                        thetaSec = asin(randomGen.rand())
                        cosThetaSec = cos(thetaSec)
                        sinThetaSec = sin(thetaSec)
                        phiSec = randomGen.rand()*pi*2.
                        cosPhiSec = cos(phiSec)
                        sinPhiSec = sin(phiSec)
                        temp01 = cosThetaSec*normalVectors[2*jj] - sinThetaSec*normalVectors[2*jj+1]
                        temp02 = sinThetaSec*normalVectors[2*jj] + cosThetaSec*normalVectors[2*jj+1]
                        secondaries[kk+2] = vSec * ( (cosPhiSec + normalVectors[2*jj]*normalVectors[2*jj]*(1-cosPhiSec))*temp01 +
                                                      normalVectors[2*jj]*normalVectors[2*jj+1]*(1-cosPhiSec)*temp02 )
                        secondaries[kk+3] = vSec * ( (cosPhiSec + 
                                                      normalVectors[2*jj+1]*normalVectors[2*jj+1]*(1-cosPhiSec))*temp02 +
                                                      normalVectors[2*jj]*normalVectors[2*jj+1]*(1-cosPhiSec)*temp01 )
                        secondaries[kk+4] = vSec * ( -sinPhiSec*normalVectors[2*jj+1]*temp01 + 
                                                      sinPhiSec*normalVectors[2*jj]*temp02 )
                        currentSec += 1
            return secondariesBuff               
        else:
            return numpy.empty((0,nCoords), dtype=numpy.double)



'''
Currently only for testing.
'''
cdef class SimpleSecEmitter:

    cdef: 
        particles.Particles particlesObj
        particleBoundary.ParticleBoundary particleBoundaryObj
                
    def __init__(self, particleBoundary.ParticleBoundary particleBoundaryObj, particles.Particles particlesObj, 
                 double seyMax=-1., double reflec=-1.):
        
        self.particlesObj = particlesObj
        self.particleBoundaryObj = particleBoundaryObj 
        
    def generateSecondaries(self):
    
        # Some local variables
        cdef: 
            double *absorbedParticles = &self.particleBoundaryObj.getAbsorbedParticles()[0,0]
            double *normalVectors = &self.particleBoundaryObj.getNormalVectors()[0,0]
            
            unsigned int jj, ii, kk, currentSec, deltaTSFloor
            unsigned int n0 = self.particleBoundaryObj.getAbsorbedMacroParticleCount()
            unsigned int nCoords = self.particlesObj.getNCoords()

            double particleMass = 9.10938291e-31
            double elementary_charge = 1.602176565e-19
            double eFactor = 0.5*particleMass/elementary_charge
            double s = 1.813
            double eHat = 310
            double nSecSum = 0
                            
            double[:] e0 = numpy.empty(n0, dtype=numpy.double)
            unsigned int[:] nSecE = numpy.zeros(n0, dtype=numpy.uintc)
            unsigned int[:] nSecR = numpy.zeros(n0, dtype=numpy.uintc)
            unsigned int[:] nSecTS = numpy.empty(n0, dtype=numpy.uintc)
            
            double[:] eSec
            double[:,:] secondariesBuff
            double *secondaries
    
        
        for jj in range(n0):
                
            v0 = sqrt(absorbedParticles[nCoords*jj+2]*absorbedParticles[nCoords*jj+2] + 
                      absorbedParticles[nCoords*jj+3]*absorbedParticles[nCoords*jj+3] + 
                      absorbedParticles[nCoords*jj+4]*absorbedParticles[nCoords*jj+4])
            e0[jj] = eFactor*v0*v0
    
            deltaE = 0.05
                         
            deltaR = 0.5
                              
            deltaTS = 1.4*s*e0[jj]/eHat/(s-1+(e0[jj]/eHat)**s)  

            if randomGen.rand()<=deltaE:
                nSecE[jj] = 1
            if randomGen.rand()<=deltaR:
                nSecR[jj] = 1

            if randomGen.rand()<=(deltaTS-(<unsigned int> deltaTS)):
                nSecTS[jj] = <unsigned int> (deltaTS+1.)
            else:
                nSecTS[jj] = <unsigned int> (deltaTS)

        for ii in range(n0):
            nSecSum += nSecE[ii] + nSecR[ii] + nSecTS[ii]
              
        if nSecSum > 0:              
            secondariesBuff= numpy.empty((nSecSum, nCoords), dtype=numpy.double)
            secondaries = &secondariesBuff[0,0]
            eSec = numpy.empty(numpy.amax(nSecTS), dtype=numpy.double)
              
            currentSec = 0
            for jj in range(n0):
                if nSecE[jj] == 1:
                    kk = nCoords*currentSec
                    secondaries[kk] = absorbedParticles[nCoords*jj]
                    secondaries[kk+1] = absorbedParticles[nCoords*jj+1]
                    secondaries[kk+5] = absorbedParticles[nCoords*jj+5]
                    secondaries[kk+2] = -absorbedParticles[nCoords*jj+2]
                    secondaries[kk+3] = -absorbedParticles[nCoords*jj+3]
                    secondaries[kk+4] = -absorbedParticles[nCoords*jj+4]
                    currentSec += 1    
                if nSecR[jj] == 1:
                    vFac = sqrt(randomGen.rand())
                    kk = nCoords*currentSec
                    secondaries[kk] = absorbedParticles[nCoords*jj]
                    secondaries[kk+1] = absorbedParticles[nCoords*jj+1]
                    secondaries[kk+5] = absorbedParticles[nCoords*jj+5]
                    secondaries[kk+2] = -absorbedParticles[nCoords*jj+2]*vFac
                    secondaries[kk+3] = -absorbedParticles[nCoords*jj+3]*vFac
                    secondaries[kk+4] = -absorbedParticles[nCoords*jj+4]*vFac
                    currentSec += 1      
                if nSecTS[jj] > 0:              
                    for ii in range(nSecTS[jj]):
                        eSec[ii] = randomGen.rand()*5
                    for ii in range(nSecTS[jj]):
                        kk = nCoords*currentSec
                        secondaries[kk] = absorbedParticles[nCoords*jj]
                        secondaries[kk+1] = absorbedParticles[nCoords*jj+1]
                        secondaries[kk+5] = absorbedParticles[nCoords*jj+5]
                        vFac = sqrt(eSec[ii]/e0[jj])           
                        secondaries[kk+2] = -absorbedParticles[nCoords*jj+2]*vFac
                        secondaries[kk+3] = -absorbedParticles[nCoords*jj+3]*vFac
                        secondaries[kk+4] = -absorbedParticles[nCoords*jj+4]*vFac
                        currentSec += 1
            return secondariesBuff               
        else:
            return numpy.empty((0,nCoords), dtype=numpy.double)
            
            
#'''
#Currently only for testing.
#'''
#cdef class TwoEnergySecEmitter:

#    cdef: 
#        particles.Particles particlesObj
#        particleBoundary.ParticleBoundary particleBoundaryObj
#                
#    def __init__(TwoEnergySecEmitter self, particleBoundary.ParticleBoundary particleBoundaryObj, particles.Particles particlesObj, 
#                 double seyMax=-1., double reflec=-1.):
#        
#        self.particlesObj = particlesObj
#        self.particleBoundaryObj = particleBoundaryObj 
#        
#    def generateSecondaries(self):
#    
#        # Some local variables
#        cdef: 
#            double *absorbedParticles = &self.particleBoundaryObj.getAbsorbedParticles()[0,0]
#            double *normalVectors = &self.particleBoundaryObj.getNormalVectors()[0,0]
#            
#            unsigned int jj, ii, kk, currentSec, deltaTSFloor
#            unsigned int n0 = self.particleBoundaryObj.getAbsorbedMacroParticleCount()
#            unsigned int nCoords = self.particlesObj.getNCoords()

#            double particleMass = self.particlesObj.getParticleMass()
#            double eFactor = 0.5*particleMass/elementary_charge
#            double s = 1.813
#            double eHat = 310
#            double nSecSum = 0
#                            
#            double[:] e0 = numpy.empty(n0, dtype=numpy.double)
#            unsigned int[:] nSecE = numpy.zeros(n0, dtype=numpy.uintc)
#            unsigned int[:] nSecTS = numpy.empty(n0, dtype=numpy.uintc)
#            
#            double[:] eSec
#            double[:,:] secondariesBuff
#            double *secondaries
#    
#        
#        for jj in range(n0):
#                
#            v0 = sqrt(absorbedParticles[nCoords*jj+2]*absorbedParticles[nCoords*jj+2] + 
#                      absorbedParticles[nCoords*jj+3]*absorbedParticles[nCoords*jj+3] + 
#                      absorbedParticles[nCoords*jj+4]*absorbedParticles[nCoords*jj+4])
#            e0[jj] = eFactor*v0*v0
#    
#            deltaE = 0.05
#                                      
#            deltaTS = 2.*s*e0[jj]/eHat/(s-1+(e0[jj]/eHat)**s)  

#            if randomGen.rand()<=deltaE:
#                nSecE[jj] = 1

#            if randomGen.rand()<=(deltaTS-(<unsigned int> deltaTS)):
#                nSecTS[jj] = <unsigned int> (deltaTS+1.)
#            else:
#                nSecTS[jj] = <unsigned int> (deltaTS)

#        for ii in range(n0):
#            nSecSum += nSecE[ii] + nSecR[ii] + nSecTS[ii]
#              
#        if nSecSum > 0:              
#            secondariesBuff= numpy.empty((nSecSum, nCoords), dtype=numpy.double)
#            secondaries = &secondariesBuff[0,0]
#            eSec = numpy.empty(numpy.amax(nSecTS), dtype=numpy.double)
#              
#            currentSec = 0
#            for jj in range(n0):
#                if nSecE[jj] == 1:
#                    kk = nCoords*currentSec
#                    secondaries[kk] = absorbedParticles[nCoords*jj]
#                    secondaries[kk+1] = absorbedParticles[nCoords*jj+1]
#                    secondaries[kk+5] = absorbedParticles[nCoords*jj+5]
#                    secondaries[kk+2] = -absorbedParticles[nCoords*jj+2]
#                    secondaries[kk+3] = -absorbedParticles[nCoords*jj+3]
#                    secondaries[kk+4] = -absorbedParticles[nCoords*jj+4]
#                    currentSec += 1    
#                if nSecR[jj] == 1:
#                    vFac = sqrt(randomGen.rand())
#                    kk = nCoords*currentSec
#                    secondaries[kk] = absorbedParticles[nCoords*jj]
#                    secondaries[kk+1] = absorbedParticles[nCoords*jj+1]
#                    secondaries[kk+5] = absorbedParticles[nCoords*jj+5]
#                    secondaries[kk+2] = -absorbedParticles[nCoords*jj+2]*vFac
#                    secondaries[kk+3] = -absorbedParticles[nCoords*jj+3]*vFac
#                    secondaries[kk+4] = -absorbedParticles[nCoords*jj+4]*vFac
#                    currentSec += 1      
#                if nSecTS[jj] > 0:              
#                    for ii in range(nSecTS[jj]):
#                        eSec[ii] = randomGen.rand()*5
#                    for ii in range(nSecTS[jj]):
#                        kk = nCoords*currentSec
#                        secondaries[kk] = absorbedParticles[nCoords*jj]
#                        secondaries[kk+1] = absorbedParticles[nCoords*jj+1]
#                        secondaries[kk+5] = absorbedParticles[nCoords*jj+5]
#                        vFac = sqrt(eSec[ii]/e0[jj])           
#                        secondaries[kk+2] = -absorbedParticles[nCoords*jj+2]*vFac
#                        secondaries[kk+3] = -absorbedParticles[nCoords*jj+3]*vFac
#                        secondaries[kk+4] = -absorbedParticles[nCoords*jj+4]*vFac
#                        currentSec += 1
#            return secondariesBuff               
#        else:
#            return numpy.empty((0,nCoords), dtype=numpy.double)
