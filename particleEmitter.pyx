#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy
cimport numpy
cimport cython
cimport gslWrap
from constants cimport *


# Import some C methods.
cdef extern from "math.h":
    double sin(double x)     
    double cos(double x)
    double exp(double x)
    double sqrt(double x)
    double asin(double x)
    double acos(double x)

cdef inline double abs(double x): return x if x >= 0. else -x
cdef inline double clip(double x, double xmin, double xmax): 
    if x<=xmax and x>=xmin:
        return x
    elif x<xmin:
        return xmin
    else:
        return xmax

cpdef homoLoader(gridObj, particlesObj, unsigned int macroParticleCount, double weightInit):
     
    cdef:
        unsigned int nCoords = particlesObj.getNCoords(), nCoordsm1 = nCoords-1
        unsigned int ii = 0
        double x, y
        numpy.ndarray particleDataNumpy = numpy.zeros((macroParticleCount,nCoords), dtype=numpy.double)
        double[:,:] particleData = particleDataNumpy
        object gridBoundaryObj = gridObj.getGridBoundaryObj()
        double lx = gridObj.getLx(), ly = gridObj.getLy()
 
    while ii < macroParticleCount:
        x = (gslWrap.rand()-0.5)*lx
        y = (gslWrap.rand()-0.5)*ly
        while not gridBoundaryObj.isInside(x, y):
            x = (gslWrap.rand()-0.5)*lx
            y = (gslWrap.rand()-0.5)*ly 
        particleData[ii,0] = x
        particleData[ii,1] = y     
        ii += 1
     
    for ii in range(macroParticleCount):
        particleData[ii,nCoordsm1] = weightInit       
         
    particlesObj.setParticleData(particleDataNumpy)
    


            
cdef class FurmanEmitter:
        
    # Import instance variables.
    cdef: 
        double p1EInf, p1Ehat, eEHat, w, p, e1, e2, p1RInf, eR, alpha, q, sigmaE
        double r, r1, r2, eHat0, t1, t2, t3, t4, s, deltaTSHat, particleMass
        numpy.ndarray epsN, pSmallN
        unsigned int m, warningFlag, scaleSigmaE
        numpy.ndarray secondaries
        object particlesObj, particleBoundaryObj
        double deltaERmax, theta0max
        unsigned int warningFlag00, warningFlag01
              
    def __init__(self, particleBoundaryObj, particlesObj, material='stainless', seyMax=None, reflec=None, scaleSigmaE=0):
        
        self.particlesObj = particlesObj
        self.particleBoundaryObj = particleBoundaryObj
        if material=='stainless':
            self.stainlessMod(seyMax, reflec)    
        elif material=='copper':
            self.copperMod(seyMax, reflec)   
        elif material=='copperVSim':
            self.copperModVSim(seyMax, reflec)   
        else:
            raise ValueError('Material ' + '"' + material + '" not implemented.')
            
        self.particleMass = self.particlesObj.getParticleMass()
        self.warningFlag00 = 0
        self.warningFlag01 = 0
        self.scaleSigmaE = scaleSigmaE
        
    def stainlessMod(self, seyMax, reflec):    
        # Model parameters stainless steel 
        self.p1EInf = 0.07; self.p1Ehat = 0.5; self.eEHat = 0.;
        self.w = 100.; self.p = 0.9; self.e1 = 0.26;  
        self.e2 = 2.; self.q = 0.4; self.sigmaE = 1.9;       
        self.p1RInf = 0.74; self.eR = 40.; self.r = 1.;
        self.r1 = 0.26 ; self.r2 = 2.; self.deltaTSHat = 1.22;
        self.s = 1.813; self.eHat0 = 310.; self.t3 = 0.7;   
        self.t4 = 1.; self.t1 = 0.66; self.t2 = 0.8;
        self.theta0max = 84.*pi/180;    self.deltaERmax = 0.99;
        
        self.m = 10 #m can't be changed just be setting here, as epsN and pSmallN are limited to 10.
        self.alpha = 1.  #alpha can't be changed just by setting here, as it is "hard coded" in some integrals below for speed.
        self.epsN = numpy.array([3.9, 6.2, 13., 8.8, 6.25, 2.25, 9.2, 5.3, 17.8, 10.], dtype=numpy.double)
        self.pSmallN = numpy.array([1.6, 2., 1.8, 4.7, 1.8, 2.4, 1.8, 1.8, 2.3, 1.8], dtype=numpy.double)
        
        if seyMax != None:  
            self.deltaTSHat = seyMax-self.p1RInf-self.p1EInf

        if reflec != None:
            self.p1Ehat = reflec


    def copperMod(self, seyMax, reflec):    
        # Model parameters copper 
        self.p1EInf = 0.02; self.p1Ehat = 0.496; self.eEHat = 0.
        self.w = 60.86; self.p = 1.; self.e1 = 0.26  
        self.e2 = 2.; self.sigmaE = 2.;
        
        self.p1RInf = 0.2; self.eR = 0.041; self.r = 0.104
        self.q = 0.5; self.r1 = 0.26; self.r2 = 2.
        
        self.deltaTSHat = 1.8848; self.s = 1.54; self.eHat0 = 276.8
        self.t3 = 0.7; self.t4 = 1.; self.t1 = 0.66; self.t2 = 0.8
        
        self.theta0max = 84.*pi/180;    self.deltaERmax = 0.99;
        self.m = 10 #m can't be changed just be setting here, as epsN and pSmallN are limited to 10.
        self.alpha = 1.  #alpha can't be changed just be setting here, as it is "hard coded" in some integrals below for speed.
        self.epsN = numpy.array([2.5, 3.3, 2.5, 2.5, 2.8, 1.3, 1.5, 1.5, 1.5, 1.5], dtype=numpy.double)
        self.pSmallN = numpy.array([1.5, 1.75, 1., 3.75, 8.5, 11.5, 2.5, 3., 2.5, 3.], dtype=numpy.double)

        if seyMax != None:  
            self.deltaTSHat = seyMax-self.p1RInf-self.p1EInf

        if reflec != None:
            self.p1Ehat = reflec
            
    def generateSecondaries(self):

        # Some local variables
        cdef: 
            numpy.ndarray[numpy.double_t, ndim=2] absorbedParticlesNumpy = self.particleBoundaryObj.getAbsorbedParticles()
            double *absorbedParticles = <double *> absorbedParticlesNumpy.data
            numpy.ndarray[numpy.double_t, ndim=2] normalVectorsNumpy = self.particleBoundaryObj.getNormalVectors()
            double *normalVectors = <double *> normalVectorsNumpy.data
            double deltaE0, deltaR0, eHat, deltaTS0, deltaTS, deltaTSPrime
            double pr, pNCumSum, theta0, u, aE, aR, p0, y, vSec, v0
            double p1EInf = self.p1EInf, p1Ehat = self.p1Ehat, eEHat = self.eEHat
            double w = self.w, p = self.p, e1 = self.e1, e2 = self.e2, particleMass = self.particleMass
            double p1RInf = self.p1RInf, eR = self.eR, alpha = self.alpha
            double q = self.q, sigmaE = self.sigmaE, r = self.r, r1 = self.r1, 
            double r2 = self.r2, eHat0 = self.eHat0, t1 = self.t1, t2 = self.t2
            double t3 = self.t3, t4 = self.t4, s = self.s, deltaTSHat = self.deltaTSHat
            double theta0max = self.theta0max, deltaERmax = self.deltaERmax
            numpy.ndarray[numpy.double_t] epsNBuff = self.epsN
            double *epsN = <double *> epsNBuff.data
            numpy.ndarray[numpy.double_t] pSmallNBuff = self.pSmallN
            double *pSmallN = <double *> pSmallNBuff.data
            unsigned int m = self.m, warningFlag = self.warningFlag
            double thetaSec, cosThetaSec, sinThetaSec, phiSec, cosPhiSec, sinPhiSec
            double eFactor = 0.5*particleMass/elementary_charge
            double eFactorInv = 1./eFactor
            double deltaE, deltaR, temp, temp01, temp02
            unsigned int jj, ii, kk, currentSec, nSecSum
            unsigned int n0 = self.particleBoundaryObj.getAbsorbedMacroParticleCount()
            unsigned int nCoords = self.particlesObj.getNCoords()
            
            numpy.ndarray[numpy.double_t] e0Numpy = numpy.empty(n0, dtype=numpy.double)
            double *e0 = <double *> e0Numpy.data
            numpy.ndarray[numpy.double_t] eSecNumpy = numpy.empty(m, dtype=numpy.double)
            double *eSec = <double *> eSecNumpy.data
            numpy.ndarray[numpy.double_t] thetaKNumpy = numpy.empty(m, dtype=numpy.double)
            double *thetaK = <double *> thetaKNumpy.data
            numpy.ndarray[numpy.double_t] yKNumpy = numpy.empty(m, dtype=numpy.double)
            double *yK = <double *> yKNumpy.data
            numpy.ndarray[numpy.double_t] pNNumpy = numpy.empty(m+1, dtype=numpy.double)
            double *pN = <double *> pNNumpy.data          
            numpy.ndarray[numpy.uint32_t] nSecNumpy = numpy.empty(n0, dtype=numpy.uint32)
            unsigned int *nSec = <unsigned int *> nSecNumpy.data
            numpy.ndarray[numpy.double_t, ndim=2] secondariesNumpy
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
            theta0 = clip(theta0, 0., theta0max)
            
            deltaE0 = p1EInf + (p1Ehat-p1EInf)*exp(-((abs(e0[jj]-eEHat)/w)**p)/p)
            deltaE = deltaE0*(1 + e1*(1 - cos(theta0)**e2))
                         
            deltaR0 = p1RInf*(1-exp(-(e0[jj]/eR)**r))   
            deltaR = deltaR0*(1 + r1*(1 - cos(theta0)**r2)) 
                
            eHat = eHat0*(1 + t3*(1 - cos(theta0)**t4))       
            deltaTS0 = deltaTSHat*s*e0[jj]/eHat/(s-1+(e0[jj]/eHat)**s)  
            deltaTS = deltaTS0*(1 + t1*(1 - cos(theta0)**t2))
            
            temp = deltaE+deltaR
            if temp>=deltaERmax:
                deltaE = deltaE/temp*deltaERmax
                deltaR = deltaR/temp*deltaERmax
                if warningFlag00==0:
                    print 'WARNING: delta_E + delta_R >= delta_ERMax (~1). Will be set to delta_ERMax. \
                           This warning is suppressed from now on.'
                    warningFlag00 = 1
                     
            deltaTSPrime = deltaTS/(1-deltaE-deltaR)        
            if deltaTSPrime>=m:
                deltaTSPrime = m
                if warningFlag01==0:
                    print 'WARNING: delta_TS^Prime >= m (most likely 10). Will be set to m. \
                           This warning is suppressed from now on.'
                    warningFlag01 = 1  
            elif deltaTSPrime<0:
                deltaTSPrime = 0
            
            pr = deltaTSPrime/m
            for ii in range(m+1):
                pN[ii] = gslWrap.binom(ii,pr,m)*(1-deltaE-deltaR)
            pN[1] += deltaE + deltaR
            pNCumSum = 0
            nSec[jj] = m
            u = gslWrap.rand()
            for ii in range(m):
                pNCumSum += pN[ii]
                if u<=pNCumSum:
                    nSec[jj] = ii
                    break

        nSecSum = 0
        for ii in range(n0):
            nSecSum += nSec[ii]
     
    
        if nSecSum > 0:              
            self.secondaries = numpy.empty((nSecSum, self.particlesObj.getNCoords()), dtype=numpy.double)
            secondariesNumpy = self.secondaries
            secondaries = <double *> secondariesNumpy.data
              
            currentSec = 0
            for jj in range(n0):
                if nSec[jj] == 0:
                    pass
                elif nSec[jj] == 1:
                    aE = deltaE/pN[1]
                    aR = deltaR/pN[1]
                      
                    u = gslWrap.rand()

                    if u<aE:
                        if self.scaleSigmaE==0:
                            eSec[0] = e0[jj] - abs(sigmaE*gslWrap.randn())
                        else: 
                            eSec[0] = e0[jj] - abs(sigmaE*sqrt(e0[jj]/300)*gslWrap.randn())
                        if eSec[0]<0.:
                            eSec[0] = gslWrap.rand()*e0[jj] 
                    elif u < aE + aR:
                        eSec[0] = e0[jj]*gslWrap.rand()**(1./(1.+q))
                    else:              
                        eSec[0] = epsN[0]*gslWrap.gammaincinv(pSmallN[0],gslWrap.rand()*gslWrap.gammainc(pSmallN[0],e0[jj]/epsN[0])) # Check gamma
           
                else:   
                    p0 = gslWrap.gammainc(nSec[jj]*pSmallN[nSec[jj]-1],e0[jj]/epsN[nSec[jj]-1])
                       
                    for ii in range(nSec[jj]-1):
                        thetaK[ii] = asin(sqrt(gslWrap.betaincinv(pSmallN[nSec[jj]-1]*(nSec[jj]-ii-1),pSmallN[nSec[jj]-1],gslWrap.rand()))) # Check signs, machine epsilon, check formula, check beta
                    y = sqrt(gslWrap.gammaincinv(nSec[jj]*pSmallN[nSec[jj]-1],gslWrap.rand()*p0))     
                    yK[0] = y*cos(thetaK[0])
                    yK[nSec[jj]-1] = y
                    for ii in range(nSec[jj]-1):
                        yK[nSec[jj]-1] *= sin(thetaK[ii])
                    for ii in range(1,nSec[jj]-1):
                        yK[ii] = y*cos(thetaK[ii])
                        for kk in range(ii):
                            yK[ii] *= sin(thetaK[kk]) 
                    for ii in range(nSec[jj]):
                        eSec[ii] = epsN[nSec[jj]-1]*(yK[ii]*yK[ii])                    
                
                eSum = 0    
                for ii in range(nSec[jj]):
                    eSum+=eSec[ii]
                for ii in range(nSec[jj]):
                    kk = nCoords*currentSec 
                    secondaries[kk] = absorbedParticles[nCoords*jj]
                    secondaries[kk+1] = absorbedParticles[nCoords*jj+1]
                    secondaries[kk+5] = absorbedParticles[nCoords*jj+5]
        
                    vSec = sqrt(eFactorInv*eSec[ii])           
                    thetaSec = asin(gslWrap.rand())
                    cosThetaSec = cos(thetaSec)
                    sinThetaSec = sin(thetaSec)
                    phiSec = gslWrap.rand()*pi*2.
                    cosPhiSec = cos(phiSec)
                    sinPhiSec = sin(phiSec)
                    temp01 = cosThetaSec*normalVectors[2*jj] - sinThetaSec*normalVectors[2*jj+1]
                    temp02 = sinThetaSec*normalVectors[2*jj] + cosThetaSec*normalVectors[2*jj+1]
                    secondaries[kk+2] = vSec*( (cosPhiSec + normalVectors[2*jj]*normalVectors[2*jj]*(1-cosPhiSec))*temp01 + normalVectors[2*jj]*normalVectors[2*jj+1]*(1-cosPhiSec)*temp02 )
                    secondaries[kk+3] = vSec*( (cosPhiSec + normalVectors[2*jj+1]*normalVectors[2*jj+1]*(1-cosPhiSec))*temp02 + normalVectors[2*jj]*normalVectors[2*jj+1]*(1-cosPhiSec)*temp01 )
                    secondaries[kk+4] = vSec*( -sinPhiSec*normalVectors[2*jj+1]*temp01 + sinPhiSec*normalVectors[2*jj]*temp02 )
                    currentSec += 1    
                
            return secondariesNumpy
          
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
        object particlesObj, particleBoundaryObj
                
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
 
         
        if seyMax >= 0.:  
            self.deltaTSHat = seyMax-self.p1RInf-self.p1EInf
 
        if reflec >= 0.:
            self.p1Ehat = reflec
    
  
    def generateSecondaries(self):
    
        # Some local variables
        cdef: 
            numpy.ndarray[numpy.double_t, ndim=2] absorbedParticlesNumpy = self.particleBoundaryObj.getAbsorbedParticles()
            double *absorbedParticles = <double *> absorbedParticlesNumpy.data
            numpy.ndarray[numpy.double_t, ndim=2] normalVectorsNumpy = self.particleBoundaryObj.getNormalVectors()
            double *normalVectors = <double *> normalVectorsNumpy.data
            
            double deltaE0, deltaR0, eHat, deltaTS0, deltaTS, deltaE, deltaR
            double theta0, u, aE, aR, p0, vSec, v0, nSecSum = 0, vFac
            double p1EInf = self.p1EInf, p1Ehat = self.p1Ehat, eEHat = self.eEHat
            double w = self.w, p = self.p, e1 = self.e1, e2 = self.e2, particleMass = self.particleMass
            double p1RInf = self.p1RInf, eR = self.eR
            double q = self.q, r = self.r, r1 = self.r1, 
            double r2 = self.r2, eHat0 = self.eHat0, t1 = self.t1, t2 = self.t2
            double t3 = self.t3, t4 = self.t4, s = self.s, deltaTSHat = self.deltaTSHat
            double theta0max = self.theta0max, deltaERmax = self.deltaERmax
            double thetaSec, cosThetaSec, sinThetaSec, phiSec, cosPhiSec, sinPhiSec, temp
            double eFactor = 0.5*particleMass/elementary_charge
            double eFactorInv = 1./eFactor
            double sigmaETS = self.sigmaETS, muETS = self.muETS
            unsigned int jj, ii, kk, currentSec, deltaTSFloor
            unsigned int n0 = self.particleBoundaryObj.getAbsorbedMacroParticleCount()
            unsigned int nCoords = self.particlesObj.getNCoords()
                
            numpy.ndarray[numpy.double_t] e0Numpy = numpy.empty(n0, dtype=numpy.double)
            double *e0 = <double *> e0Numpy.data              
            numpy.ndarray[numpy.uint32_t] nSecENumpy = numpy.zeros(n0, dtype=numpy.uint32)
            unsigned int *nSecE = <unsigned int *> nSecENumpy.data
            numpy.ndarray[numpy.uint32_t] nSecRNumpy = numpy.zeros(n0, dtype=numpy.uint32)
            unsigned int *nSecR = <unsigned int *> nSecRNumpy.data
            numpy.ndarray[numpy.uint32_t] nSecTSNumpy = numpy.empty(n0, dtype=numpy.uint32)
            unsigned int *nSecTS = <unsigned int *> nSecTSNumpy.data
            numpy.ndarray[numpy.double_t] eSecNumpy = numpy.empty(<unsigned int> (deltaTSHat+2), dtype=numpy.double)
            double *eSec = <double *> eSecNumpy.data
            
            numpy.ndarray[numpy.double_t, ndim=2] secondariesNumpy
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
            theta0 = clip(theta0, 0., theta0max)
    
            deltaE0 = p1EInf + (p1Ehat-p1EInf)*exp(-(abs(e0[jj]-eEHat)/w)**p/p)
            deltaE = deltaE0*(1 + e1*(1 - cos(theta0)**e2))
                         
            deltaR0 = p1RInf*(1-exp(-(e0[jj]/eR)**r))   
            deltaR = deltaR0*(1 + r1*(1 - cos(theta0)**r2)) 
                     
            eHat = eHat0*(1 + t3*(1 - cos(theta0)**t4))            
            deltaTS0 = deltaTSHat*s*e0[jj]/eHat/(s-1+(e0[jj]/eHat)**s)  
            deltaTS = deltaTS0*(1 + t1*(1 - cos(theta0)**t2))

            temp = deltaE+deltaR
            if temp>=deltaERmax:
                deltaE = deltaE/temp*deltaERmax
                deltaR = deltaR/temp*deltaERmax
                                           
            if gslWrap.rand()<=deltaE:
                nSecE[jj] = 1
            if gslWrap.rand()<=deltaR:
                nSecR[jj] = 1
            deltaTSFloor = <unsigned int> deltaTS
            if gslWrap.rand()<=(deltaTS-deltaTSFloor):
                nSecTS[jj] = deltaTSFloor + 1
            else:
                nSecTS[jj] = deltaTSFloor

        for ii in range(n0):
            nSecSum += nSecE[ii] + nSecR[ii] + nSecTS[ii]
              
        if nSecSum > 0:              
            secondariesNumpy = numpy.empty((nSecSum, nCoords), dtype=numpy.double)
            secondaries = <double *> secondariesNumpy.data
            eSecNumpy = numpy.empty(nSecTSNumpy.max(), dtype=numpy.double)
            eSec = <double *> eSecNumpy.data   
              
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
                    vFac = sqrt(gslWrap.rand()**(1./(1.+q)))
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
                        eSec[ii] = exp(sigmaETS*gslWrap.randn()+muETS)
                        eSum += eSec[ii]   
                    
                    if eSum>e0[jj]:               
                        for ii in range(nSecTS[jj]):
                            eSec[ii] = gslWrap.rand()*e0[jj]/nSecTS[jj]
                    for ii in range(nSecTS[jj]):
                        kk = nCoords*currentSec
                        secondaries[kk] = absorbedParticles[nCoords*jj]
                        secondaries[kk+1] = absorbedParticles[nCoords*jj+1]
                        secondaries[kk+5] = absorbedParticles[nCoords*jj+5]
                        vSec = sqrt(eFactorInv*eSec[ii])           
                        thetaSec = asin(gslWrap.rand())
                        cosThetaSec = cos(thetaSec)
                        sinThetaSec = sin(thetaSec)
                        phiSec = gslWrap.rand()*pi*2.
                        cosPhiSec = cos(phiSec)
                        sinPhiSec = sin(phiSec)
                        temp01 = cosThetaSec*normalVectors[2*jj] - sinThetaSec*normalVectors[2*jj+1]
                        temp02 = sinThetaSec*normalVectors[2*jj] + cosThetaSec*normalVectors[2*jj+1]
                        secondaries[kk+2] = vSec * ( (cosPhiSec + normalVectors[2*jj]*normalVectors[2*jj]*(1-cosPhiSec))*temp01 + normalVectors[2*jj]*normalVectors[2*jj+1]*(1-cosPhiSec)*temp02 )
                        secondaries[kk+3] = vSec * ( (cosPhiSec + normalVectors[2*jj+1]*normalVectors[2*jj+1]*(1-cosPhiSec))*temp02 + normalVectors[2*jj]*normalVectors[2*jj+1]*(1-cosPhiSec)*temp01 )
                        secondaries[kk+4] = vSec * ( -sinPhiSec*normalVectors[2*jj+1]*temp01 + sinPhiSec*normalVectors[2*jj]*temp02 )
#                         vFac = sqrt(eSec[ii]/e0[jj]) 
#                         secondaries[kk+2] = -absorbedParticles[nCoords*jj+2]*vFac
#                         secondaries[kk+3] = -absorbedParticles[nCoords*jj+3]*vFac
#                         secondaries[kk+4] = -absorbedParticles[nCoords*jj+4]*vFac
                        currentSec += 1
            return secondariesNumpy               
        else:
            return numpy.empty((0,nCoords), dtype=numpy.double)


cdef class SimpleSecEmitter:

    cdef: 
        object particlesObj, particleBoundaryObj
                
    def __init__(self, particleBoundaryObj, particlesObj, double seyMax=-1., double reflec=-1.):
        
        self.particlesObj = particlesObj
        self.particleBoundaryObj = particleBoundaryObj 
        
    def generateSecondaries(self):
    
        # Some local variables
        cdef: 
            numpy.ndarray[numpy.double_t, ndim=2] absorbedParticlesNumpy = self.particleBoundaryObj.getAbsorbedParticles()
            double *absorbedParticles = <double *> absorbedParticlesNumpy.data
            numpy.ndarray[numpy.double_t, ndim=2] normalVectorsNumpy = self.particleBoundaryObj.getNormalVectors()
            double *normalVectors = <double *> normalVectorsNumpy.data
            
            unsigned int jj, ii, kk, currentSec, deltaTSFloor
            unsigned int n0 = self.particleBoundaryObj.getAbsorbedMacroParticleCount()
            unsigned int nCoords = self.particlesObj.getNCoords()

            double particleMass = 9.10938291e-31
            double elementary_charge = 1.602176565e-19
            double eFactor = 0.5*particleMass/elementary_charge
            double s = 1.813
            double eHat = 310
            double nSecSum = 0
                            
            numpy.ndarray[numpy.double_t] e0Numpy = numpy.empty(n0, dtype=numpy.double)
            double *e0 = <double *> e0Numpy.data              
            numpy.ndarray[numpy.uint32_t] nSecENumpy = numpy.zeros(n0, dtype=numpy.uint32)
            unsigned int *nSecE = <unsigned int *> nSecENumpy.data
            numpy.ndarray[numpy.uint32_t] nSecRNumpy = numpy.zeros(n0, dtype=numpy.uint32)
            unsigned int *nSecR = <unsigned int *> nSecRNumpy.data
            numpy.ndarray[numpy.uint32_t] nSecTSNumpy = numpy.empty(n0, dtype=numpy.uint32)
            unsigned int *nSecTS = <unsigned int *> nSecTSNumpy.data
            numpy.ndarray[numpy.double_t] eSecNumpy = numpy.empty(10, dtype=numpy.double)
            double *eSec = <double *> eSecNumpy.data
            
            numpy.ndarray[numpy.double_t, ndim=2] secondariesNumpy
            double *secondaries
    
        
        for jj in range(n0):
                
            v0 = sqrt(absorbedParticles[nCoords*jj+2]*absorbedParticles[nCoords*jj+2] + 
                      absorbedParticles[nCoords*jj+3]*absorbedParticles[nCoords*jj+3] + 
                      absorbedParticles[nCoords*jj+4]*absorbedParticles[nCoords*jj+4])
            e0[jj] = eFactor*v0*v0
    
            deltaE = 0.05
                         
            deltaR = 0.5
                              
            deltaTS = 1.4*s*e0[jj]/eHat/(s-1+(e0[jj]/eHat)**s)  

            if gslWrap.rand()<=deltaE:
                nSecE[jj] = 1
            if gslWrap.rand()<=deltaR:
                nSecR[jj] = 1

            if gslWrap.rand()<=(deltaTS-(<unsigned int> deltaTS)):
                nSecTS[jj] = <unsigned int> (deltaTS+1.)
            else:
                nSecTS[jj] = <unsigned int> (deltaTS)

        for ii in range(n0):
            nSecSum += nSecE[ii] + nSecR[ii] + nSecTS[ii]
              
        if nSecSum > 0:              
            secondariesNumpy = numpy.empty((nSecSum, nCoords), dtype=numpy.double)
            secondaries = <double *> secondariesNumpy.data
            eSecNumpy = numpy.empty(nSecTSNumpy.max(), dtype=numpy.double)
            eSec = <double *> eSecNumpy.data   
              
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
                    vFac = sqrt(gslWrap.rand())
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
                        eSec[ii] = gslWrap.rand()*5
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
            return secondariesNumpy               
        else:
            return numpy.empty((0,nCoords), dtype=numpy.double)
