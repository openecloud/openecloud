#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True

from constants cimport *

# Simple C methods
cdef extern from "math.h":
    double exp(double x)
    double sqrt(double x)
    double log(double x)
    double sin(double x)

# GSL imports of functions.
cdef extern from "gsl/gsl_sf_gamma.h":
    double gsl_sf_beta(double a, double b) nogil
    double gsl_sf_beta_inc(double a, double b, double x) nogil
    double gsl_sf_gamma_inc_P(double a, double x) nogil
    double gsl_sf_gamma(double a) nogil
    double gsl_sf_lngamma(double a) nogil
    
cdef extern from "gsl/gsl_cdf.h":
    double gsl_cdf_ugaussian_Pinv(double p) nogil
    double gsl_cdf_ugaussian_Qinv(double q) nogil
    double gsl_cdf_tdist_Pinv(double p, double nu) nogil

cdef extern from "gsl/gsl_randist.h":
    double gsl_ran_binomial_pdf(unsigned int k, double p, unsigned int n) nogil
    
# Simple wrappers.
# Could be replaced by direct implementation in the future.
cdef double binom(unsigned int k, double p, unsigned int n):
    return (<double> gsl_ran_binomial_pdf(k, p, n))
    
cdef double betainc(double a, double b, double x):
    return (<double> gsl_sf_beta_inc(a, b, x))

cdef double gammainc(double a, double x):
    return (<double> gsl_sf_gamma_inc_P(a, x))

cdef double gammaincd(double a, double x, double gammaa):
    return (<double> x**(a-1)/exp(x)/gammaa)

cdef double gamma(double a):
    return (<double> gsl_sf_gamma(a))

cdef double lngamma(double a):
    return (<double> gsl_sf_lngamma(a))

# Inspired by boost library documentation.
cdef double betaincinv(double a, double b, double p, double acc = 1.e-12):

    cdef:
        double q = 1 - p
        double sqrt2 = sqrt(2)
        double EPS = sqrt(acc)
        double x, temp, eta0, beta, eps1, eps2, eps3, eps, eta, dxdenom, t
        double betafunci, a1, a2, b1, b2, der1, der2, y, xa1, xa2, yb1, yb2
        double xr, xl, c, s, r, s2, c2, u, alpha, mu, 
        double w, w2, w3, w4, w5, w6, w7, w8, w9, w10, wa1, wa1p2, wa1p3, wa1p4
        double eta0mmu, eta0mmu2, eta0mmu3, eta0mmu4, errl
        unsigned int nIterHalley = <unsigned int> ((-log(acc)/log(10))+1.)
        unsigned int nIterBisect = <unsigned int> ((-log(acc)/log(2))+5.)
        unsigned int ii, jj, abSwitch = 0
    

    # Trivial cases.
    if q<=machineEpsilon:
        return 1.
    elif p<=machineEpsilon:
        return 0.
    elif a==1 and b==1:
        return p
    elif a==0.5 and b==0.5:
        return sin(p*0.5*pi)
    elif a==0.5 and b>=0.5:
        x = gsl_cdf_tdist_Pinv(0.5*q, 2.*b)
        return 1 - 2.*b/(2.*b+x*x)
    elif a>=0.5 and b==0.5:
        x = gsl_cdf_tdist_Pinv(0.5*p, 2.*a)
        return 2.*a/(2.*a+x*x)
    
    # Switch to a<=b if not the case.    
    if a>b:
        swap(&a,&b)
        swap(&p,&q)
        logical_not(&abSwitch)    
    
    # Temme inverse beta incomplete.
    if a+b>=5:           
        if (b-a)<=sqrt(a):
            # Section 2
            eta0 = -1./sqrt(a)*gsl_cdf_ugaussian_Qinv(p)
            beta = b - a
            eps1 = (-0.5*beta*sqrt2 + 0.125*(1.-2.*beta)*eta0 -
                    1./48.*beta*sqrt2*eta0*eta0 - 1./192.*eta0*eta0*eta0 -
                    1./3840.*beta*sqrt2*eta0*eta0*eta0*eta0)
            eps2 = (1./12.*beta*sqrt2*(3.*beta-2.) + 1./128.*(20.*beta*beta - 12*beta + 1.)*eta0 +
                    1./960.*beta*sqrt2*(20.*beta-1.)*eta0*eta0 + 1./4608.*(16.*beta*beta+30.*beta-15.)*eta0*eta0*eta0 +
                    1./53760.*beta*sqrt2*(21.*beta+32.)*eta0*eta0*eta0*eta0 +
                    1./368640.*(-32.*beta*beta+63.)*eta0*eta0*eta0*eta0*eta0 + 
                    1./25804480*beta*sqrt2*(120.*beta+17.)*eta0*eta0*eta0*eta0*eta0*eta0)
            eps3 = (1./480.*beta*sqrt2*(-75.*beta*beta+80.*beta-16.) +
                    1./9216.*(-1080.*beta*beta*beta+868.*beta*beta-90.*beta*beta-45.)*eta0 +
                    1./53760*beta*sqrt2*(-1190.*beta*beta+84.*beta+373.)*eta0*eta0 +
                    1./368640.*(-2240.*beta*beta*beta-2508.*beta*beta+2100.*beta-165.)*eta0*eta0*eta0)
            eps = eps1/a + eps2/(a*a) + eps3/(a*a*a)
            eta = eta0 + eps
            if eta>0:
                x = 0.5 + sqrt(0.25 - 0.25*exp(-0.5*eta*eta))               
            else:
                x = 0.5 - sqrt(0.25 - 0.25*exp(-0.5*eta*eta))  
        elif 0.2<=a/(a+b)<=0.8:            
            if p**(1./a)<0.0025:
                x = (a*p*gsl_sf_beta(a,b))**(1./a)
            elif q**(1./b)<0.0025:
                x = 1 - (b*q*gsl_sf_beta(a,b))**(1./b)
            else:
                # Section 3
                r = a + b;      s2 = a/r;   c2 = b/r;
                s = sqrt(s2);     c = sqrt(c2);
                eta0 = -1./sqrt(r)*gsl_cdf_ugaussian_Qinv(p)
                eps1 = ((2.*s2-1)/(2*s*c) - (5.*s2*s2-5.*s2-1)/(36.*s2*c2)*eta0 +
                        (46.*s2*s2*s2-69.*s2*s2+21.*s2+1)/(1620.*s2*s*c2*c)*eta0**2 -
                        (-2.*s2-62.*s2*s2*s2+31.*s2*s2*s2*s2+33.*s2*s2+7.)/(6480.*s2*s2*c**4)*eta0*eta0*eta0 +
                        (88.*s2*s2*s2-52.*s2-115.*s2*s2*s2*s2+46.*s2*s2*s2*s2*s2-17.*s2*s2+25.)/(90720.*s2*s2*s*c2*c2*c)*eta0*eta0*eta0*eta0)
                eps2 = (-(52.*s2*s2*s2-78.*s2*s2+12.*s2+7.)/(405.*s2*s*c2*c) + 
                        (2.*s2-370.*s2*s2*s2+185.*s2*s2*s2*s2+183.*s2*s2-7.)/(2592.*s2*s2*c**4)*eta0 -
                        (776.*s2+10240.*s2*s2*s2-13525.*s2*s2*s2*s2-533.+5410.*s2*s2*s2*s2*s2-1835.*s2*s2)/(204120.*s2*s2*s*c2*c2*c)*eta0*eta0 + 
                        (3747.*s2+15071.*s2*s2*s2*s2*s2*s2-15821.*s2*s2*s2+45588.*s2*s2*s2*s2-45213.*s2*s2*s2*s2*s2-3372.*s2*s2-1579.)/(2099520.*s2*s2*s2*c2*c2*c2)*eta0*eta0*eta0)
                eps3 = ((3704.*s2*s2*s2*s2*s2-9620.*s2*s2*s2*s2+6686.*s2*s2*s2-769.*s2*s2-1259.*s2+449.)/(102060.*s2*s2*s*c2*c2*c) -
                        (750479.*s2*s2*s2*s2*s2*s2-151557.*s2-727469.*s2*s2*s2+2239932.*s2*s2*s2*s2-2251437.*s2*s2*s2*s2*s2+140052.*s2*s2+63149.)/(20995200.*s2*s2*s2*c2*c2*c2)*eta0 +
                        (729754.*s2*s2*s2*s2*s2*s2*s2-78755.*s2-2554139.*s2*s2*s2*s2*s2*s2+146879.*s2*s2*s2-1602610.*s2*s2*s2*s2+3195183.*s2*s2*s2*s2*s2+105222.*s2*s2+29233.)/ \
                        (36741600.*s2*s2*s2*s*c2*c2*c2*c)*eta0*eta0)
                eps = eps1/r + eps2/(r*r) + eps3/(r*r*r)
                eta = eta0 + eps
                if eta*eta<0.49:
                    x = (s2 + s*c*eta + (1.-2.*s2)/3.*eta*eta + 
                         (13.*s2*s2-13.*s2+1.)/(36.*s*c)*eta*eta*eta +
                         (46.*s2*s2*s2-69.*s2*s2+21.*s2+1.)/(270.*s2*c2)*eta*eta*eta*eta)
                else:
                    if eta<0:
                        alpha = c2/s2
                        u = exp(1./s2*(-0.5*eta*eta + s2*log(s2) + c2*log(c2)))
                        x = (u + alpha*u*u + 3.*alpha*(3.*alpha+1.)/6.*u*u*u +
                             4.*alpha*(4.*alpha+1.)*(4.*alpha+2.)/24.*u*u*u*u +
                             5*alpha*(5.*alpha+1.)*(5.*alpha+2.)*(5.*alpha+3.)/120.*u*u*u*u*u)
                    else:
                        alpha = s2/c2
                        u = exp(1./c2*(-0.5*eta*eta + s2*log(s2) + c2*log(c2)))
                        x = (u + alpha*u*u + 3.*alpha*(3.*alpha+1.)/6.*u*u*u +
                             4.*alpha*(4.*alpha+1.)*(4.*alpha+2.)/24.*u*u*u*u +
                             5*alpha*(5.*alpha+1.)*(5.*alpha+2.)*(5.*alpha+3.)/120.*u*u*u*u*u)
                        x = 1 - x

        else:
            # Section 4
            # Switched a and b here, as I am assuming a<b.
            eta0 = 1./b*gammaincinv(a,p)
            mu = a/b
            w = sqrt(1.+mu)
            w2 = w*w;   w3 = w*w2;  w4 = w3*w;  w5 = w4*w;
            w6 = w5*w;  w7 = w6*w;  w8 = w7*w;  w9 = w8*w;  w10 = w9*w;
            wa1 = w+1;  wa1p2 = wa1*wa1;    wa1p3 = wa1p2*wa1;  wa1p4 = wa1p3*wa1;
            eta0mmu = eta0-mu;              eta0mmu2 = eta0mmu*eta0mmu;
            eta0mmu3 = eta0mmu2*eta0mmu;    eta0mmu4 = eta0mmu3*eta0mmu;
            eps1 = ((w+2.)*(w-1.)/(3.*w) +
                    (w3+9.*w2+21.*w+5.)/(36.*w2*wa1)*eta0mmu -
                    (w4-13.*w3+69.*w2+167.*w+46.)/(1620.*wa1p2*w3)*eta0mmu2 -
                    (7.*w5+21.*w4+70.*w3+26.*w2-93.*w-31.)/(6480.*wa1p3*w4)*eta0mmu3 -
                    (75.*w6+202.*w5+188.*w4-888.*w3-1345.*w2+118.*w+138.)/(272160.*wa1p4*w5)*eta0mmu4)      
            eps2 = ((28.*w4+131.*w3+402.*w2+581.*w+208.)*(w-1.)/(1620.*wa1*w3) -
                    (35.*w6-154.*w5-623.*w4-1636.*w3-3983.*w2-3514.*w-925.)/(12960.*wa1p2*w4)*eta0mmu -
                    (2132.*w7+7915.*w6+16821.*w5+35066.*w4+87490.*w3+141183.*w2+95993.*w+21640.)/(816480.*w5*wa1p3)*eta0mmu2 -
                    (11053.*w8+53308.*w7+117010.*w6+163924.*w5+116188.*w4-258428.*w3-677042.*w2-481940.*w-105497.)/(14696640.*wa1p4*w6)*eta0mmu3) 
            eps3 = (-((3592.*w7+8375.*w6-1323.*w5-29198.*w4-89578.*w3-154413.*w2-116063.*w-29632.)*(w-1.))/(816480.*w5*wa1p2) -
                    (442043.*w9+2054169.*w8+3803094.*w7+3470754.*w6+2141568.*w5-2393568.*w4-19904934.*w3-34714674.*w2-23128299.*w-5253353.)/(146966400.*w6*wa1p3)*eta0mmu -
                    (116932.*w10+819281.*w9+2378172.*w8+4341330.*w7+6806004.*w6+10622748.*w5+18739500.*w4+30651894.*w3+30869976.*w2+15431867.*w+2919016.)/(146966400.*wa1p4*w7)*eta0mmu2)
            eps = eps1/b + eps2/(b*b) + eps3/(b*b*b)
            eta = eta0 + eps
            u = exp(-eta+mu*log(eta)-(1+mu)*log(1+mu)+mu)
            if eta==mu:
                x = 1./(1.+mu)
            else:
                if eta<mu:
                    x = 0.5*(1./(1.+mu)+1.)
                else:
                    x = 0.5/(1.+mu)                
                for jj in range(nIterHalley):
                    if x<= 0.0:
                        return 0.0
                    elif x>=1.:
                        return 1.0
                    err = x*(1-x)**mu - u
                    y = 1.-x
                    der1 = y**mu - mu*x*y**(mu-1)
                    der2 = mu*(mu-1.)*x*y**(mu-2.)
                    dxdenom = 2.*der1**2 - err*der2
                    t = 2*err*der1/dxdenom
                    x -= t
                    if abs(t) < 0.1*x:
                        jj = 0
                        break
            x = 1-x
    
    elif a<1. and b<1.:
        if p>gsl_sf_beta_inc(a,b,(1.-a)/(2.-a-b)):
            temp = (b*q*gsl_sf_beta(b,a))**(1./b)
            x = 1 - temp/(1+temp)
        else:
            temp = (a*p*gsl_sf_beta(a,b))**(1./a)
            x = temp/(1+temp)
    elif a>1. and b>1.:
        if p>gsl_sf_beta_inc(b,a,(1.-b)/(2.-b-a)):
            temp = (b*q*gsl_sf_beta(b,a))**(1./b)
            x =  1. - temp + (a-1.)/(b+1.)*temp*temp + (a-1.)*(b*b+3.*a*b-b+5.*a-4.)/(2.*(b+1.)**2*(b+2.))*temp*temp*temp 
        else:
            temp = (a*p*gsl_sf_beta(a,b))**(1./a)
            x =  temp + (b-1.)/(a+1.)*temp*temp + (b-1.)*(a*a+3.*b*a-a+5.*b-4.)/(2.*(a+1.)**2*(a+2.))*temp*temp*temp
    elif a<=1. and b>1.:
            x = (1-q**(a*gsl_sf_beta(b,a)))**(1./a)
    
    if x<=0:
        x = 0.
    elif x>=1.:
        x = 1.
    
    betafunci = 1./gsl_sf_beta(a,b)
    a1 = a-1.;  a2 = a1-1.;
    b1 = b-1.;  b2 = b1-1.;
    for jj in range(nIterHalley):
        if x<= 0.0:
            return 0.0
        elif x>=1.:
            return 1.0
        err = gsl_sf_beta_inc(a, b, x) - p
        y = 1.-x
        xa1 = x**a1;    xa2 = x**a2;
        yb1 = y**b1;    yb2 = y**b2;
        der1 = xa1*yb1*betafunci
        der2 = betafunci*(xa2*yb1 - xa1*yb2)
        dxdenom = 2.*der1**2 - err*der2
        t = 2*err*der1/dxdenom
        x -= t
        if abs(t) < EPS*x:
            break 

    if abSwitch == 1:
        return 1-x
    else:
        return x

cdef void swap(double* a, double* b):
    cdef double temp
    temp = a[0]
    a[0] = b[0]
    b[0] = temp

cdef void logical_not(unsigned int* boolInt):
    if boolInt[0] == 1:
        boolInt[0] = 0
    else:
        boolInt[0] = 1
    
# Numerical Recipes p. 263
cdef double gammaincinv(double a, double p, double acc = 1.e-9):
    cdef:
        int j
        double x, err, t, u, pp, lna1, afac, a1=a-1
        double EPS = sqrt(acc)                                #Accuracy is square of EPS
        double gln
        double xl, xr
    
    
    if a<=0:
        raise ValueError('a has to be larger than 0.')
    if p>=1.:
        return max(100., a + 100.*sqrt(a))
    if p<=0.:
        return 0.0
    
    gln = gsl_sf_lngamma(a)
    if a>1.:
        lna1 = log(a1)
        afac = exp(a1*(lna1-1.)-gln)
        pp = p if p<0.5 else (1-p)
        t = sqrt(-2.*log(pp))
        x = (2.30753+t*0.27061)/(1+t*(0.99229+t*0.04481)) - t
        if p<0.5:
            x = -x
        x = max(1.e-3, a*(1.-1./(9.*a)-x/(3.*sqrt(a)))**3)
    else:
        t = 1. - a*(0.253+a*0.12)
        if p<t:
            x = (p/t)**(1./a)
        else:
            x = 1.-log(1.-(p-t)/(1.-t))

    for j in range(20):
        if x<= 0.0:
            return 0.0
        err = gsl_sf_gamma_inc_P(a, x) - p
        if a>1.:
            t = afac*exp(-(x-a1)+a1*(log(x)-lna1))
        else:
            t = exp(-x+a1*log(x)-gln)
        u = err/t
        t = u/(1.-0.5*min(1.,u*((a-1.)/x - 1.)))
        x -= t
        if x<=0.:
            x = 0.5*(x + t)
        if abs(t) < EPS*x:
            j = 0
            break 
    if j==19 and x<a:
        EPS = EPS*EPS
        err = gsl_sf_gamma_inc_P(a, x) - p
        xr = x
        while err<=0:
            xr *= 2.
            err = gsl_sf_gamma_inc_P(a, xr) - p
        err = gsl_sf_gamma_inc_P(a, x) - p
        xl = 0.
        x = 0.5*(xr+xl)
        for j in range(100):
            err = gsl_sf_gamma_inc_P(a, x) - p
            if err<=0.:
                xl = x
            else:
                xr = x
            x = 0.5*(xr+xl)
            if 0.5*(xr-x) < EPS*x:
                break 
            if x < EPS*a:
                x = 0.
                break
         
    return x
