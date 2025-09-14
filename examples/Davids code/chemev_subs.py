
import numpy as np

def boxcar(a,nhalf):
    """
    Boxcar smooth an array a with a constant window of with 2*nhalf+1
    nhalf elements are copied past edges before smoothing, so returned
      array has same length as original
    """
    n=2*nhalf+1
    b=np.pad(a,nhalf,mode='edge')
    ret=np.cumsum(b,dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:] / n

def SetupSNIa(t,tauIa,tdmin,dtd):	
    """
    Fill the SNIa DTD array
      t = time array
      tauIa = e-folding time for exponential DTD
      tdmin = minimum delay time
      dtd = 'exp' or 'plaw' for exponental or t^{-1.1}
    """
    if (dtd=='exp'):
        RIa=np.exp(-t/tauIa)
    elif (dtd=='plaw'):
        RIa=(t+1.e-12)**(-1.1)		# avoid divergence at t=0
    else:
        raise ValueError("dtd must be exp or plaw")
    RIa[t<tdmin]=0.			# set to zero before minimum td

    # normalize
    RIa /= np.sum(RIa)
    return RIa

def SetupSimple(mdotfid,t,taustar,eta,tausfh,sfhmode):
    """
    Set up array values for a simple case: 
      mdotfid = fiducial SFR, in Msun/Gyr
      t = time array
      taustar = value of taustar, assumed constant
      eta = value of eta, assumed constant
      tausfh = SFH e-folding time, assumed constant
      sfhmode = 'constant' or 'exp' or 'linexp'
    """
    taustar_a = 0*t+ taustar		# 0*t sets up array same length as t
    eta_a = 0*t + eta
    if (sfhmode=='exp'):
        mdotstar=mdotfid*np.exp(-t/tausfh)
    elif (sfhmode=='linexp'):
        mdotstar=mdotfid*(t/tausfh)*np.exp(-t/tausfh)
    elif (sfhmode=='constant'):
        mdotstar=0*t+mdotfid
    else:
        raise ValueError("sfhmode must be exp, linexp, or constant")
    mgas=taustar_a*mdotstar
    return taustar_a,eta_a,mdotstar,mgas

def SetupRiseFall(mdotfid,t,taustar,eta,tausfh1,tausfh2):
    """
    Set up array values for a simple case: 
      mdotfid = fiducial SFR, in Msun/Gyr
      t = time array
      taustar = value of taustar, assumed constant
      eta = value of eta, assumed constant
      tausfh1, tausfh2 = timescales in SFR ~ (1-exp(-t/tausfh1))*exp(-t/tausfh2)
    """
    taustar_a = 0*t+ taustar		# 0*t sets up array same length as t
    eta_a = 0*t + eta
    mdotstar=mdotfid*(1.-np.exp(-t/tausfh1))*np.exp(-t/tausfh2)
    mgas=taustar_a*mdotstar
    return taustar_a,eta_a,mdotstar,mgas

def SetupTwoPhase(mdotfid,t,tc,tausfh1,taustar1,eta1,tausfh2,taustar2,eta2):
    """
    Set up array values for a case with discontinuous parameter changes
      mdotfid = fiducial SFR, in Msun/Gyr
      t = time array
      tc = time of transition
      tausfh1, tausfh2 = SFH e-folding timescales before and after tc
      taustar1, taustar2 = SFE timescales before and after tc
      eta1, eta2 = outflow mass loading factors before and after tc
      nhalf = arrays will be boxcar smoothed with window width 2*nhalf+1
    NOTE: Gas supply is continuous at tc, mdotstar changes by taustar1/taustar2
    """
    taustar_a = 0*t+ taustar1		# 0*t sets up array same length as t
    eta_a = 0*t + eta1
    mdotstar=mdotfid*np.exp(-t/tausfh1)
    taustar_a[t>tc]=taustar2
    eta_a[t>tc]=eta2
    mdottc = mdotfid*np.exp(-tc/tausfh1)*taustar1/taustar2
    t2=t[t>tc]-tc
    mdotstar[t>tc] = mdottc*np.exp(-t2/tausfh2)
    mgas=taustar_a*mdotstar

    return taustar_a,eta_a,mdotstar,mgas

def SetupPLSchmidt(mdotfid,t,tausfh,taustar0,eta,sfhmode):
    """
    Set up array values for Schmidt-law efficiency, taustar \propto 1/sqrt(mgas)
      mdotfid = fiducial SFR, in Msun/Gyr
      t = time array
      tausfh = SFH e-folding timescale 
      taustar0 = taustar at t=0 (for sfhmode=exp)
                 taustar at t=tsfh (for sfhmode=linexp)
      eta = outflow mass loading factor, assumed constant
      sfhmode = exp, linexp, or constant
    """
    taustar_a = 0*t           		# 0*t sets up array same length as t
    eta_a = 0*t + eta
    if (sfhmode=='exp'):
        mdotstar=mdotfid*np.exp(-t/tausfh)
    elif (sfhmode=='linexp'):
        mdotstar=mdotfid*(t/tausfh)*np.exp(-t/tausfh)
    elif (sfhmode=='constant'):
        mdotstar=0*t+mdotfid
    else:
        raise ValueError("sfhmode must be exp, linexp, or constant")
    if ((sfhmode=='exp') | (sfhmode=='constant')):
        mgas0=mdotfid*taustar0
    else:
        mgas0=mdotfid*np.exp(-1.0)*taustar0
    mgas=(np.sqrt(mgas0)*taustar0*mdotstar)**(0.666667)
    # 1.e-4 is added below to avoid divide by zero at first timestep
    taustar_a=taustar0/(np.sqrt(mgas/mgas0)+1.e-4) 
    return taustar_a,eta_a,mdotstar,mgas

def SetupConstantInfall(t,taustar0,eta0,mdotinf0):
    """
    Set up array values for a simple case with constant infall
      t = time array
      taustar0 = value of taustar, assumed constant
      eta0 = value of eta, assumed constant
      mdotinf0 = value of constant infall rate, Msun/Gyr
    """
    taustar_a = 0*t+ taustar0		# 0*t sets up array same length as t
    eta_a = 0*t + eta0
    mdotinf = 0*t + mdotinf0
    return taustar_a,eta_a,mdotinf

def SetupMultiBurst(t,taustar0,eta0,mdotinf0,t1,dt1,dt2,factor,burstmode):
    """
    Set up array values for multiple bursts
      t = time array
      taustar0 = value of taustar, assumed constant
      eta0 = value of eta, assumed constant
      mdotinf0 = value of constant infall rate, Msun/Gyr
      t1 = start time of first burst
      dt1 = burst duration
      dt2 = time between bursts
      tmax = maximum time
      factor = factor by which to multiply SFE or mdotinfall
      burstmode = SFE or Mdot, to boost one or the other during bursts
    """
    taustar_a = 0*t+ taustar0		# 0*t sets up array same length as t
    eta_a = 0*t + eta0
    mdotinf = 0*t + mdotinf0
    inburst = (t<0) 			# all false to begin with
    tnext = t1
    for i in range(len(t)):
        if (t[i]>=tnext):
            inburst[i]=True
        if (t[i]>=tnext+dt1):
            tnext += dt2
    if ((burstmode=='sfe') | (burstmode=='SFE')):
        taustar_a[inburst] /= factor
    elif ((burstmode=='mdot') | (burstmode=='Mdot')):
        mdotinf[inburst] *= factor
    else:
        raise ValueError("Valid choices for burstmode are sfe or mdot")
    return taustar_a,eta_a,mdotinf

def bolus(t,mdotstar,mgas,taustar_a,tc,factor,dt,taudep):
    """
    Add bolus of gas, inducing a burst of star formation 
    t, mdotstar, mgas, taustar_a = pre-existing arrays
    tc = time at which to add gas
    factor = factor by which mgas is multiplied at tc
    dt = integration timestep, used to figure out which index corresponds to tc
    taudep = depletion time, used for solution of gas evolution

    Gas evolution s based on dhw analytic solution of 9/4/2016 for 
    unchanged mdotinf.  Assumes constant taudep.
    """
    deltamg = 0.*mgas			# array of same length as mgas
    itc = int(tc/dt)
    deltam0 = (factor-1)*mgas[itc]
    efactor = (t[t>=tc]-tc)/taudep
    mgas[t>=tc] += deltam0*np.exp(-efactor)
    mdotstar = mgas/taustar_a
    return mdotstar,mgas

