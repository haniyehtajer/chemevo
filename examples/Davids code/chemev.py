#! /Users/weinberg.21/anaconda3/bin/python
"""
chemev.py -- evolve a one-zone chemical evolution model, oxygen and iron
chemev.py case outfile [case-dependent parameters]
    case = case to set up; 'options' will give list of options and arguments
    outfile = file for output
Time unit is Gyr
Mass unit is Msun
SFR  unit is Msun/yr
"""

import numpy as np
import chemev_subs as cs
import sys

dtd='exp'	        # form for SNIa DTD: exp or plaw
tauIa=1.5		# e-folding time for exponential DTD
tdmin=0.15		# minimum delay time

mocc=0.015		# IMF-averaged CCSN oxygen yield
mfecc=0.0015		# IMF-averaged CCSN iron yield
mfeIa=0.0013		# SNIa iron yield over time interval tmax
r=0.4			# recyling fraction

SolarO=0.0056		# solar oxygen abundance by mass
SolarFe=0.0012		# solar iron abundance by mass

dt=0.002		# timestep for integration
dtout=0.10 		# timestep for outputs
tmax=12.0		# maximum time
mdotfid=1.0		# fiducial SFR in Msun/year

if (len(sys.argv)>1):
    case=sys.argv[1]
else:
    case='options'

if ((case=='options') | (case=='Options')):
    print("\n")
    print("Available cases and arguments are")
    print("Simple: taustar, eta, tausfh, sfhmode")
    print("RiseFall: taustar, eta, tau1, tau2")
    print("TwoPhase: taustar1, eta1, tausfh1, taustar2, tausfh2, eta2, tc")
    print("PLSchmidt: taustar, eta, tausfh sfhmode")
    print("FixedMgasBurst: taustar, eta, tausfh, sfhmode t1 t2 factor etaburst burstmode")
    print("GasBolus: taustar, eta, tausfh, tc, factor")
    print("ConstantInfall: taustar, eta, mdotinf, mdotstar0")
    print("ConstantInfallBurst: taustar, eta, mdotinf, mdotstar0, t1, t2, factor, burstmode")
    print("MultiBurst: taustar eta mdotinf mdotstar0 t1 dt1 dt2 factor burstmode")
    print("\n")
    sys.exit()

if (len(sys.argv)>2):
    outfile=sys.argv[2]

# Default is fixed SFH, not fixed mdotinfall.  Changes to True for some cases.
infall=False

if (case=='Simple'):
    if (len(sys.argv)!=7):
        sys.exit("Args: taustar eta tausfh sfhmode")
    taustar0=float(sys.argv[3])
    eta0=float(sys.argv[4])
    tausfh=float(sys.argv[5])
    sfhmode=sys.argv[6]
elif (case=='RiseFall'):
    if (len(sys.argv)!=7):
        sys.exit("Args: taustar eta tau1 tau2")
    taustar0=float(sys.argv[3])
    eta0=float(sys.argv[4])
    tausfh1=float(sys.argv[5])
    tausfh2=float(sys.argv[6])
elif (case=='TwoPhase'):
    if (len(sys.argv)!=10):
        sys.exit("Args: taustar1 eta1 tausfh1 taustar2 eta2 tausfh2 tc")
    taustar1=float(sys.argv[3])
    eta1=float(sys.argv[4])
    tausfh1=float(sys.argv[5])
    taustar2=float(sys.argv[6])
    eta2=float(sys.argv[7])
    tausfh2=float(sys.argv[8])
    tc=float(sys.argv[9])
    sfhmode='exp'
elif (case=='PLSchmidt'):
    if (len(sys.argv)!=7):
        sys.exit("Args: taustar eta tausfh sfhmode")
    taustar0=float(sys.argv[3])
    eta0=float(sys.argv[4])
    tausfh=float(sys.argv[5])
    sfhmode=sys.argv[6]
elif (case=='FixedMgasBurst'):
    if (len(sys.argv)!=12):
        sys.exit("Args: taustar eta tausfh sfhmode t1 t2 factor etaburst burstmode")
    taustar0=float(sys.argv[3])
    eta0=float(sys.argv[4])
    tausfh=float(sys.argv[5])
    sfhmode=sys.argv[6]
    t1=float(sys.argv[7])
    t2=float(sys.argv[8])
    factor=float(sys.argv[9])
    etaburst=float(sys.argv[10])
    burstmode=sys.argv[11]
elif (case=='GasBolus'):
    if (len(sys.argv)!=9):
      sys.exit("Args: taustar eta tausfh tc factor")
    taustar0=float(sys.argv[3])
    eta0=float(sys.argv[4])
    tausfh=float(sys.argv[5])
    sfhmode=sys.argv[6]
    tc=float(sys.argv[7])
    factor=float(sys.argv[8])
elif (case=='ConstantInfall'):
    if (len(sys.argv)!=7):
        sys.exit("Args: taustar eta mdotinf mdotstar0")
    taustar0=float(sys.argv[3])
    eta0=float(sys.argv[4])
    mdotinf0=float(sys.argv[5])
    mdotstar0=float(sys.argv[6])
    infall=True
elif (case=='ConstantInfallBurst'):
    if (len(sys.argv)!=11):
        sys.exit("Args: taustar eta mdotinf mdotstar0 t1 t2 factor burstmode")
    taustar0=float(sys.argv[3])
    eta0=float(sys.argv[4])
    mdotinf0=float(sys.argv[5])
    mdotstar0=float(sys.argv[6])
    t1=float(sys.argv[7])
    t2=float(sys.argv[8])
    factor=float(sys.argv[9])
    burstmode=sys.argv[10]
    infall=True
elif (case=='MultiBurst'):
    if (len(sys.argv)!=12):
        sys.exit("Args: taustar eta mdotinf mdotstar0 t1 dt1 dt2 factor burstmode")
    taustar0=float(sys.argv[3])
    eta0=float(sys.argv[4])
    mdotinf0=float(sys.argv[5])
    mdotstar0=float(sys.argv[6])
    t1=float(sys.argv[7])
    dt1=float(sys.argv[8])
    dt2=float(sys.argv[9])
    factor=float(sys.argv[10])
    burstmode=sys.argv[11]
    infall=True
else:
    sys.exit("Invalid, run chemev.py options for available cases")

t=np.arange(0.0,tmax+dt,dt)		# time array

# allocate arrays
nstep=int(tmax/dt)
mgas=		np.zeros(nstep+1,dtype=float)	# gas mass
mdotstar=	np.zeros(nstep+1,dtype=float)	# star formation rate
mdotinf=	np.zeros(nstep+1,dtype=float)	# infall rate
taustar=	np.zeros(nstep+1,dtype=float)	# star formation efficiency
eta=		np.zeros(nstep+1,dtype=float)	# outflow mass loading
foxy=		np.zeros(nstep+1,dtype=float)	# ISM oxygen mass fraction
firon=		np.zeros(nstep+1,dtype=float)	# ISM iron mass fraction
yd=		np.zeros(nstep+1,dtype=float)	# Y_deut/Y_deut,primordial
RIa=		np.zeros(nstep+1,dtype=float)	# SNIa delay time distribution

mdotfid *= 1.e9				# convert from Msun/yr to Msun/Gyr
RIa = cs.SetupSNIa(t,tauIa,tdmin,dtd)

if (case=='Simple'):
    taustar,eta,mdotstar,mgas = \
        cs.SetupSimple(mdotfid,t,taustar0,eta0,tausfh,sfhmode)
elif (case=='RiseFall'):
    taustar,eta,mdotstar,mgas = \
        cs.SetupRiseFall(mdotfid,t,taustar0,eta0,tausfh1,tausfh2)
elif (case=='TwoPhase'):
    taustar,eta,mdotstar,mgas = \
        cs.SetupTwoPhase(mdotfid,t,tc,tausfh1,taustar1,eta1,
	                 tausfh2,taustar2,eta2)
elif (case=='PLSchmidt'):
    taustar,eta,mdotstar,mgas = \
        cs.SetupPLSchmidt(mdotfid,t,tausfh,taustar0,eta0,sfhmode)
elif (case=='FixedMgasBurst'):
    taustar,eta,mdotstar,mgas = \
        cs.SetupSimple(mdotfid,t,taustar0,eta0,tausfh,sfhmode)
    inburst = (t>=t1)
    inburst = inburst & (t<=t2)
    mdotstar[inburst] *= factor
    if (burstmode=='gas'):
        mgas[inburst] *= factor
    elif ((burstmode=='sfe') | (burstmode=='SFE')):
        taustar[inburst] /= factor
    else:
        raise ValueError("Valid choices for burstmode are gas or sfe")
    # negative value of etaburst means don't change eta during burst
    if (etaburst >=0.):		
        eta[inburst] = etaburst
elif (case=='GasBolus'):
    taustar,eta,mdotstar,mgas = \
        cs.SetupSimple(mdotfid,t,taustar0,eta0,tausfh,sfhmode)
    taudep=taustar0/(1.+eta0-r)
    mdotstar,mgas = cs.bolus(t,mdotstar,mgas,taustar,tc,factor,dt,taudep)
elif (case=='ConstantInfall'):
    mdotinf0 *= 1.e9			
    taustar,eta,mdotinf = cs.SetupConstantInfall(t,taustar0,eta0,mdotinf0)
    mgas[0]=1.e9*mdotstar0*taustar[0]
elif (case=='ConstantInfallBurst'):
    mdotinf0 *= 1.e9			
    taustar,eta,mdotinf = cs.SetupConstantInfall(t,taustar0,eta0,mdotinf0)
    mgas[0]=1.e9*mdotstar0*taustar[0]
    inburst = (t>=t1)
    inburst = inburst & (t<=t2)
    if ((burstmode=='sfe') | (burstmode=='SFE')):
        taustar[inburst] /= factor
    elif ((burstmode=='mdot') | (burstmode=='Mdot')):
        mdotinf[inburst] *= factor
    else:
        raise ValueError("Valid choices for burstmode are sfe or mdot")
elif (case=='MultiBurst'):
    mdotinf0 *= 1.e9
    taustar,eta,mdotinf = cs.SetupMultiBurst(t,taustar0,eta0,mdotinf0,
                                             t1,dt1,dt2,factor,burstmode)
    mgas[0]=1.e9*mdotstar0*taustar[0]
else:
    raise ValueError("Invalid case, run chemev.py options for available cases")

# for scenarios in which infall history rather than star formation history
# is specified, integrate to get gas supply mgas(t), and compute mdotstar(t)
if (infall):
    for i in range(1,nstep+1):
        mdotstar[i-1]=mgas[i-1]/taustar[i-1]
        mgas[i]=mgas[i-1]+dt*(mdotinf[i-1]-(1+eta[i-1]-r)*mdotstar[i-1])
    mdotstar[nstep]=mgas[nstep]/taustar[nstep]
i=0
moxy=0.0
miron=0.0
foxy[0]=0.0
firon[0]=0.0
if (case=='MultiBurst'):		# start this case out at equilibrium
    moxy = mgas[0]*mocc/(1.+eta[0]-r)
    miron = mgas[0]*(mfecc+mfeIa)/(1.+eta[0]-r)
    foxy[0] = moxy/mgas[0]
    firon[0] = miron/mgas[0]
mdotinf[0]=(mgas[1]-mgas[0])/dt + (1+eta[0]-r)*mdotstar[0]
yd[0]=1.0		# Normalized so that 1.0 = primordial abundance
mdeut=yd[0]*mgas[0]

for i in range(1,nstep+1):
    moxy += dt*(mocc*mgas[i-1]-(1+eta[i-1]-r)*moxy)/taustar[i-1]
    mdotstarIa=0.0
    for j in range(i):				# compute <Mdotstar>_{Ia}
        mdotstarIa += RIa[i-j]*mdotstar[j]
    miron += dt*(mfecc*mgas[i-1]-(1+eta[i-1]-r)*miron)/taustar[i-1] + \
             dt*mfeIa*mdotstarIa
    foxy[i]=moxy/mgas[i]
    firon[i]=miron/mgas[i]
    mdotinf[i]=(mgas[i]-mgas[i-1])/dt + (1+eta[i-1]-r)*mdotstar[i-1]
    mdeut += dt*mdotinf[i-1]-dt*yd[i-1]*(1+eta[i-1])*mdotstar[i-1]
    yd[i]=mdeut/mgas[i]

foxy = np.log10(foxy/SolarO+1.e-6)
firon = np.log10(firon/SolarFe+1.e-6)
ofe=foxy-firon

# convert back to Msun/yr units for output
mdotstar /= 1.e9	
mdotinf /= 1.e9	

# output every nth step
if (dtout>dt):
    n=int(dtout/dt)
else:
    n=1

np.savetxt(outfile,
           np.transpose([t[n::n],mdotstar[n::n],firon[n::n],ofe[n::n],yd[n::n],
	                mgas[n::n],mdotinf[n::n],eta[n::n],taustar[n::n]]),
           '%6.3f %6.3f %7.3f %7.3f %6.3f %8.3e %6.3f %6.3f %6.3f')
