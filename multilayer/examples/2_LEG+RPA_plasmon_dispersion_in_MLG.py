"""
  Plot script for the dispersion of the pi-plasmon in N-layer 
  gaphene with in-plane momentum q (see [1] Fig. 2c).

  REFERENCES: 
    [1] Wachsmuth, Hambach, Benner, and Kaiser, PRB (90) 235434 (2014)

  Copyright (c) 2014, rhambach, P. Wachsmuth. 
    This file is part of the EELcalc package and released
    under the MIT-Licence. See LICENCE file for details.
"""
import numpy as np;
import matplotlib.pylab as plt;
import scipy.stats as stats;
from   scipy.optimize import curve_fit;

from EELcalc.monolayer.graphene import rpa;
from EELcalc.multilayer.leg import areels;
from EELcalc.tools.DPio.dp_mdf import broad;

dispersion=[];
_debug= True;
BOHR = 0.529177; # [Angstrom]

def _fitplot(E,spectrum,mask,fit):
  ax=_fitplot.ax; 
  offset=_fitplot.count; 
  ax.plot(E,spectrum+offset,'k-');
  ax.plot(E[mask],spectrum[mask]+offset,'b-');
  ax.plot(E[mask],fit+offset,'r-');
  _fitplot.count+=0.2;
if _debug:
  _fitplot.count=0;
  _fitplot.ax=plt.figure().add_subplot(111);
  plt.title(u'DEBUG: test fitting of $\pi$-plasmon position')

def find_max(spectrum,E,Emin=0,Emax=np.inf,debug=False):
  # finding maximum in spectrum within fixed energy range [eV]
  mask  = np.logical_and(Emax>E, E>Emin);
  imax  = spectrum[mask].argmax();
  A0 = (spectrum[mask])[imax]; 
  E0 = (E[mask])[imax];
  if debug:  _fitplot(E,spectrum/A0,mask,np.arange(sum(mask))==imax);
  return (A0,E0);
	
def fit_lorentz(spectrum,E,dE,Emin=0,Emax=np.inf,debug=False):
  # fit lorentz function in a range +-dE around the maximum
  # returns fit parameters [a,x0,sigma]
  A0,E0 = find_max(spectrum,E,Emin=Emin,Emax=Emax); # get starting point
  mask = np.logical_and(E0+dE>E, E>E0-dE);
  f = lambda x,a,x0,sigma:  a/((x-x0)**2 + sigma**2);
  popt, pcov = curve_fit(f, E[mask], spectrum[mask],p0=(A0,E0,dE))
  if debug: _fitplot(E,spectrum/A0,mask,f(E[mask],*popt)/A0);
  return popt


# setup calculations
Nmax=6;
DP = rpa.GrapheneDP(qdir='GM',verbosity=0);
d = 3.334 / BOHR;        # interlayer distance [a.u]
q_calc = np.sort((DP.get_q()));
q_calc = q_calc[q_calc<0.5]; # restrict range of q

# iterate over different multilayers
fig=plt.figure(); ax=fig.add_subplot(111);
cm=plt.cm.cool;              # colors
colors = ['k']+[ cm(x) for x in np.linspace(0,1,Nmax) ];
for N in [0]+range(Nmax,0,-1):# 0=graphite
  print " calculating %d-layer system ..."%N;
  
  if N==0:  ML = areels.InfiniteStack(d,DP.get_pol2D);  # graphite 
  else:     ML = areels.Multilayer(N,d,DP.get_pol2D);   # multilayer

  disp=[];
  for q in q_calc:
    eels = ML.areel(q,0);  # qz=0, only in-plane spectra
    if q<0.01:             # lorentz fit does not work for q=0
      A0,E0=find_max(eels,DP.E,Emin=3,Emax=8,debug=_debug);
    else:
      A,E0,width=fit_lorentz(eels,DP.E,0.7,Emin=2,Emax=13,debug=_debug);
    disp.append([q/BOHR,E0]); # q [1/A], E0 [eV]   
  disp=np.asarray(disp);

  plt.plot(disp[:,0],disp[:,1],c=colors[N],label='N=%d'%N if N>0 else 'graphite');

plt.title('Dispersion in multilayer graphene'); 
plt.suptitle('(layered-electron-gas model + RPA ab-initio calculations for graphene polarizability)');
plt.xlabel('momentum transfer q [1/A]');
plt.ylabel('energy [eV]');

plt.legend(loc=4);
plt.show();

