"""
  Plot weigths for the normal-mode decomposition of AR-EEL 
  spectra of multilayer systems. Reproduces Fig. 2b in [1].

  REFERENCES: 
    [1] Wachsmuth, Hambach, Benner, and Kaiser, PRB ()...(2014)

  Copyright (c) 2014, rhambach, P. Wachsmuth. 
    This file is part of the EELcalc package and released
    under the MIT-Licence. See LICENCE file for details.
"""
import numpy as np;
import matplotlib.pylab as plt;
from EELcalc.multilayer.leg import areels;

BOHR = 0.529177; # [Angstrom]
HARTREE = 27.2114; # eV
N = 3;           # number of layers
d = 3.333;       # interlayer distance [A]
qz_range = np.arange(0,2*np.pi/d,0.01); # [1/A]
q = 0.1; # [1/A] 
# -------------------------------------------------------
# plotting
print " calculating %d-layer system ..."%N;
ML= areels.Multilayer(N,d/BOHR,None);   # multilayer
cm=plt.cm.spectral;                     # colors
colors = [ cm(x) for x in np.linspace(0,1,N) ];

# calculate weights for q_range
weights = [];
for qz in qz_range*BOHR:     # a.u.
  weights.append(ML.arweights(q*BOHR,qz));
weights=np.asarray(weights); # shape (Nq,N);

# plot weights for different qz
fig=plt.figure(); ax=fig.add_subplot(111);
for l in range(N):
  plt.plot(qz_range,weights[:,l],color=colors[l],label='l=%d'%l);

plt.title('Weights of normal-modes in AR-EEL spectrum for %d-layer graphene'%N); 
plt.xlabel(u'on-axis momentum transfer $q_z$ [1/A]');
plt.ylabel(u'weights $|\\tilde u(q,q_z)|^2$');
plt.legend(loc=0);
plt.show();

