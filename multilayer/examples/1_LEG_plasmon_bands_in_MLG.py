"""
  Plot plasmon dispersion for N-layer graphene using the
  hydrodynamic model of [1] for the dielectric response of 
  graphene. The plasmon dispersion is evaluated analytically
  in the limit of vanishing broadening. Reproduces Fig. 2a in [2].

  REFERENCES: 
    [1] Jovanovic, Radovic, Borka, and Miskovic, PRB(84) 155416 (2011)
    [2] Wachsmuth, Hambach, Benner, and Kaiser, PRB (90) 235434 (2014)

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
q_range = np.arange(1e-6,0.7,0.01); # [1/A]

def w_dispersion(q,v=1):
  """
  Implementation of eq. (17) in [1] (using atomic units e=me=hbar=1)

  Dispersion relation for eigenvalue v(q) of the Coulomb matrix V(q)
     w^2 = 0.5(w1^2+Q1^2 + w2^2+Q2^2) 
           +- Sqrt [ ( 0.5(w1^2+Q1^2 - w2^2-Q2^2) )^2 + ( Q_1 Q_2 )^2  ]
  where
     Q1^2=2pi*n1*q*v (and Q2^2 analogous) is the generalized plasma freq.
     w1  = wr1 is the restoring frequency of the pi electons 
     w2  = wr2                "               sigma electrons

  RETURNS: [w01,w02]
     the two plasmon frequencies (pi and pi+sigma) for each q
  """
  # parameters for two-fluid hydrodynamic model from [1]
  Vol  = np.sqrt(3)/2 * 4.63**2; # unit cell volume in graphene
  wr1= 4.08 / HARTREE;           # Pi-electrons [eV]
  n1 = 2/Vol;
  wr2= 13.06 / HARTREE;          # Sigma-electrons [eV]
  n2 = 6/Vol;
  
  # resonance frequencies
  w12 = wr1**2; # we neglect the acoustic velocity s=0
  w22 = wr2**2;

  # generalized plasma frequencies
  Q12 = 2*np.pi*n1*q * v ;       # effective Omega_nu^2
  Q22 = 2*np.pi*n2*q * v ;

  # dispersion formula (17) in [1]
  A = 0.5*(w12 + Q12 + w22 + Q22);
  B = np.sqrt( 0.25*( w12 + Q12 - w22 - Q22 )**2 + Q12 * Q22  );

  return np.asarray([np.sqrt(A-B), np.sqrt(A+B)]);



# -------------------------------------------------------
# plotting
print " calculating %d-layer system ..."%N;
ML= areels.Multilayer(N,d/BOHR,None);   # multilayer
cm=plt.cm.spectral;                     # colors
colors = [ cm(x) for x in np.linspace(0,1,N) ];

# calculate dispersion for q_range
disp_w = []; disp_v = [];
for q in q_range*BOHR:     # a.u.
  v,_=ML.normal_modes(q);  # N eigenvalues
  w1,w2=w_dispersion(q,v); # N plasmon-frequencies for pi and pi+sigma
  disp_w.append([w1,w2]); 
  disp_v.append(v);
disp_w=np.asarray(disp_w);   # shape (Nq,2,N);
disp_v=np.asarray(disp_v);   # shape (Nq,N);

# plot energy dispersion for pi and pi+sigma plasmon
fig=plt.figure(); ax=fig.add_subplot(111);
for l in range(N):
  plt.plot(q_range,disp_w[:,0,l],color=colors[l],label='l=%d'%l);
  plt.plot(q_range,disp_w[:,1,l],color=colors[l]);

plt.title('Plasmon-bands in multilayer graphene'); 
plt.suptitle('(layered-electron-gas model + hydrodynamic model for graphene polarizability)');
plt.xlabel('momentum transfer q [1/A]');
plt.ylabel(u'energy $\omega_l$ [eV]');
plt.legend(loc=0);


# plot dispersion of eigenvalues v of the Coulomb matrix
fig=plt.figure(); ax=fig.add_subplot(111);
for l in range(N):
  plt.plot(np.exp(-d*q_range),disp_v[:,l],color=colors[l],label='l=%d'%l);

plt.title('Dispersion of eigenvalues of the coupling matrix V(q)');
plt.xlabel('coupling constant exp(-qd)');
plt.ylabel(u'eigenvalues $v_l(q)$');
plt.legend(loc=0);
plt.show();

