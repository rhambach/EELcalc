"""
  Hydrodynamic model for the response of two-dimensional systems.
  (similar to Lorentz-model). For graphene, we use parameters from [1].

  REFERENCES:
   [1] Jovanovic, Radovic, Borka, and Miskovic, PRB(84) 155416 (2011)

  Copyright (c) 2014, rhambach. 
    This file is part of the EELcalc package and released
    under the MIT-Licence. See LICENCE file for details.
"""
import numpy as np;

HARTREE = 27.2114; # eV
BOHR    = 0.529177;# Angstrom

def coul2D(q):
  return (2*np.pi)/q;
  
  
class one_fluid_hydrodynamic_model:
  """ 
  Parameters for the one_fluid hydrodynamic model
    pol(q,w) = A / ( w^2 + i*w*gamma - w0^2 ),
  where 
    A  = n*q^2,      w0^2 =  wr^2 + (s*q)^2 
  and the following parameters are given (in atomic units)
    wr   ... resonance frequency [Hartree] 
    gamma... damping [Hartree]
    n    ... two-dimensional electron density [bohr^-2]
    s    ... acoustic speed [?]
  the energy-axis w can be specified as optional argument
    w    ... energy-axis [Hartree]
  """  
  def __init__(self,wr=0.1,gamma=0.01,n=0.01,s=0,w=None):
    self.wr    = wr;
    self.gamma = gamma;
    self.n     = n;
    self.s     = s;
    if w is not None: self.set_E(w);

  def set_E(self,w):
    " w ... energy-axis [Hartree]"
    self.w = np.asarray(w,dtype=complex);

  def pol(self,q,w=None):
    """
    Return polarisability for momentum q and energy w in atomic units
    """
    w  = self.w if w is None else np.asarray(w,dtype=complex);
    A  = self.n * q**2;
    w02= self.wr**2 + (self.s*q)**2;
    return A / ( w**2 + 1j*w*self.gamma - w02 );

  def eps(self,q,w=None):
    return 1-coul2D(q)*self.pol(q,w);


class multipole_model:
  """ 
  Generalisation of the one-fluid hydtrodynamical model 
  for arbitrary number of independent fluids given by
  
    Pol(q,w) = Sum_n  Pol_n(q,w)
 
  where Pol_n(q,w) is the polarisability of the n'th fluid.
  """
  
  def __init__(self, pol_list, w=None):
    """ 
    pol_list ... list of polarisabilities for each fluid 
    w        ... (opt) list of energies [a.u.]
    """
    self.n = len(pol_list);
    self.pol_list = pol_list;
    if w is not None: self.set_E(w);

  def set_E(self,w):
    " w ... energy-axis [Hartree]"
    self.w = np.asarray(w,dtype=complex);

  def pol(self,q,w=None):
    """
    Return polarisability for momentum q and energy w in atomic units
    """
    w  = self.w if w is None else np.asarray(w,dtype=complex);
    pol= 0.j * w;   # create complex array of zeros (dtype argument in 
                    #   numpy.zeros_like() only available since v.1.6.0)
    for fluid in self.pol_list:
      pol += fluid.pol(q,w);
    return pol;

  def eps(self,q,w=None):
    return 1-coul2D(q)*self.pol(q,w);



def get_2FHM_Graphene_Jovanovic(gamma=None,w=None):
    """ 
     Return multipole_model instance for polarisability
     of graphene within the two-fluid hydrodynamical model
     from [Jovanovic, et.al. PRB(84) 155416 (2011)]

       gamma ... (opt) broadening [Hartree]
       w     ... (opt) energy axis [Hartree]
    """

    V = np.sqrt(3)/2 * 4.63**2;   # unit cell volume in graphene

    # Pi-electrons
    wr    = 4.08 / HARTREE;
    gamma = 2.45 / HARTREE if gamma is None else gamma;
    n     = 2/V;
    s     = 0;
    Pol_pi = one_fluid_hydrodynamic_model(wr,gamma,n,s);

    # Sigma-electrons
    wr    = 13.06 / HARTREE;
    gamma = 2.72 / HARTREE if gamma is None else gamma;
    n     = 6/V;
    s     = 0;
    Pol_sigma = one_fluid_hydrodynamic_model(wr,gamma,n,s)

    return multipole_model([Pol_pi, Pol_sigma],w=w);

# -----------------------------------------------------------------
if __name__ == "__main__":
  import matplotlib.pylab as plt;

  # two-fluid hydrodynamic model for graphene
  E = np.arange(0,40,0.1);   # energy axis [eV]
  Graphene = get_2FHM_Graphene_Jovanovic(w=E/HARTREE)
  q = 0.7 * BOHR;            # momentum transfer [1/bohr]
  eps = Graphene.eps(q);
  plt.title('Graphene, two-fluid hydrodynamic model, q=%5.3f 1/A'%(q/BOHR));
  plt.plot(E,eps.real,'k--',label=u'Re $\epsilon$');
  plt.plot(E,eps.imag,'k-', label=u'Im $\epsilon$');
  plt.plot(E,-(1./eps).imag,label=u'-Im $\epsilon^{-1}$');
  plt.legend(loc=0);
  plt.xlabel('energy [eV]');

  # plasmon map for single-plasmon model
  GraphenePI=Graphene.pol_list[0];
  q = np.arange(0.001,1.01,0.01)*BOHR;
  eels = [ -(1./GraphenePI.eps(_q,w=E/HARTREE)).imag for _q in q ];
  plt.figure();
  plt.title('Plasmon Dispersion for one-pole Lorentz model'%(q/BOHR));
  plt.imshow(np.transpose(eels),extent=[0,1,0,40],cmap='Blues',
                                        origin='ll',aspect='auto');
  # theory
  omega = np.sqrt( GraphenePI.wr**2 + 2*np.pi*GraphenePI.n*q );
  plt.plot(q/BOHR,omega*HARTREE,'DarkOrange',label=u'plasmon resonance (Re $\epsilon=0$)');
  plt.legend(loc=0);
  plt.ylabel('energy [eV]');
  plt.xlabel('q [1/A]');
  plt.xlim(xmax=1);
  plt.show();
  
