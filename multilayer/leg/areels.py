"""
  Angular-Resolved EELS for layered systems using the
  layered-electron-gas model. For the theory, see [1,2].

  DESCRIPTION

    The Multilayer class implements all equations of [1,2]
      for the calculation of AR-EELS spectra for a N-layer 
      system within the layered-electron-gas model. The 
      polarizability of each individual layer has to be 
      provided (see EELcalc.monolayer module) and must be 
      the same for all layers.

    The InfiniteStack class implements the special case
      of a bulk-system of infinitely many layers.

    Execute this script for self-tests and see ../examples
    for several example calculations for multilayer graphene
    reproducing our results in [1].  

  REFERENCES

    [1] Wachsmuth, Hambach, Benner, and Kaiser, PRB (90) 235434 (2014)
    [2] Supplemental Material of [1]

  Copyright (c) 2014, rhambach. 
    This file is part of the EELcalc package and released
    under the MIT-Licence. See LICENCE file for details.
"""
import numpy as np;

def coul2D(q):
  return (2*np.pi)/q;

class Multilayer:
  """
    Implements Layered electron-gas model for a stack of N 
    sheets with distance d and the _same_ in-plane polariz-
    ability pol for each layer. Atomic units are used.
  """

  def __init__(self,N,d,pol,verbosity=1):
    """
    N   ... number of layers
    d   ... distance of sheets [bohr]
    pol ... function pol(q) which returns the in-plane
            polarisability of each slab for momentum q
    verbosity...(0) silent, (1) print warnings, (3) debug
    """
    self.N = N;
    self.d = d;
    self.pol = pol;
    self.verbosity=verbosity;
  
  def coulomb_matrix(self,q):
    """
    Construct NxN Coulomb matrix  V_nm = c^|n-m|
    with the coupling constant c = exp(-qd) for
      q    ... in-plane momentum transfer [a.u.]

    RETURNS:  V_nm 
    """
    # prepare coulomb pot: coul[n,m] == cpow[N+n-m]
    N = self.N;
    cpow = np.exp(-q*self.d) ** np.abs(np.arange(N,-N,-1));
    coul = np.asarray([ cpow[N-i:2*N-i] for i in range(N)]);
    return coul;
  
    ## alternative calculation for comparison
    #c = np.exp(-q*self.d);
    #coul2 = [[ c**np.abs(n-m) for m in range(N)] for n in range(N)];
    #assert np.allclose(coul, coul2)
  
  def normal_modes(self,q):
    """
    Calculate eigenspectum of the Coulomb matrix V_nm
      q    ... in-plane momentum transfer [a.u.]
    
    RETURNS: v[:], U[:,:]
      ordered list of eigenvalues v[l] and corresponding
      eigenvectors U[:,l]
    """
    if q==0 and self.verbosity>0:
      print "WARNING in Multilayer.normal_modes(q): ",
      print "   singular behaviour of coupling matrix for q=0";
    v,U = np.linalg.eig(self.coulomb_matrix(q));
    ind = np.argsort(-v);  # l=0 should be largest v_l (symmetric mode)
    return v[ind], U[:,ind];

  def arweights(self,q,qz,U=None):
    """
    For the N-layer graphene system, we calculate the weights 

      |u^l(q,qz)|^2 = |\sum_n u^l_n(q) exp [ -i n*d*qz ]|^2 
    
    of each normal-mode excitation l to the total electron 
    energy-loss spectrum with momentum transfer (q,qz). 
    Note: for details, see [2] eq. (S10).

      q ... in-plane momentum transfer [a.u.]
      qz... out-of-plane momentum transfer [a.u.]
      U ... (opt) eigenvectors of Coulomb matrix

    RETURNS: list of weights [shape: (N)]
    """
    if U is None: _,U=self.normal_modes(q); # U(n,l)
    FTline = np.exp(-1.j*qz*self.d) ** np.arange(self.N);
    weights= np.tensordot(FTline,U,axes=(-1,0));
    return np.real(weights*weights.conjugate());

  def sus_matrix(self,q):
    """
    Calculate the susceptibility matrix sus_nm of the multilayer for
      q    ... in-plane momentum transfer [a.u.]
    
    RETURNS array of shape(nomega, N, N)
    """
    # sus_nm = pol / eps_nm = [ 1/pol - coul2D * V_nm ]^-1
    polinv = np.outer(1./self.pol(q), np.identity(self.N));   # nomega x N^2
    susinv = polinv.reshape(-1,self.N,self.N) \
               - coul2D(q)*self.coulomb_matrix(q);            # nomega x N x N
    for io in range(susinv.shape[0]):                         # nomega
      susinv[io] = np.linalg.inv(susinv[io]);                 # invert NxN matrix
    return susinv;  # contains actually sus after inversion

  def areel(self,q,qz):
    """ 
    Calculate energy-loss probability for parameters 
      q    ... in-plane momentum transfer [a.u.]
      qz   ... out-of-plane momentum [a.u.] 
    using eq. (S9) of [2]

    RETURNS array with same shape as energy axis E
    """
    # prepare matrix for Fourier transform FT=e^{-iqzd(n-m)}
    # analogous to the Coulomb matrix, FT[n,m] == FTline[N+n-m]
    N = self.N;
    FTline = np.exp(-1.j*qz*self.d) ** np.arange(N,-N,-1);
    FT     = np.asarray([ FTline[N-i:2*N-i] for i in range(N)]);
    ## alternative calculation
    #FT2 = [[ np.exp(-1.j*qz*self.d)**(n-m) for m in range(N)] for n in range(N)];
    #assert np.allclose(FT,FT2)
 
    # calculate susceptibility in real-space z=nd, z'=md
    sus_zzp = self.sus_matrix(q);
    nomega  = sus_zzp.shape[0];

    # perform Fourier transform for diagonal element qz=qz'
    sus_qz = np.empty(nomega,dtype=complex);
    for io in range(nomega):
      sus_qz[io] = np.sum( FT * sus_zzp[io] );     

    # return areels spectrum: -sus_qz.imag * coul3D^2
    return - sus_qz.imag * ( (4*np.pi)/(q**2 + qz**2) )**2;

  def areel_lmodes(self,q,qz):
    """ 
    Calculate all weighted normal-mode contributions to the
    total energy-loss probability of the multlayer for parameters 
      q    ... in-plane momentum transfer [a.u.]
      qz   ... out-of-plane momentum [a.u.] 
    using eq. (S5) and (S10) in [2].

    RETURNS array of shape (N,len(E))
    """
    pol2D = self.pol(q);
    # eigenvalues of dielectric matrix [see [2] eq. (S5)]
    v_l,U = self.normal_modes(q);                # eigenspectrum of Coulomb matrix
    eps_l = 1 - coul2D(q)*np.outer(v_l,pol2D);   # shape: (N,len(E))
    # calculate AR-EEL spectrum [see [2] eq. (S10)]
    coul3D= (4*np.pi)/(q**2 + qz**2);            # prefactor (Rutherford cross-section)
    weights=self.arweights(q,qz,U=U).reshape(self.N,1); # weight of each normal mode
    sus = weights * (pol2D / eps_l);
    return - coul3D**2 * sus.imag;               # shape(N,len(E));


class InfiniteStack(object):
  """
    Implements Layered-Electron-Gas model for Graphite, 
    i.e., an infinite stack of sheets with distance d and 
    in-plane polarisability pol. Atomic units are used.
  """

  def __init__(self,d,pol):
    """
    d   ... distance of sheets [bohr]
    pol ... function pol(q) which returns the in-plane
            polarisability of the slabs for momentum q
    """
    self.d = d;
    self.pol = pol;

  def _S(self,q,qz):
    "Structure factor for infinite stack (eigenvalue of Coulomb matrix)"
    return np.sinh(self.d*q) / (np.cosh(self.d*q) - np.cos(self.d*qz));

  def areel(self,q,qz):
    """ 
    Calculate energy-loss probability for parameters 
      q    ... in-plane momentum transfer [a.u.]
      qz   ... out-of-plane momentum [a.u.] 
 
    RETURNS array with same shape as energy axis E
    """
    pol2D = self.pol(q);
    eps2D = 1 - coul2D(q)*pol2D*self._S(q,qz);
    coul3D= (4*np.pi)/(q**2 + qz**2);
    return - coul3D**2 * (pol2D/eps2D).imag;


# -----------------------------------------------------------------
if __name__ == "__main__":
  import matplotlib.pylab as plt;
  import EELcalc.monolayer.graphene.hydrodynamic_model as hm;

  N = 6;
  d = 3.334 / hm.BOHR;     # interlayer distance [a.u]
  E = np.arange(0,40,0.1); # energies [eV]

  Graphene = hm.get_2FHM_Graphene_Jovanovic(w=E/hm.HARTREE);#gamma=0.1 / hm.HARTREE);
  Graphite = InfiniteStack(d,Graphene.pol);
  ML = Multilayer(N,d,Graphene.pol,verbosity=0);

  # -- test Normal-modes of Multilayer for infinite coupling and zero coupling
  print "testing normal modes of Coulomb matrix ..."
  v,U=ML.normal_modes(0);       # strong coupling eigenspectrum:
  assert np.allclose(v[0],N);       #   1st  eigenvalue  N
  assert np.allclose(v[1:N],0);     #  (N-1) eigenvalues 0
  
  v,U=ML.normal_modes(1000/d);  # no coupling eigenspectrum:
  assert np.allclose(v,1);          #  N eigenvalues 1
  assert np.allclose(U,np.eye(N));  #  U is identity matrix

  v,U=ML.normal_modes(1/d);     # test order of eigenmodes
  sU =np.sign(U)/np.sign(U[0,0]);
  assert np.allclose(sU[:,0],1);    # symmetric mode
  assert np.allclose(sU[:,N-1],(-1)**np.arange(N)); #antisymmetric mode

  # -- test weights
  q=0.1; qz=[0.1,0.5,2*np.pi/d+0.1];
  w1 = [ ML.arweights(q,_qz) for _qz in qz ];
  assert np.allclose(w1[0], w1[2]); # weights have to be periodic with 2pi/d
  #w2 = ML.arweights(q,qz);
  #assert np.allclose(w1,w2);       # qz-values as array

  # -- plotting of AR-EEL spectra
  print "plotting AR-EEL spectra for %d-layer graphene..."%N 
  qz_list = np.linspace(0,np.pi/d,10);
  cm=plt.cm.spectral;  colors = [ cm(x) for x in np.linspace(0,1,len(qz_list)) ];
  offset = 0;

  ## fixed in-plane component
  q = 0.1   * hm.BOHR;  # momentum transfer [a.u.]
  for iqz,qz in enumerate(qz_list):

  ## fixed total momentum transfer
  #Q = 0.5 * hm.BOHR; # momentum transfer [a.u.]
  #for theta in range(0,90,10):
  #  q = Q*np.cos(np.pi*theta/180);
  #  qz= Q*np.sin(np.pi*theta/180);

    P = ML.areel(q,qz)/N;         # N-layer system  
    Pinv = Graphite.areel(q,qz);  # infinite stack

    # test normal-mode decomposition
    Pref = np.sum(ML.areel_lmodes(q,qz),axis=0)/N;
    assert np.allclose(P,Pref);   # test normal-mode decomposition

    # plotting     
    scale = ( q**2 + qz**2 )**2;
    offset-= np.max(Pinv*scale)/1.5;
    plt.plot(E, P*scale + offset, c=colors[iqz], label="qz=%4.2f 1/A"%(qz/hm.BOHR));
    plt.plot(E, Pinv*scale+offset, c=colors[iqz], ls='--' );
    #plt.plot(E, Pref*scale + offset, 'k-.'); # check areel_lmodes

  plt.legend(loc=0,prop={'size':9});
  plt.suptitle('Layered-Electron-Gas model');
  plt.title('%d-layer graphene multilayers (solid) vs graphite (dashed)'%N);
  plt.xlabel('Energy [eV]');
  plt.ylabel('EELS');
  plt.show();

