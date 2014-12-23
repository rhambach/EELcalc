"""
  Dielectric response of graphene for arbitrary momentum transfers
  both along the GM and GK direction.

  USAGE
     executing this script will run the example code at the end of this file

  Copyright (c) 2014, rhambach. 
    This file is part of the EELcalc package and released
    under the MIT-Licence. See LICENCE file for details.
"""
import numpy as np;
from   copy  import deepcopy
import glob;
import os;

from EELcalc.tools.DPio import dp_mdf;
from EELcalc.external.interspectus import interspectus;


class GrapheneDP(object):
  """
    Dielectric properties of graphene from RPA ab-initio calculations
    for different q-vectors along GM and GK direction and energies up 
    to 40eV.
  """
  def __init__(self, qdir='GM', verbosity=1, filepattern=None):
    """
      qdir      ... (opt) 'GM' or 'GK' for symmetry direction of q
      verbosity ... (opt) 0 silent, 1 minimal output, 3 debug, >3 debug interpolation
      filepattern.. (opt) read eps2D from files matching filepattern
                          qdir has no effect in this case
    """
    self.qdir      = qdir;
    self.verbosity = verbosity;

    # read DP-output files containing EPS2D (sorted by momentum transfer)
    if filepattern is None:
      self.path   = os.path.dirname(os.path.abspath(__file__))+'/data/'+qdir;
      filepattern = self.path+'/CUTOFF_R12.6_grapheneAA-2d0-HIGH-RPA*-high-%s-q*_outlf.eps'%(self.qdir);
    self.spectra= dp_mdf.GetMDF(filepattern);
    self.spectra.sort(key=lambda mdf: np.linalg.norm(mdf.get_q('cc','au')));

    # check that we have eps2D
    assert len(self.spectra)>0
    for mdf in self.spectra:
      assert mdf.param['quantity']=='mdf';
      assert (''.join(mdf.param['comment'])).find('eps2D'); 

    # extract data
    self.eps2D  = np.asarray([ mdf.eps for mdf in self.spectra ]);
    q           = [ np.linalg.norm(mdf.get_q('cc','au')) for mdf in self.spectra ];
    self.q      = np.asarray(q, dtype=float);  # in 1/bohr
    self.E      = self.spectra[0].get_E();     # in eV
    self.calc_param = deepcopy(self.spectra[0].param);
    self.set_qprecision();

  def set_qprecision(self,qprec=1e-3):
    " determine maximal deviation from calculated q "
    self.qprec = qprec;

  def get_E(self):
    " Return energy axis [eV]"
    return self.E.copy();

  def get_q(self):
    " Return q-vectors of RPA calculations [a.u.]"
    return self.q.copy();

  def get_qmax(self):
    " Return largest q-vector for which the screening is available [a.u.]"
    return self.q[-1]; 

  def __find_q(self,q):
    " returns index for spectrum corresponding to q"

    # search closest q in list of input spectra (nearest interpolation)
    q  = np.atleast_1d(q);  qlist = q.flat;
    iq = [ np.argmin( np.abs(self.q - _q)) for _q in qlist ];

    # test if deviation in q is less than qprec
    if any( np.abs( self.q[iq] - qlist ) > self.qprec ):
      iqerr = np.where( np.abs( self.q[iq] - qlist ) > self.qprec )[0][0];
      raise ValueError(" could not find eps2D for q = %f a.u. \n"%qlist[iqerr] +
        "  closest q-vector found: q = %f a.u." % self.q[iq[iqerr]]);
 
    # return list of indices with same shape as q
    return iq[0] if len(iq)==1 else np.reshape(iq, q.shape);


  def get_eps2D(self,q,return_q=False):
    """
      Return eps2D = 1 - coul2D pol2D for given q [a.u],
      raises ValueError, if q is not found within precision
    """
    iq = self.__find_q(q);
    if return_q:  return ( self.q[iq], self.eps2D[iq] );
    else:         return self.eps2D[iq];                 # shape (Nq, NE)

  def get_pol2D(self,q,return_q=False):
    """
      Return 2D polarizability pol2D for given q [a.u.],
      raises ValueError, if q is not found within precision

      If return_q==True, also the q-values of the calculation
      are returned.
    """
    # calculate pol2D with exact q for each spectrum
    qcalc,eps2D= self.get_eps2D(q,return_q=True);
    coul2D = np.expand_dims( 2*np.pi / qcalc, -1 );       # shape (Nq, 1)
    pol2D  = (1 - eps2D) / coul2D;

    if return_q:  return ( qcalc, pol2D); 
    else:         return pol2D;
    
  def get_eels(self,q,return_q=False):
    """
      Return loss function for given q [a.u.], 
      raises ValueError, if q is not found within precision
    """
    # calculate eel-spctrum with exact q of each calculation
    qcalc,eps2D= self.get_eps2D(q,return_q=True);
    coul3D = np.expand_dims( 4*np.pi / qcalc**2, -1);     # shape (Nq, 1)
    coul2D = np.expand_dims( 2*np.pi / qcalc,    -1);
    chi2D  = (1./eps2D - 1) / coul2D;
    eels   = -coul3D*chi2D.imag

    if return_q:  return ( qcalc, eels); 
    else:         return eels;



class GrapheneScreening(GrapheneDP):
  """
    Energy-loss function for graphene from RPA ab-initio calculations 
    for arbitrary momentum transfers q along GM and GK and energies 
    up to 40eV. Starting from the polarisability calculated for a discrete
    set of q-vectors (see GrapheneDP), we use different interpolation methods:

     1. dipole approximation, valid for small q<0.05 1/A (0.1 1/A*)

         knowing the screening epsilon in the limit q->0 we calculate
         energy-loss spectrum for small q assuming that eps(q) ~ eps(0)

     2. interpolation of spectra satisfying a sum rule, for 0.05 (0.1*) < q < 1.4 1/A

         the loss spectrum for arbitrary q is obtained by linear 
         interpolation between the available spectra using the interspectus
         Python module (http://etsf.polytechnique.fr/Software/Interpolation)

     3. nearest calculated spectrum, for q>1.4 1/A

         at large momentum transfers, we return the calculated loss-function
         for the momentum transfer closest to q

     * valid for GK
  """
  def __init__(self, qdir='GM', verbosity=1):
    """
      qdir      ... (opt) 'GM' or 'GK' for symmetry direction of q
      verbosity ... (opt) 0 silent, 1 minimal output, 3 debug, >3 debug interpolation
    """
    self.qdir      = qdir;
    self.verbosity = verbosity;

    # set up LRA calculation starting from chi for q=0
    self.d     =  2*6.294;          # distance between sheets Lz [a.u.] 
    self.path  = os.path.dirname(os.path.abspath(__file__))+'/data/'+qdir;
    filename   = glob.glob(self.path + '/grapheneAA-2d0-HIGH-RPA*-high-q0.000*_outlf.mdf')[0];
    self.mdfq0 = dp_mdf.DPmdf(filename).get_mdf()[0];

    # get EPS2D and calculate EELS spectra for q>0
    self.__DP = GrapheneDP(qdir=qdir,verbosity=verbosity-1);
    q    = self.__DP.get_q();
    E    = self.__DP.get_E();
    eels = self.__DP.get_eels(q);
    self.calc_param = deepcopy(self.__DP.calc_param);
    self.eels_IN = interspectus.FamilyOfCurves(q,E,eels, \
                    qdesc='q', xdesc='E', fdesc='eels');

    # set up interpolation for small momentum transfers 0<q<0.77 1/bohr
    # NOTE: Here, we are using an unphysical normalisation function !
    #   The usual sum rule Int dE E * -Im(1/eps(q,E)) = const is not 
    #   fulfilled in the RPA calculations for Graphene. Using the standard
    #   wfunc = lambda q,X; x; as normalisation function, the 
    #   interpolation would introduce additional peaks at about 10eV 
    #   shifting spectral weight to higher energies
    #
    # NOTE: We also add an exponential tail to the end of the spectra 
    #   which avoids a truncation of the energy range during interpolation.

    #wfunc = lambda q,x: x;  # physical sum rule
    wfunc = lambda q,x: (x+x**3+1e-5) / (1+1/(0.1+q)**1);  
    tail  = lambda q,x: 1+np.exp((x-45));             
    Emax  = 50.5;                                    
    self.Interp_lowq = self.__get_interpolator(0,0.77,wfunc,tail,Emax);


  def __get_interpolator(self, qmin, qmax, wfunc, tail=None, Emax=None):
    """
      Set up the interpolator for given spectra with momentum transfer qmin<q<qmax [1/bohr].
      Additional parameter:
         
         wfunc ... function f(q,x) for the normalisation of the spectra
         tail  ... (opt) function t(q,x) appended to the tail of the spectra
         Emax  ... (opt) maximal energy for tail [eV]
    """

    # select spectra with qmin < q < qmax [1/bohr]
    index  = np.logical_and(qmin < self.eels_IN.q, self.eels_IN.q < qmax);
    q = self.eels_IN.q[index]; E = self.eels_IN.x[0]; eels = self.eels_IN.f[index];

    # append tail, which avoids restriction of energy range during interpolation
    if tail is not None:
      E_t    = np.arange(E[-1], Emax, E[1]-E[0], dtype=float);
      eels_t = [[ tail(q[iq], E_t[io]) for iq in range(len(q))] for io in range(len(E_t))];
      eels_t = np.asarray(eels_t) * eels[:,-1] / eels_t[0];
      E     = np.hstack((E, E_t[1:]));
      eels  = np.hstack((eels, eels_t[1:].T));

    return interspectus.Interpolator(q, E, eels, wfunc=wfunc, num_F=1000, k=1, verbosity=self.verbosity-3);


  def _eels_dipole_approximation(self,q):
    """
      Calculate EELS within dipole approximation (valid for 0<q<0.1 1/A) 
        q ... momentum transfer [a.u.], scalar or array
    """
    if self.verbosity>2: print "_eels_dipole_approximation, ", q
    if np.any(q>0.05) and self.verbosity>0:
      print " WARNING: eels_non_dispersive() is not accurate for q > 0.1 1/A "

    q = np.maximum(q,1e-6);       # tackle q=0 case (where chi->chi0)
    coul3D = np.expand_dims( 4*np.pi / q**2, -1); # shape (iq, 1)
    coul2D = np.expand_dims( 2*np.pi / q,    -1);
    # dipole approx: eps(q) ~ eps(0)
    chi0   = (1.-self.mdfq0.eps) / coul3D; 
    eps2D  = 1. - chi0 * coul2D * self.d;
    chi2D  = (1./eps2D - 1.)/coul2D;
    eels   = - coul3D * chi2D.imag;
    return eels;  # shape(nq, nE);


  def _eels_interpolated(self,q):
    """
      Calculate EELS for arbitrary momentum transfer q by interpolation
        q ... momentum transfer [a.u.]
    """
    if self.verbosity>2: print "_eels_interpolated, ", q
    try:
      lenE = len(self.eels_IN.x[0]);
      (E,eels) = self.Interp_lowq.get_spectrum(q);

    except:
      print " WARNING: eels_interpolated() only accepts q < %5.3f "\
                  % (self.Interp_lowq.input.q[-1]);
      raise

    return eels[:,:lenE];


  def _eels_nearest_spectrum(self,q):
    """
      Return input eels spectrum with momentum transfer closest to q
    """
    if self.verbosity>2: print "_eels_nearest_spectrum, ", q
    # search closest q in list of input spectra (nearest interpolation)
    q_IN = self.get_q();
    q    = np.atleast_1d(q);
    iq   = [ np.argmin( np.abs(q_IN - _q)) for _q in q ];
    eels = self.eels_IN.f[iq];
    return eels[0] if len(iq)==1 else eels;

  def get_eels(self,q):
    """
      Calculate EELS for arbitrary momentum transfer q 
      (wrapper for different interpolation methods depending
       on size of q)
        q ... momentum transfer [a.u.]
    """
    qlim_DA = self.get_q()[1];  # use dipole approximation up to first q>0 spectrum
    q = np.abs(q); # only positive q allowed

    if np.isscalar(q):
      if   q < qlim_DA:            
        return self._eels_dipole_approximation(q);
      elif q < self.Interp_lowq.input.q[-1]:            
        return self._eels_interpolated(q)[0];
      elif q < self.get_qmax(): 
        return self._eels_nearest_spectrum(q);
      else:                     
        raise ValueError("momentum transfer too large!" + \
          " q must be smaller than %5.3f 1/bohr"%(self.get_qmax()))

    else: # list of q vectors:
      q    = np.asarray(q);
      eels = np.empty( (len(q), len(self.eels_IN.x[0])) );    

      iDA  = q < qlim_DA;
      if any(iDA):
        eels[iDA] = self._eels_dipole_approximation(q[iDA]);

      iIP  = np.logical_xor(iDA, q<self.Interp_lowq.input.q[-1]);
      if any(iIP):  
        eels[iIP] = self._eels_interpolated(q[iIP]);

      iNN  = np.logical_and(~iDA,~iIP);
      if any(iNN):
        eels[iNN] = self._eels_nearest_spectrum(q[iNN]);

      if np.any(q>self.get_qmax()):
        raise ValueError("momentum transfer too large!" + \
          " q must be smaller than %5.3f 1/bohr"%(self.get_qmax()))

    return eels;

  def get_E(self):
    " Return energy axis [eV]"
    return self.eels_IN.x[0];
 
  def get_q(self):
     " Return q-vectors of RPA calculations [a.u.]"
     return self.eels_IN.q;
 
  def get_qmax(self):
    " Return largest q-vector for which the screening is available [a.u.]"
    return self.eels_IN.q[-1]; 


### Self-Testing
if __name__ == '__main__':

  import matplotlib.pylab as plt;
  import sys;
  __file__ = sys.argv[0];
 
  # test GrapheneDP
  print(" Tests for graphene_RPA.py are running ... ");
  DP = GrapheneDP(qdir='GM',verbosity=0);

  # - check multidimensional q-input
  q = [[0.0,0.106],[0.212,0.318],[0.212,0.318]];  
  pol2D = DP.get_pol2D(q);
  assert pol2D.shape == (3,2,len(DP.E));
  assert np.allclose( pol2D, [[ DP.get_pol2D(q[i][j]) for j in (0,1)] for i in (0,1,2)] )
  
  # - check polarisation (should be equivalent to DFT calculation)
  DP.set_qprecision(0.01);
  mdf  = DP.spectra[0]; 
  qcalc= np.linalg.norm(mdf.get_q('cc','au'));
  coul2D = 2*np.pi/qcalc;
  polref = (1-mdf.eps)/coul2D;
  q,pol  = DP.get_pol2D(qcalc+0.01,return_q=True);
  assert np.allclose( q, qcalc );
  assert np.allclose( pol, polref );

  # test GrapheneScreening
  Graphene = GrapheneScreening('GK',verbosity=0);

  # test scalar function calls
  Graphene.get_eels(0.77);
  Graphene._eels_interpolated(0.74); 
  Graphene._eels_dipole_approximation(0.01);
  Graphene._eels_nearest_spectrum(1.9);

  # test array call
  q = [1e-5, 0.01, 0.045, 0.055, 0.55, 0.77, 1.88];
  eels = Graphene.get_eels(q);
  for iq in range(len(q)):
    assert( np.allclose( eels[iq][1:], Graphene.get_eels(q[iq])[1:] ) );

  print(" Running example for graphene_RPA.py ...");
  ax=plt.figure().add_subplot(111);
  plt.suptitle("Graphene screening for small q, %s direction" % (Graphene.qdir))

  # Graphene input data  
  IN = Graphene.Interp_lowq.input;
  for iq, q in enumerate(IN.q):
    offset = q*400;
    scale  = q**0.5;
    ax.text(41, offset, "%4.2f"% (q/0.529177));
    ax.plot(IN.x[iq],IN.f[iq]*scale+offset,"r-",linewidth=2,label="input" if iq==0 else "");
  ax.text(40, offset*1.1, "q [1/A]");

  # Interpolate for dense grid of Q vectors
  Q  = IN.q.tolist();
  Q.extend(((np.sort(Q)[:-1] + np.sort(Q)[1:]) / 2.).tolist())  
  Q.extend(((np.sort(Q)[:-1] + np.sort(Q)[1:]) / 2.).tolist())  
 
  E   = Graphene.get_E();
  eel = Graphene._eels_interpolated(Q); 
  for iq,q in enumerate(Q):
    offset = q*400;
    scale  = q**0.5;
    ax.plot(E,eel[iq]*scale+offset,"k-",linewidth=1,label="interpolated" if q<0.001 else "");

  # Dipole Approximation for q<0.1 1/bohr
  Q = np.asarray(Q);
  eel = Graphene._eels_dipole_approximation(Q[Q<0.1]);
  for iq, q in enumerate(Q[Q<0.1]):
    offset = q*400;
    scale  = q**0.5;
    ax.plot(E,eel[iq] * scale + offset, "g-", label="dipole approx" if iq==0 else "");

  ax.set_xlim(E[0],E[-1]);
  ax.set_ylim(0,200);
  plt.legend(loc=0);

  # nearest interpolation for large q
  ax=plt.figure().add_subplot(111);
  plt.suptitle("Graphene screening for large q, %s direction" % (Graphene.qdir));
  Q   = Graphene.get_q().tolist();
  for iq,q in enumerate(Q):
    offset = q*50;
    scale  = q*2;
    if (iq % 5 == 0): ax.text(41, offset, "%3.1f"% (q/0.529177));
    ax.plot(E,Graphene.eels_IN.f[iq]*scale+offset,"r-",linewidth=2,label="input" if iq==0 else "");
  ax.text(40, offset*1.05, "q [1/A]");

  Q.extend(((np.sort(Q)[:-1] + np.sort(Q)[1:]) / 2.).tolist());
  eel = Graphene._eels_nearest_spectrum(Q);
  for iq,q in enumerate(Q):
    offset = q*50;
    scale  = q*2;
    ax.plot(E, eel[iq] * scale + offset,"k-",label="nearest interpolation" if iq==0 else "");

  plt.legend(loc=0);
  plt.show();
