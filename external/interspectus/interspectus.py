"""                                                             
   INTERpolation of SPECTra Using Sum rules

   USAGE
 
     Examples are included at the end of this file and can be run by
     $ python interspectus.py

   DESCRIPTION
   
     A detailed description of the procedure can be found in
     [Weissker et.al., Phys. Rev. B 79, 094102 (2009)].
     
     The present implementation is divided in two steps. First,
     during creation of an instance of the Interpolator class, the
     function prepare_interpolation() will already

       A normalise the input-curves f(x) for each q and x           (i)
       B compute the integrals F(x) for each q                      (ii)
       C invert this function to get x(q) for each F (at num_F points)
       D store x(q,F) in the private member variable F_inverted

     Second, each time the user calls get_spectrum(), we

       F interpolate x(q,F) to get x(q_new) at the new parameters
         q_new for each fixed F (we use an adaptive grid for F)     (iii)
       G re-invert interpolated curves to get F(x) for each q_new
       H differentiate to get f(x) = dF/dx                          (iv)
       I re-normalise output-curves f(x) for each q_new and x

   IMPLEMENTATION
   
     We need to interpolate the curves (linear or using cubic splines):
       a) along q (step F), which is the goal of the calculation
       b) along x (step C+G) in order to bring the curves to common
          x_new values after inversion (exchanging f and x)
     By interchanging x and q in the input data, a single routine for
     interpolation along x is sufficient (see FamilyOfCurves.interpolate_x).

     Unfortunately, the interpolation steps may fail! One usually has
     to change the sum-rule or the x-ranges of input or output data to
     circumvent this problem. To help with the debugging, a figure
     with the corresponding curves is raised in this case.

   FAQ

     1. Should I use linear or non-linear interpolation ?

       To our experience, non-linear interpolation in q may introduce  
       artefacts, e.g., the reappearance of peaks from the neighbouring 
       spectra. Therefore linear interpolation is used as default.

     2. Why does the interpolation not exactly reproduce the input data ?

       In step C, x(q,F) is sampled at a finite set of F points, which
       are interpolated from the input data F(x). By default we use a 
       reduced sampling set of 100 points and thus introduce small errors. 
       Increase num_F or use num_F=None to avoid this restriction (might be
       much slower though).

     3. Why do you use a simple summation or difference instead of more
       sophisticated methods to calculate the integral and derivative?

       We want to be sure, that the input data is exactly reproduced
       by the interpolation method. This requires that d/dx Int(f(x))=f(x)
       is fulfilled by our numerical method which is the case for a 
       simple summation and difference. Using the trapezoidal rule and 
       a higher-order finite difference method, instead, will smoothen 
       the spectra f(x).

     Numerical methods:
     
       interpolation:   linear            (alt: cubic splines)
       integration:     simple summation  (alt: Simpson's rule/Trapezoid rule)
       differentiation: simple difference (alt: 5-point finite differences)
       
   COPYRIGHT

     Copyright (c) 2011, Ralf Hambach, Hans-Christian Weissker. 
     All rights reserved. Use of this source code is governed by 
     a BSD-style license that can be found in the LICENSE file.
"""
import matplotlib.pylab as plt;
import numpy as np;
import scipy.integrate
import scipy.interpolate


class FamilyOfCurves:
  """
    Encapsulates a set of curves f(x)=f(q,x)
    that depend on the parameter q. It implements
    easy plotting and interpolation along x.
  """ 

  def __init__(self,q,x,f,qdesc="q",xdesc="x",fdesc="f"):
    """
    q    ... 1D-array of shape (q.size)
    x    ... 2D-array of shape (q.size,x.size)
    f    ... 2D-array of shape (q.size,x.size)
    *desc... description string for q,x,f
    """
    
    self.f = np.asarray(f,dtype='float');
    self.x = np.asarray(x,dtype='float');
    self.q = np.asarray(q,dtype='float');
    self.fdesc = fdesc;
    self.xdesc = xdesc;
    self.qdesc = qdesc;

    # complete input data (if constant for different q)
    if (len(self.x.shape)==1):
      self.x=np.tile(self.x,(len(self.q),1));
    if (len(self.f.shape)==1):
      self.f=np.tile(self.f,(len(self.q),1));

    # test shape of input data
    if (self.q.shape[0] <> self.x.shape[0]) or \
       (self.x.shape    <> self.f.shape):
      raise ValueError("Invalid shape of arguments.");

    # test for double parameters
    if (np.unique(self.q).size < self.q.size):
      raise ValueError("Parameters are not unique: \n " + np.array_repr(np.sort(self.q)));

    
  def sort(self):
    """
    sort the family of curves f(q,x) along q
    """ 
    index_array=self.q.argsort();
    self.q = self.q[index_array];
    self.x = self.x[index_array];
    self.f = self.f[index_array];

  def plot(self,ax=None,title=None,legend=True):
    """
    plot the family of curves
    ax    ... (opt) axes object, default: current axes
    title ... (opt) string for the figure title
    legend... (opt) True if the legend should be shown
    """
    if ax is None: ax = plt.gca();
    for iq in range(self.q.size):
      colors=['b','g','r','c','m','y','k','w'];
      ax.plot(self.x[iq],self.f[iq],\
              colors[iq%6],label="%s=%5.3f"%(self.qdesc,self.q[iq]));
    ax.set_xlabel(self.xdesc); ax.set_ylabel(self.fdesc);
    if title is not None: ax.set_title(title);
    if legend: ax.legend();

  def interpolate_x(self,xnew,k=3):
    """
    Calculation of f at xnew for each parameter q via interpolation
    using the scipy.interpolate module. Additionally, we try to help
    with the error analysis in case of an exception.

    xnew ... 1D-array of new x values (common for each q)
    k    ... (opt) order of spline, default 3, 
             even order splines should be avoided, 1 <= k <= 5

    RETURNS f(q,xnew) (2D-array)
    """
    fnew=np.zeros((self.q.size,xnew.size));
    for iq in range(self.q.size):

      # STEP 1: cubic spline representation
      try:
        tckp    = scipy.interpolate.splrep(self.x[iq],self.f[iq],k=k);
      except (ValueError, TypeError):
        print "\n ERROR: in cubic spline representation!"
        print "   during interpolation of %s(%s) for parameter %s[%d]=%f " \
              %(self.fdesc,self.xdesc,self.qdesc,iq,self.q[iq]);

        # E: input x-array not ascending ?
        if np.any(self.x[1:]-self.x[:-1]<0):
          print("   reason:   input x-values are not ascending");
          print("   solution: you may change the range of %s\n"%(self.xdesc));

        # E: not enough input points ?
        if self.x.shape[1]<=k:
          print("   reason:   not enough %s values "% (self.xdesc) \
                +"for cubic spline interpolation (should be >3)" );
          print("   solution: reduce order of the spline fit (optional flag 'k')\n");

        # plot curves for info
        self.plot(title="interpolation error!",ax=plt.figure().add_subplot(111));
        plt.show();
        raise

      # STEP 2: get new f-values
      try:
        fnew[iq,:]= scipy.interpolate.splev(xnew,tckp);
      except ValueError:
        print "\n ERROR: in cubic spline evaluation!"
        print "   during interpolation of %s(%s) for parameter %s[%d]=%f " \
              %(self.fdesc,self.xdesc,self.qdesc,iq,self.q[iq]);

        # E: xnew outside x-range ?
        if np.any((xnew.min < self.x) | (self.x<xnew.max)):
          print("   reason: input x-values are not in order");
          
        # plot curves for info
        self.plot(title="interpolation error!",ax=plt.figure().add_subplot(111));
        plt.show();
        raise

    return fnew;


        
class Interpolator:
  """
    Interpolation of a family of curves f(x)=f(q,x) along the
    parameter q.  The curves should obey a sum-rule of the type

      1 = Int f(q,x) * w(q,x) dx

    A detailed description can be found in
    [Weissker et.al., Phys. Rev. B 79, 094102 (2009)].
  """

  def __init__(self,q,x,f,wfunc=None,verbosity=1,debug=False,k=1,num_F=100):
    """
    q    ... 1D-array of shape (q.size)
    x    ... 2D-array of shape (q.size,x.size)
    f    ... 2D-array of shape (q.size,x.size) 
    wfunc... (opt) function reference to w(q,x)
    verbosity...(opt) defines verbosity 0: silent, 1: output of Warnings, 3: debug
    debug... (opt) boolean for debugging info's (same as verbosity=3)
    k    ... (opt) order of spline interpolation along x and q
             even order splines should be avoided, 1 <= k <= 5
    num_F... (opt) reduced sampling size for x(F) in step C
             if num_F=None, all F points in input data are used
    """
  
    self.__debug=debug;
    self.verbosity = 3 if debug else verbosity;

    # input curves
    self.input = FamilyOfCurves(q,x,f);
    self.input.sort();

    # check if x are equidistant and same for all q
    x = self.input.x;
    if not np.allclose( 0, x - x[0] ):
      raise ValueError("ERROR in Iterpolator.__init__(): \n"+
        "  x values must be identical for all parameters q ");
    dx = x[0,1:]-x[0,:-1];
    if not np.allclose( 1, dx/dx[0] ):
      raise ValueError("ERROR in Iterpolator.__init__(): \n"+
        "  x values must be equidistant. Found differences \n"+str(np.unique(dx)));


    # spline order
    self.k = k;
    if self.k >= len(q): raise ValueError("ERROR in Iterpolator.__init__(): \n"+\
           "  order k=%d spline along q needs at least %d q-values.\n" %(k,k+1)+\
           "  Give more input spectra or decrease k (optional argument).");
    # sampling precision
    self.num_F=num_F;
  
    # sum rule for normalisation
    if wfunc is None:
      # default: w(q) = 1/( Int f(q,x) dx )
      # NOTE: physically this is not justified. Depending on the quantity,
      #       it might be also better to use 1/( Int f(q,x)*x dx ) !
      if self.verbosity>0:
        print "\n WARNING: using summation rule w(q) = 1/( Int f(q,x) dx ) " +\
              "\n          which is probably unphysical. Use wfunc option instead."
      Fmax=scipy.integrate.trapz(self.input.f,self.input.x);
      self.__Fmax_tckp = scipy.interpolate.splrep(self.input.q,Fmax,k=self.k);
      self.wfunc = lambda q,x: 1./scipy.interpolate.splev(q,self.__Fmax_tckp);
    else:
      self.wfunc=wfunc
    
    self.__prepare_interpolation();

  def __prepare_interpolation(self):
    "prepares the interpolation (private function)"

    q=self.input.q;
    x=self.input.x;
    f=self.input.f.copy();
    

    # -- (A) normalisation: f -> f*w -------------------------------------------
    
    for iq in range(f.shape[0]):
      for ix in range(f.shape[1]):
        f[iq,ix] = f[iq,ix]*self.wfunc(q[iq],x[iq,ix]);

    # -- (B) integration: F = Int f dx ----------------------------------------
    def integrate(f,x):
      F=np.zeros(x.size);
      for i in range(x.size):
        # Simpson rule seems to be problematic for f(x)~0
        #F[i]=scipy.integrate.simps(f[0:i+1],x[0:i+1]);
        # Trapezoidal rule
        #F[i]=scipy.integrate.trapz(f[0:i+1],x=x[0:i+1]);

        # we use simple summation (see FAQ 3)
        F[i]=np.sum(f[0:i+1])*(x[1]-x[0]);
      return F;

    F=np.zeros(f.shape);
    for iq in range(q.size):
      F[iq]=integrate(f[iq],x[iq]);

    # DEBUG: plot f(q,x) and F(q,x)
    if self.__debug:
      fig=plt.figure(); ax1 = fig.add_subplot(211); ax2 = fig.add_subplot(212);
      FamilyOfCurves(q,x,f).plot(ax=ax1, title="Debug: (B) Integration");
      FamilyOfCurves(q,x,F,fdesc="F").plot(ax=ax2,legend=False);
      ax1.legend(loc="upper left", bbox_to_anchor=(1,1))
      fig.subplots_adjust(right=0.75);

    # -- (C) inversion F(q,x) -> xi(q,Fi) -------------------------------------
    Fi=F.flatten(); Fi.sort();                  # adaptive grid for new F vals
    if self.num_F is not None:                  # reduce number of F values 
      index=np.linspace(0,len(Fi)-1,self.num_F);
      Fi=Fi[map(int,np.round(index))];
    Fi=np.unique(Fi); 
    
    Fmin=F[:,-1].min(); Fmax=F[:,-1].max();     # last value of F should be 1
    Fi=Fi[np.where(Fi<Fmin)];                   # make F(q,x) surjective
    if (Fmax-Fmin)/(Fmax+Fmin) > 0.001:         # check normalisation
      if self.verbosity>0:
        print "\n WARNING: normalisation of integrals deviates by "      +\
              "%4.1f%% at x=%4.1f"%(100*(Fmax-Fmin)/(Fmax+Fmin),x[0,-1]) +\
              "\n         check the sum-rule and eventually increase the x-range."
    xi=FamilyOfCurves(q,F,x,
                      qdesc="q",xdesc="F",fdesc="x").interpolate_x(Fi,k=self.k);
                                                # interp. curves x(q,F) along
                                                # abscissa at Fi => xi(q,Fi)


    # -- (D) save family of curves xi(Fi,q) -----------------------------------
    self.__xi_of_Fi_q = FamilyOfCurves(Fi,q,xi.swapaxes(0,1),  
                                       qdesc="Fi",xdesc="q",fdesc="xi");
    # NOTE: in order to interpolate along q later on, we take q for
    #       the abscissa (interpolation direction) and Fi as parameter




  def get_spectrum(self,q_new,x_new=None):
    """
    Calculates the interpolated spectrum f(q_new,x_new) for new
    parameters q_new (and eventually new arguments x_new)

    q_new ... 1D-array of parameters
    x_new ... (opt) 1D-array of arguments,
              Default: same arguments as the first input spectrum

    RETURNS a the tuple (x_new,f_new)
      x_new ... 1D-array of x-range that may have been truncated
      f_new ... 2D-array of shape (q_new.size,x_new.size) that
                contains the values f(q_new,x_new)
    """


    # arguments
    q_new = np.asarray([q_new],dtype=float).flatten();
    if x_new is None:  x_new = self.input.x[0]; # default: x-range of
    x_new = np.asarray(x_new,dtype=float);      #  first input spectrum

    # TODO, check if q/x_new is in q/x
    for q in q_new:
      if q < self.input.q[0] or q > self.input.q[-1]:
        raise(ValueError("q=%5.3f is outside of the input parameter range!"%q));

    # -- (F) q-interpolation of  xi(Fi,q) -> xi_new(Fi,q_new) -----------------
    xi_new = self.__xi_of_Fi_q.interpolate_x(q_new,k=self.k);

    # DEBUG: plot xi(Fi,q) and xi_new(Fi,q_new)
    if self.__debug:
      ax = plt.figure().add_subplot(111);
      colors=['b','g','r','c','m','y','k','w'];
      self.__xi_of_Fi_q.plot(ax=ax,legend=False,
         title="Debug: (F) Interpolation of xi(Fi,q)"); #  xi(Fi,q)
      for i in range(xi_new.shape[0]):
        ax.plot(q_new,xi_new[i],colors[i%6]+'.');       # xi_new(Fi,q_new)

    # -- (G) inversion: xi_new(Fi,q_new) -> F_new(q_new,x_new) ----------------
    x_max = xi_new[-1].min();  
    if (x_new.max() > x_max):                   # make xi_new(Fi,q_new) surj.
      x_new = x_new[np.where(x_new<x_max)];
      if self.verbosity>0:
        print(" WARNING: x-range truncated during inversion to " +\
              "[%5.3f,%5.3f]"%(xi_new[0,0],x_max));
    F_new=FamilyOfCurves(q_new, xi_new.swapaxes(0,1), self.__xi_of_Fi_q.q, 
         qdesc="q_new",xdesc="xi_new",fdesc="Fi").interpolate_x(x_new,k=self.k);
                                    # interpolate curves Fi(q_new,xi_new) along
                                    # abscissa at x_new => F_new(q_new,x_new)

    # -- (H) differentiation: f_new = dF_new/dx_new ---------------------------

    # http://en.wikipedia.org/wiki/Finite_difference_coefficients
    def difference(F,x):
      "difference for debugging"
      f=np.empty(len(x));
      f[1:]=(F[1:]-F[:-1])/(x[1]-x[0]); # vectorized for speed up
      f[0]=F[0];
      return f;

    def differentiate3(F,x):
      "three point method (error in O(h**2))"
      f=np.empty(len(x));
      f[1:-1]=(F[2:]-F[0:-2])/(x[1]-x[0])/2; # vectorized for speed up
      f[0]=f[-1]=0.
      return f;

    def differentiate5(F,x):
      "five point method (error in O(h**4))"
      f=np.empty(len(x));
      f[2:-2]=(-F[4:]+8*F[3:-1]-8*F[1:-3]+F[0:-4])/(12*(x[1]-x[0]));
      for i in (1,-2):
        f[i]=(F[i+1]-F[i-1])/(x[i+1]-x[i-1]);
        # differentiation is only possible _inside_ the x-interval
        # the boundaries can be calculated with the 3 point-rule
        # or set to zero
      f[0]=f[-1]=0.
      return f;
    
    f_new=np.zeros(F_new.shape);
    for iqn in range(q_new.size):
      # we use a simple difference (see FAQ 3)
      f_new[iqn]=difference(F_new[iqn],x_new);
    

    # -- (I) re-normalisation: f_new -> f_new / w_new -------------------------
    for iqn in range(f_new.shape[0]):
      for ixn in range(f_new.shape[1]):
        f_new[iqn,ixn] = f_new[iqn,ixn]/self.wfunc(q_new[iqn],x_new[ixn]);

    return (x_new,f_new);





### Self-Testing
if __name__ == '__main__':
  print(" Example for Interspectus.py is running ... ");

  # input stuff
  def readfile(filename):
    file = open(filename,'rb');
    x=[]; f=[];
    while True:
      line = file.readline();
      if not line :    break       # EOF reached
      if line[0]=='#': continue    # ignore comment
      col = map(float,line.split());
      x.append(col[0]); f.append(col[1]);
    file.close();
    return(f,x);

  # read input-spectra (see also dp_io-module)
  infiles = ["./tests/EELS/tdlda_q=.25_.25_.25_outlf.eel_gauss_abs_.42",
             "./tests/EELS/tdlda_q=.5_.5_.5_outlf.eel_gauss_abs_.42",
             "./tests/EELS/tdlda_q=.75_.75_.75_outlf.eel_gauss_abs_.42"];
  x_inp=[]; f_inp=[]; q_inp = [0.25,0.5,0.75];
  for file in infiles:
    (f,x) = readfile(file);
    x_inp.append(x[0:600]);  f_inp.append(f[0:600]); # trim to common length


  # read spectra for comparison
  dpfiles = ["./tests/EELS/tdlda_q=.375_.375_.375_outlf.eel_gauss_abs_.42",
             "./tests/EELS/tdlda_q=.625_.625_.625_outlf.eel_gauss_abs_.42"];
  x_comp=[]; f_comp=[]; q_comp = [0.375,0.625];
  for file in dpfiles:
    (f,x) = readfile(file);
    x_comp.append(x);  f_comp.append(f);      




  # initialize Interpolator
  wfunc = lambda q,x: x; # not useable
  I = Interpolator(q_inp,x_inp,f_inp,wfunc=wfunc,debug=True,k=2);

  # get interpolated spectra  
  (x_out,f_out) = I.get_spectrum(q_comp);

  # plot
  ax=plt.figure().add_subplot(111);
  plt.suptitle("Interpolation for momentum dependent EELS, bulk Silicon");
  plt.title("[Weissker, et.al.: PRB(79) 094102 (2009), fig. 10]");

  for iq in range(len(q_inp)):
    ax.plot(x_inp[iq],f_inp[iq],  "g-",label="input, q=%5.3f"%(q_inp[iq]));
  for iq in range(len(q_comp)):
    ax.plot(x_comp[iq],f_comp[iq],"r-",label="comp,  q=%5.3f"%(q_comp[iq]));
    ax.plot(x_out, f_out[iq],     "b.",label="interp,q=%5.3f"%(q_comp[iq]));

  ax.legend();
  ax.set_xlim([0,40]);
  ax.set_ylim([0,4.5]);
  plt.show();

