"""
   Input/Output module for files generated by DP

   USAGE

     # read single output file (header + data)
     data = DPmdf("tests/1pol.mdf");

     # plot data (skip first row of the test input file)
     plot(data.E[1:], data.Ieps[0][1:])

     # use interface class MDF ...
     mdf = data.get_mdf()[0];

     # ... for interpolation
     mdf.set_E(np.arrange(10,20,0.1));
     plot(mdf.get_E(),mdf.get_Ieps());

     # ... or easy plotting
     plot(mdf.get_E(),mdf.get_eel());

   IMPLEMENTATION

     This module contains two classes:

       DPmdf represents the input file as a whole and may contain data
         for several polarization directions.
       
       MDF is an interface to DPmdf representing only a single spectrum
         and implements basic data-manipulation.

   TODO

     Not all header parameters implemented, e.g.
     testparticle, different kernels, so, alpha

  Copyright (c) 2014, rhambach. 
    This file is part of the EELcalc package and released
    under the MIT-Licence. See LICENCE file for details.
"""
import re;
import sys;
import string;
import copy;
import glob;
import numpy as np;


def GetMDF(file):
  """
  Reads all *.eps and *.mdf files that match the file name pattern <file>
  RETURN list of spectra (MDF-objects)
  """
  spectra=[];
  for file in glob.glob(file):
    #print "read file "+file;
    spectra.extend(DPmdf(file).get_mdf());
  return spectra;

     

class MDF:
  """
    (Macroscopic Dielectric Function)
    This class is an interface to DPmdf for *.eps and *.mdf files
    representing a single spectrum (for a given polarization/q vector).
  """
  def __init__(self,param=None,uparam=None,E=None,eps=None):
    """
    param  ... dictionary that contains the keys
                quantity, lfe, kernel, emin, emax, dsf_fact
                nkbz, nkibz, npwmat, npwwfn, nbands, lomo, broad,
                G_rc, qred_rc, G_cc, qred_cc,
                (opt) ucvol, comment
    pupdate... (opt) dict to overwrite param keys
    out    ... (opt) file descriptor for output
    """
    from copy import deepcopy;

    # parameter array
    self.param = deepcopy(param);
    if uparam is not None:
      self.param.update( deepcopy(uparam) );      

    # copy data
    self.E     = np.array(E,float);
    self.eps   = np.array(eps,complex);

        
  # interface
  def get_E(self):
    return self.E;
  def get_Reps(self):
    return self.eps.real;
  def get_Ieps(self):
    return self.eps.imag;
  def get_eel(self):
    return (-1./self.eps).imag;
  def get_dsf(self):
    fact = self.param['dsf_fact'];
    if (fact==0): fact=1;
    return fact*self.get_eel() / 27.2116;

  def get_q(self,coord='rc',unit='au'):
    """
     returns the q-vector
     coord ... (optional) specifiction of the basis
               'rc' for reciprocal coordinates (default)
               'cc' for cartesian coordinates
     units ... (optional) specifiction of the units
               'au' for atomic units           (default)
               'si' for SI units (1/Angstrom)
    """
    if(coord=='rc'):
      return self.param['q_rc'];
    if(coord=='cc' and unit=='au'):
      return self.param['q_cc'];
    if(coord=='cc' and unit=='si'):
      return self.param['q_cc']/0.529177249;
    
  # manipulation
  def set_Emin(self,Emin):
    "restrict energy range to E>=Emin"
    index=np.where(self.E>=Emin);
    self.E   = self.E[index];
    self.eps = self.eps[index];
    self.param['comment'].append("#  -restricted energy range to Emin=%f\n"%Emin);

  def set_Emax(self,Emax):
    "restrict energy range to E<=Emax"
    index=np.where(self.E<=Emax);
    self.E   = self.E[index];
    self.eps = self.eps[index];    
    self.param['comment'].append("#  -restricted energy range to Emax=%f\n"%Emax);
    
  def set_E(self,E):
    """
    Change energy sampling E by interpolation 
    E ... 1D-array of floats
    """
    from scipy.interpolate import splrep,splev;
    # Reps
    tckp      = splrep(self.E,self.eps.real,k=3);
    eps       = np.array(splev(E,tckp),dtype='complex');
    # Ieps
    tckp      = splrep(self.E,self.eps.imag,k=3);
    eps      += 1j*np.array(splev(E,tckp),dtype='complex');
    self.E    = E;
    self.eps  = eps;
    self.param['comment'].append("#  -performed interpolation along energy axis\n");

  # output
  def write(self, out=sys.stdout, quantity="mdf"):
    """
    Write mdf to outfile [same format as out(n)lf.mdf].
      out     ...  (opt) file descriptor or sys.stdout (default)
      quantity...  (opt) "mdf" (default), "eel", or "dsf" 
    """
    # header
    p = { 'nqpol'  : 1          ,
         'quantity': quantity      };
    mdf_header(self.param, p, out);

    # data
    if   quantity=='mdf':   y = zip(self.eps.real,self.eps.imag);
    elif quantity=='eel':   y = self.get_eel().reshape((len(self.E),1));
    elif quantity=='dsf':   y = self.get_dsf().reshape((len(self.E),1));
    else: raise ValueError("ERROR! unexpected value '%s'"%quantity \
                           + " for parameter 'quantity'\n");
    for (io,E) in enumerate(self.E):
      out.write("%8.3g   "%E);
      for yi in y[io]: out.write("\t%12.8g"%(yi));
      out.write("\n");


class DPmdf:
  """
  DPmdf: reads the DP-output file (*.eps, *.mdf, *.eel, *.dsf), i.e.
    -> all calculation parameters given in the header
    -> the raw-data for all polarization directions (except averages)

  working for dp-versions: >4.5.6 
  """
  def __init__(self,filename):
    "filename ... full name of the inputfile"
    
    try:
      self.__scan_header(filename);
      self.__scan_data(filename);     
    except:
      print "\nERROR! while reading file '%s'.'\n\n"%(filename)
      raise
 

  def __scan_header(self,filename):
    # scan entire header
    file=open(filename);
    header=[]; p=dict();
    while True:
      line=file.readline();
      if not line or line[0]<>"#": break;
      header.append(line);
    file.close();

    # filename
    p['filename']=filename;

    line=0;
    # type of calculation
    if header[line].find("# macroscopic dielectric function")==0: p['quantity']="mdf"; 
    if header[line].find("# electron energy loss"           )==0: p['quantity']="eel";
    if header[line].find("# dynamic structure factor"       )==0: p['quantity']="dsf";
    # additional format for python results (only 1 polarisation direction)
    if header[line].find("# multi-column"                   )==0: 
      p['quantity']="multcol";
      p['desc'] = re.findall("# multi-column\s*\((.*)\)", header[line])[0];

    line+=1;
    if header[line].find("# without local fields"           )==0: p['lfe']="nlf"; 
    if header[line].find("# with local fields"              )==0: p['lfe']="lf"; 

    # "# testparticle  rpa" old version
    line+=1;
    if header[line].find(" rpa"                             ) >0: p['kernel']="rpa";
    if header[line].find(" lda"                             ) >0: p['kernel']="lda";
    if header[line].find(" jdos") >0 and p['lfe']=='nlf'        : p['kernel']="jdos";
    if header[line].find(" so+alpha"                        ) >0:
      p['kernel']="so+alpha";
      line+=2; # skip following two lines: "# so = 0.65 [eV]"
               # "# alpha = -0.20 "
    # "# testparticle  rpa" new version
    if header[line].find(" RPA"                             ) >0: p['kernel']="rpa";
    if header[line].find(" Adiabatic lda (TDLDA"            ) >0: p['kernel']="lda";
    if header[line].find(" Long-range kernel al") >0:
      p['kernel']="so+alpha";
      line+=1; # skip following line:      "# so = 0.65 [eV]"
               
    # "# npwmat = 135  npwwfn = 983  nbands =  96  lomo =   1  nkibz = 110  nkbz =****"
    line+=1;
    ( p['npwmat'],p['npwwfn'],p['nbands'],p['lomo'],p['nkibz'],p['nkbz'])=   \
       map(int,   map(lambda s: s.replace('*','0'), \
           re.findall("^\# npwmat\s*=\s*([\*\d]+)\s*npwwfn\s*=\s*([\*\d]+)\s*"+ \
                          "nbands\s*=\s*([\*\d]+)\s*lomo\s*=\s*([\*\d]+)\s*"+   \
                          "nkibz\s*=\s*([\*\d]+)\s*nkbz\s*=\s*([\*\d]+)",header[line])[0]));
  
    # "# energy range   0.00 50.00 [eV]"
    line+=1;
    (p['emin'],p['emax'])= map(float,re.findall("^\# energy range\s*"+\
                                     "(-?[\d\.]+)\s*(-?[\d\.]+)\s*\[eV\]",header[line])[0]);

    # "# lorentzian broadening = 0.1000 [eV]"
    line+=1;
    p['broad']=float(re.findall("^\# lorentzian broadening =\s*"+\
                                                 "([\d\.]+)\s*\[eV\]",header[line])[0]);
    # "# dsf factor      .106410 [a.u.]"
    line+=1;
    p['dsf_fact']=float(re.findall("^\# dsf factor\s*([\d\.]+)\s*\[a\.u\.\]",header[line])[0]);

    # opt: "# unit cell volume   1870.367432 [a.u.] "
    if header[line+1].find(" unit cell volume") >0:
      line+=1;
      p['ucvol']=float(re.findall("^\# unit cell volume\s*([\d\.]+)\s*\[a\.u\.\]",header[line])[0]);
    
    # Get G: "# G = (  0,  0,  0) [r.l.u.]"
    #        "# G = (  0.000000,  0.000000,  0.000000) [c.c.]"
    #        "# G = (  .000000,  .000000,  .520275) [c.c. units of 2pi/a =   .959379 a.u.]"
    def rd_vec(name,unit,line):
      vec=re.findall("^\# "+name+"\s*=\s*"+\
       "\(\s*(-*[\d\.]+),\s*(-*[\d\.]+),\s*(-*[\d\.]+)\s*\)\s*"+unit,line)[0];
      return np.array(map(float,vec));

    line+=1; self.G_rc=rd_vec("G","\[r\.l\.u\.\]",header[line]);
    line+=1; self.G_cc=rd_vec("G","\[c\.c\.\]"   ,header[line]);
    line+=1; # skip 

    # get number of averages (search for lines with "# q -> 0 average" in the following header)
    self.nqavg = len(re.findall("^\# q -> 0 average",string.join(header[line+1:],''),re.M));

    # get number of polarizations (search for lines with "# q = (" in the following header)
    self.nqpol = len(re.findall("^\#\s*q\s*=\s*\(.*\[c\.c\.\]",string.join(header[line+1:],''),re.M));

    # go to (line before) first line that contains pattern "# q = ("
    while not re.match("^\#\s*q\s*=\s*\(",header[line+1]): line+=1

    # Get q: "# q = ( 0.200000, 0.000000,-0.113636) [r.l.u.]"
    #        "# q = ( 0.068728,-0.039680,-0.154474) [c.c.]  "
    #        "# q = (  .176777,  .102062,  .000000) [c.c. units of 2pi/a =   .959379 a.u.]"
    self.q_rc=[];   self.q_cc=[];          # init
    self.qred_rc=[];self.qred_cc=[];
 
    for iqpol in range(self.nqpol):
      #print header[ilineq+3*iqpol];
      line+=1; self.qred_rc.append(rd_vec("q","\[r\.l\.u\.\]",header[line]));
      line+=1; self.qred_cc.append(rd_vec("q","\[c\.c\.\]"   ,header[line]));
      line+=1; # skip one line
      self.q_rc.append( self.G_rc+self.qred_rc[iqpol] );
      self.q_cc.append( self.G_cc+self.qred_cc[iqpol] );

    # convert to array
    self.q_rc    = np.asarray(self.q_rc,float);
    self.q_cc    = np.asarray(self.q_cc,float);
    self.qred_rc = np.asarray(self.qred_rc,float);
    self.qred_cc = np.asarray(self.qred_cc,float);
    self.G_rc    = np.asarray(self.G_rc,float);    # might be non-int in some cases! see interpolate_chi0.py
    self.G_cc    = np.asarray(self.G_cc,float);

    # comment lines at the end of the header
    line+=1;
    if line < len(header):
      p['comment'] = header[line:];

    self.param  = p;       # all DP-parameter except polarizations
    
  def __scan_data(self,filename):
    # skip header
    file=open(filename);
    while True:
      line=file.readline();
      if not line or line[0]<>"#": break;

    # IF  Im(eps^-1) given 
    if self.param['quantity'] in ("dsf","eel"):
      self.Iepsi=[]; self.E=[];
      for iqpol in range(self.nqpol):
        self.Iepsi.append([]);
 
      while len(line)<>0:       # for each following line
        data=map(float,line.split());
        self.E.append(data[0]);
        for iqpol in range(self.nqpol): # skip averaged data
          self.Iepsi[iqpol].append(data[iqpol+self.nqavg+1]);   
        line=file.readline();

    # IF  Re(eps) and Im(eps) given
    if self.param['quantity'] in ("mdf"):
      self.Reps=[]; self.Ieps=[]; self.E=[];
      for iqpol in range(self.nqpol):
        self.Reps.append([]); self.Ieps.append([]);

      while len(line)<>0:       # for each following line
        data=map(float,line.split());
        self.E.append(data[0]);
        for iqpol in range(self.nqpol): # skip averaged data
          self.Reps[iqpol].append(data[2*(iqpol+self.nqavg)+1]); 
          self.Ieps[iqpol].append(data[2*(iqpol+self.nqavg)+2]);
        line=file.readline();

    # IF other quantity given (specified by p['desc'])
    if self.param['quantity'] in ("multcol"):
      self.multcol=[]; self.E=[];
      while len(line)<>0:       # for each following line
        data=map(float,line.split());
        self.E.append(data[0]);
        self.multcol.append(data[1:]);
        line=file.readline();
      # change to standard order: column index first, then energy index    
      self.multcol = np.transpose(self.multcol);  

    file.close();



  def write_header(self, out=sys.stdout):
    "write all calculation parameters as DP-header to out (default: stdout)"

    p=dict(); p['nqpol']=self.nqpol;
    for key in ('G_rc','G_cc','qred_rc','qred_cc'):
      p[key] = np.array(self.__dict__[key],float);

    mdf_header(self.param, p, out); 
 

  def get_mdf(self):
    """
    return a list containing the macroscopic dielectric function
    for each polarization direction (instances of class MDF)
    """
    if (self.param['quantity'] <> "mdf"):
      raise ValueError("Expected DP input: *.mdf or *.eps files");

    spectra=[];
    for iqpol in range(self.nqpol): # create MDF for each polarisation

      E     = np.asarray(self.E,float);
      eps   = np.asarray(self.Reps[iqpol],complex)\
            +1j*np.array(self.Ieps[iqpol],complex);
    
      # DP parameter (for the polarization iqpol)
      p={'iqpol': iqpol};
      p['iqpol']   = iqpol;
      p['q_rc']    = self.q_rc[iqpol];
      p['q_cc']    = self.q_cc[iqpol];
      p['qred_rc'] = self.qred_rc[iqpol];
      p['qred_cc'] = self.qred_cc[iqpol];
      p['G_rc']    = self.G_rc;
      p['G_cc']    = self.G_cc;

      # log
      if 'comment' not in self.param:
        self.param['comment'] = \
            ["# created with $Id: dp_mdf.py 479 2014-03-10 11:29:40Z hambach $ \n",
             "#  -read from file: %s, iqpol: %d\n" %(self.param['filename'],iqpol)];

      # create MDF
      spectra.append(MDF(self.param,p,E,eps));
    return spectra;





def mdf_header(param, pupdate=None, out=sys.stdout):
  """
  write calculation parameters in p as DP-header to out (default: stdout)
  param  ... dictionary that contains the keys
              quantity, lfe, kernel, emin, emax, dsf_fact
              nkbz, nkibz, npwmat, npwwfn, nbands, lomo, broad,
              G_rc, qred_rc, G_cc, qred_cc,
              (opt) ucvol, comment
  pupdate... (opt) dict to overwrite param keys
  out    ... (opt) file descriptor for output
  """
  if pupdate is None:
    p = param;
  else:
    from copy import deepcopy;
    p = deepcopy(param);
    p.update(pupdate);

  # type of calculation
  if   p['quantity']=="mdf": out.write("# macroscopic dielectric function\n");
  elif p['quantity']=="eel": out.write("# electron energy loss\n"           );
  elif p['quantity']=="dsf": out.write("# dynamic structure factor\n"       );
  elif p['quantity']=="multcol": out.write("# multi-column (%s)\n" % p['desc']);
  else: raise ValueError("Unknown parameter p['quantity'] = '%s'"%p['quantity']);

  if   p['lfe']=="nlf": out.write("# without local fields\n"           );
  elif p['lfe']=="lf" : out.write("# with local fields\n"              );
  else: raise ValueError("Unknown parameter .p['lfe'] = '%s'"%p['lfe']);

  # "# testparticle  rpa" new version
  if   p['kernel']=="rpa": out.write("# RPA\n"                          );
  elif p['kernel']=="lda": out.write("# Adiabatic lda (TDLDA\n"         );
  elif p['kernel']=="so+alpha": out.write("# Long-range kernel al\n# \n"); # skip
  else: raise ValueError("Unknown parameter p['kernel'] = '%s'"%p['kernel']);
                 
  # calculation parameters
  out.write("# npwmat = %d  npwwfn = %d  nbands = %d  lomo = %d  nkibz = %d  nkbz = %d\n"
            %( p['npwmat'],p['npwwfn'],p['nbands'],p['lomo'],p['nkibz'],p['nkbz']));
  out.write("# energy range   %f  %f [eV]\n"      % (p['emin'],p['emax']));
  out.write("# lorentzian broadening = %f [eV]\n" % (p['broad']));
  out.write("# dsf factor    %f [a.u.]\n"         % (p['dsf_fact']));
  if 'ucvol' in p: out.write("# unit cell volume   %f [a.u.]\n" % (p['ucvol'])); # optional

  # G-vectors and polarisation directions
  out.write("# G = ( %g, %g, %g ) [r.l.u.]\n" % tuple(p['G_rc']));  # might be non-int in some cases (see interpolate_chi0.py)
  out.write("# G = ( %f, %f, %f ) [c.c.]\n"   % tuple(p['G_cc']));
  out.write("# \n"); # skip

  # write nqpol polarisation vectors
  nqpol=p['nqpol'];
  out.write("# %d polarizations + 0 averages:\n" % (nqpol)); # no averages
  for iqpol in range(nqpol):  
    out.write("# q = ( %f, %f, %f ) [r.l.u.]\n" \
                   % tuple(p['qred_rc'].reshape(nqpol,3)[iqpol]));
    out.write("# q = ( %f, %f, %f ) [c.c.]\n"   \
                   % tuple(p['qred_cc'].reshape(nqpol,3)[iqpol]));
    out.write("# \n"); # skip

  if 'comment' in p.keys():
    for line in p['comment']:
      out.write(line);


# ----------------------------------------------------------------
# useful functions for plotting

def broad(spectrum,E,broad):
  if broad<1e-5: return spectrum;

  dE     = (E[-1]-E[0])/(len(E)-1);
  xmax   = np.ceil(5*broad/dE);
  E_gauss= np.arange(-xmax,xmax+1)*dE;        # symmetric energies around 0 with step-size dE
  gauss  = np.exp(-0.5*(E_gauss/broad)**2);
  return  np.convolve(spectrum,gauss,'same')/np.sum(gauss);


def PrintFilenames(spectra):
  """
  Print list of files that contain the mdf's in spectra
  spectra... list of MDF-objects
  """
  files=[mdf.param['filename'] for mdf in spectra];
  seen=set();
  for f in files:
    if f not in seen:  
      print("using "+f);
      seen.add(f);

def find_mdf(pattern,q,comp=None,atol=1e-6,verbosity=1):
  '''
    search for mdf-file that match the specified pattern and
    has a momentum transfer q [a.u.] in cartesian coordinates
      comp ... (opt) function comp(q,qp) to compare the q vectors
      atol ... (opt) alternatively use absolute tolerance atol
      verbosity ... (opt) 0...silent, 3...debug
  '''
  # comparison between two q-vectors (cartesian coordinates)
  if comp is None: 
    comp = lambda q,qp: np.allclose( qcc, q, atol=atol);

  # iterate over all mdf-files that match the pattern
  for mdf in sorted(GetMDF(pattern),key=lambda mdf: mdf.param['filename']):
    qcc=mdf.get_q('cc','au');
    if verbosity>2:
      print mdf.param['filename'], q, qcc
    # found spectrum, return
    if comp(q,qcc):
      if verbosity>0: 
        print 'using file '+mdf.param['filename'];
      return mdf

  # no matching q-vector found
  error='no spectrum found for q = '+str(q)+' \n  pattern= '+pattern ;
  raise ValueError(error);


def find_DPmdf(pattern,q,comp=None,atol=1e-6,verbosity=1):
  '''
    search for DPmdf-file that match the specified pattern and
    has a momentum transfer q [a.u.] in cartesian coordinates
      comp ... (opt) function comp(q,qp) to compare the q vectors
      atol ... (opt) alternatively use absolute tolerance atol
      verbosity ... (opt) 0...silent, 3...debug
  '''
  # comparison between two q-vectors (cartesian coordinates)
  if comp is None: 
    comp = lambda q,qp: np.allclose( qcc, q, atol=atol);

  # iterate over all mdf-files that match the pattern
  for filename in sorted(glob.glob(pattern)):
    mdf=DPmdf(filename);
    # iterate over all polarisation directions in file
    for qcc in mdf.q_cc:
      if verbosity>2:
        print mdf.param['filename'], q, qcc
      # found spectrum, return
      if comp(q,qcc):
        if verbosity>0: 
          print 'using file '+mdf.param['filename'];
        return mdf

  # no matching q-vector found
  error='no spectrum found for q = '+str(q)+' \n  pattern= '+pattern ;
  raise ValueError(error);




### Self-Tests
if __name__ == '__main__':

  def is_different(a1,a2,epsilon=1e-4):
    a1=np.asarray(a1);
    a2=np.asarray(a2);
    return np.any( np.abs(a1-a2) > epsilon*(np.abs(a1)+np.abs(a2)) );

  def write_header_fails(orig):
    # write and read again
    out  = open("tmp.mdf","w");
    orig.write_header(out); out.close()
    copy = DPmdf("tmp.mdf");

    # compare parameters
    for key in orig.param.keys():
      if key=='filename': continue  # skip
      if orig.param[key]<>copy.param[key]:
        print("-> DPmdf.write_header() failed for parameter key '%s' \n" % (key));
        print("  orig: \n"+str(orig.param[key])+"\n\n   copy: \n" +str(copy.param[key]));
        return True;

    # check q-vectors
    for iqpol in range(orig.nqpol):
      if any(np.abs(orig.q_cc[iqpol] - copy.q_cc[iqpol]) > 1e-6) or \
         any(np.abs(orig.q_rc[iqpol] - copy.q_rc[iqpol]) > 1e-6):
        print("-> DPmdf.write_header() failes for q-vector (iqpol=%d)" % iqpol);        return True;

    # check G-vectors 
    if any(np.abs(orig.G_cc - copy.G_cc) > 1e-6) or \
       any(np.abs(orig.G_rc - copy.G_rc) > 1e-6):
      print("-> DPmdf.write_header() failes for q-vector (iqpol=%d)" % iqpol); 
      return True;
      
    return False

  def write_mdf_fails(mdf,orig):
    # write and read again
    out  = open("tmp.mdf","w");
    mdf.write(out, orig.param['quantity']); out.close()
    copy = DPmdf("tmp.mdf");

    # compare parameters
    for key in  ['lfe', 'kernel', 'nkibz', 'emin', 'emax', 'npwmat', 
      'nbands', 'broad', 'dsf_fact', 'lomo', 'nkbz', 'npwwfn', 'quantity']:
      if orig.param[key]<>copy.param[key]:
        print("-> MDF.write() failed for parameter key '%s' \n" % (key));
        print("  orig: \n"+str(orig.param[key])+"\n\n   copy: \n" +str(copy.__dict__[key]));
        return True;

    # check q-vectors
    for key in [ 'G_rc', 'G_cc', 'q_rc', 'qred_rc', 'q_cc', 'qred_cc']:
      if any(np.abs(orig.__dict__[key].flatten()  \
                  - copy.__dict__[key].flatten()) > 1e-6):
        print("-> MDF.write() failes for q-vector '%s'" % key);
        return True;

      
    return False
  


  print("\n Tests for dp_io.py ... ");

  ### 1. order of data columns + header
  print("\n   1.1 DPmdf(*.mdf), 1 polarization");
  data = DPmdf("tests/1pol.mdf");
  if np.array(data.Reps,int)[:,0].flatten().tolist()<>[1] or \
     np.array(data.Ieps,int)[:,0].flatten().tolist()<>[2] or \
     write_header_fails(data):
    print("ERROR! in Test 1.1");
  
  print("   1.2 DPmdf(*.dsf), 1 polarization");
  data = DPmdf("tests/1pol.dsf");
  if np.array(data.Iepsi,int)[:,0].flatten().tolist()<>[1] or \
     write_header_fails(data):
    print("ERROR! in Test 1.2");
  
      
  print("   1.3 DPmdf(*.mdf), 6 polarizations");
  data = DPmdf("tests/6pol.mdf");
  if np.array(data.Reps,int)[:,0].flatten().tolist()<>[5,7, 9,11,13,15] or \
     np.array(data.Ieps,int)[:,0].flatten().tolist()<>[6,8,10,12,14,16] or \
     write_header_fails(data):
    print("ERROR! in Test 1.3");
  
  print("   1.4 DPmdf(*.eel), 6 polarizations");
  data = DPmdf("tests/6pol.eel");
  if np.array(data.Iepsi,int)[:,0].flatten().tolist()<>[3,4,5,6,7,8] or \
     write_header_fails(data):
    print("ERROR! in Test 1.4");

  print("   1.5 DPmdf(*.mdf), 9 polarizations (dpforexc)");
  data = DPmdf("tests/9pol.mdf");
  if np.array(data.Reps,int)[:,0].flatten().tolist()<>[5,7, 9,11,13,15] or \
     np.array(data.Ieps,int)[:,0].flatten().tolist()<>[6,8,10,12,14,16] or \
     write_header_fails(data):
    print("ERROR! in Test 1.5");
  
  print("   1.6 DPmdf(*.dsf), 9 polarizations (dpforexc)");
  data = DPmdf("tests/9pol.dsf");
  if np.array(data.Iepsi,int)[:,0].flatten().tolist()<>[3,4,5,6,7,8] or \
     write_header_fails(data):
    print("ERROR! in Test 1.6");

  print("   1.7 DPmdf(*.mcl), 1 polarization  (generic multi-column format)");
  data = DPmdf("tests/1pol.mcl");
  if np.array(data.multcol,int)[:,0].flatten().tolist()<>[1,2] or \
     write_header_fails(data):
     print("ERROR! in Test 1.7");
  
  ### 2. MDF interface
  print("\n   2.1 MDF.get_dsf()");
  mdf = DPmdf("tests/1pol.mdf").get_mdf()[0];
  dsf = DPmdf("tests/1pol.dsf");
  # skip first row of the test input file
  if is_different( mdf.get_dsf()[1:], dsf.Iepsi[0][1:]):
    print("ERROR! in Test 2.1");
    
  print("   2.2 MDF.get_q()");
  if is_different( mdf.get_q(coord="cc"), dsf.q_cc[0] ):
    print("ERROR! in Test 2.2");

  print("   2.3 MDF for multiple polarizations");
  data = DPmdf("tests/9pol.mdf"); 
  if is_different( data.get_mdf()[4].get_Ieps(), data.Ieps[4] ):
    print("ERROR! in Test 2.3: spectra differ");
  if is_different( data.get_mdf()[4].get_q(),    data.q_rc[4] ):
    print("ERROR! in Test 2.3: q differ");

  print("   2.4 MDF.set_E()");
  data = DPmdf("tests/1pol.mdf");
  mdf2 = data.get_mdf()[0];
  mdf2.set_Emin(1.2);       # should influence interpolation
  mdf2.set_E(np.arange(3,10,0.5,dtype=float));
  if is_different( mdf2.get_Ieps(), data.Ieps[0][7:21] ):
    print("ERROR! in Test 2.4");

  print("   2.5 MDF.write()");
  mdf = DPmdf("tests/1pol.mdf");
  dsf = DPmdf("tests/1pol.dsf");
  if write_mdf_fails(mdf.get_mdf()[0],mdf):
    print("ERROR! in Test 2.5: mdf->mdf read and write");
  if write_mdf_fails(mdf.get_mdf()[0],dsf):
    print("ERROR! in Test 2.5: mdf->dsf read and write");

  ### 3. Helper functions
  print("\n   3.1 find_mdf()");
  q1 = (0.000000, 0.493369, 0.284847);
  q2 = (0.000000, 0.000000, 0.000010);
  mdf= find_mdf('tests/1pol.mdf', q1, verbosity=0);
  assert np.allclose(mdf.get_q('cc'), q1); # 1 polarisation 
  mdf= find_mdf('tests/9pol.mdf', q2, verbosity=0); 
  assert np.allclose(mdf.get_q('cc'), q2); # 6 polarisations
  mdf= find_mdf('tests/?pol.mdf', q1, verbosity=0); 
  assert np.allclose(mdf.get_q('cc'), q1); # multiple files

  print("   3.2 find_DPmdf()");
  q1 = (0.000000, 0.493369, 0.284847);
  q2 = (0.000000, 0.000000, 0.000010);
  mdf= find_DPmdf('tests/1pol.mdf', q1, verbosity=0);
  assert np.allclose(mdf.q_cc[0], q1); # 1 polarisation 
  mdf= find_DPmdf('tests/9pol.mdf', q2, verbosity=0); 
  assert np.allclose(mdf.q_cc[2], q2); # 6 polarisations
  mdf= find_DPmdf('tests/?pol.mdf', q1, verbosity=0); 
  assert np.allclose(mdf.q_cc[0], q1); # multiple files
  
  print("\n Done.");