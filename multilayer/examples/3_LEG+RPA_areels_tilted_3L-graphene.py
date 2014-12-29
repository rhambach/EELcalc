"""
  Plot script for decomposition of AR-EEL spectra of tilted
  multilayer graphene into normal mode spectra (see [1] Fig. 3b).

  USAGE:
   

   Choose the number of layers N, interlayer distance d, sample tilt
   and the type of approximation for the polarizablity of graphene 
   and execute the plot script. Possible approximations are

   'RPA' calculation uses results of ab-initio calculations
         for the polarizability of graphene. As these calculations
         have been performed for a discrete set of q-values, 
         the qtot-values have to be chosen accordingly for each
         given sample tilt. (see RPA.get_q() for a list of allowed
         in-plane q-values).

   'hydrodynamic' calculation uses a very simple 
         two-pole lorentz model for the polarizability 
         of graphene, which can be evaluated for arbitrary q
         but gives rather poor quantitative results.

  REFERENCES: 
    [1] Wachsmuth, Hambach, Benner, and Kaiser, PRB (90) 235434 (2014)

  Copyright (c) 2014, rhambach, P. Wachsmuth. 
    This file is part of the EELcalc package and released
    under the MIT-Licence. See LICENCE file for details.
"""


import numpy as np;

from EELcalc.monolayer.graphene import rpa;
from EELcalc.multilayer.leg import areels;
import EELcalc.monolayer.graphene.hydrodynamic_model as hm;
import EELcalc.external.GracePlot.GracePlot as gp;


# -- PARAMETERS --------------------------------------
N = 3;                # number of layers
d = 3.334 / hm.BOHR;  # interlayer distance [a.u]
calc = 'RPA';         # 'RPA' or 'hydrodynamic' calc for graphene
tilt = 60;            #  tilt angle in degree
q_list= np.asarray([0.00001,0.1,0.2,0.25,0.3,0.4,0.5]);
                      # in-plane components of q-values [1/A]
                      # selected accoring to RPA calculations
qtot_list = q_list/np.cos(np.deg2rad(tilt));
                      # measured q-range (= q_tot [1/A])

 

# --- draw graph -----------------------------------
def SetPalette(fig,NUM_COLORS,offset=2):
  "re-define colors except for white (0) and black (1)"
  color=offset; i=1;
  for x in map(int, np.linspace(0,255,NUM_COLORS)):
    c=np.array( (x, 0, 255-x) );
    fig.assign_color(color, map(int,c), "color%d"%(i));
    color+=1; i+=1;

def SetGreys(fig,NUM_COLORS,offset=2):
  "re-define colors except for white (0) and black (1)"
  color=offset; i=1;
  for x in map(int, np.linspace(0,255,NUM_COLORS)):
    c=np.array( (x, 0, 255-x) );
    o=0.4-0.15*(i%2);              # opacity 
    c=(o*c+(1-o)*255*np.ones(3));
    fig.assign_color(color+NUM_COLORS, map(int,c), "grey%d"%(i));
    color+=1; i+=1;

i = 0; data=[];
fig = gp.GracePlot(width=6, height=8 );
ax  = fig[0];

# --- setup polarizability for graphene
gamma = np.deg2rad(tilt); # tilt angle in rad
if calc == 'hydrodynamic':
 E = np.arange(0,40,0.1) # energies [ev]
 Graphene = hm.get_2FHM_Graphene_Jovanovic(w=E/hm.HARTREE);#gamma=0.1 / hm.HARTREE);
 ML = areels.Multilayer(N,d,Graphene.pol);
 
elif calc == 'RPA':
 RPA= rpa.GrapheneDP(qdir='GM',verbosity=0);
 E  = RPA.get_E();
 ML = areels.Multilayer(N,d,RPA.get_pol2D);


# --- calculate EELS (fixed pair of values q and qz)
for i,qtot in enumerate(qtot_list):
  qz = qtot*np.sin(gamma)*hm.BOHR;  # out-of-plane q
  q =  qtot*np.cos(gamma)*hm.BOHR;  # in-plane q
  offset=i;

  # calculate partial spectra
  l_modes = ML.areel_lmodes(q,qz);
  P = np.sum(l_modes,axis=0);  # total spectrum (sum of all modes)
  scale = 1./P.max();

  # plot l-modes partial sums
  for l in range(N,-1,-1):
    data.append( gp.Data(E, y= np.sum(l_modes[0:l+1],axis=0)*scale + offset,
        line=gp.Line(linestyle=0,filltype=2,baselinetype=1,fillcolor="\"grey%d\""%(l+1))) );

  # plot total spectrum
  data.append( gp.Data(x=E, y= P*scale + offset,
    line=gp.Line(linestyle=1,color=1,linewidth=1)));
  
  # add on-axis spectrum for comparison (l=0 mode only)
  P_ref = ML.areel(q,0);
  scale = 1./P_ref.max();
  data.append( gp.Data(x=E, y= P_ref*scale + offset,
             line=gp.Line(linestyle=3,color=1,linewidth=1)));

  # plot q-values
  ax.text(" %3.2f, %3.2f"%(qz/hm.BOHR,q/hm.BOHR),32,offset+0.2+q,color=1,charsize=1,font=4);
ax.text(' qz  \\oq\\O [1/\\cE\\C]',32,offset+1,color=1,charsize=1);

# plot legend for lmodes
for l in range(N):
  data.append(gp.Data(x=[0,0],y=[0,0],
    line=gp.Line(linestyle=1,linewidth=10,color="\"grey%d\""%(l+1)),
    legend="l=%d"%l));
data.append(gp.Data(x=[0,0],y=[0,0],line=gp.Line(linestyle=3,linewidth=1,color=1),legend="untilted"));

# -- plot figure ------------------------------------------------
SetPalette(fig,i,16);
SetGreys(fig,N+1,16+i);
ax.plot(data);

ax.title("AR-EELS, %d-layer graphene, tilt \\xg\\f{}=%d\\c0"%(N,tilt));
if calc=='hydrodynamic':
  ax.subtitle("(two-fluid hydrodynamic model, parameters from [Jovanovic, et.al. PRB(84) 155416 (2011)])",font=4,size=0.6);

elif calc=='RPA':
  p = RPA.calc_param;
  ax.subtitle("(RPA-calculations, job: %s, \\n pm%d, bd%d, pw%d, lo%d, broad %3.2f )"
              % (p['filename'].split('/')[-1],p['npwmat'], p['nbands'],
                 p['npwwfn'],   p['lomo'],  p['broad']),font=4,size=0.6 );

ax.yaxis(ymin=0, ymax=8,label=gp.Label('Intensity [arb. u.]',font=4,charsize=1.5))
ax.xaxis(xmax=40.0001,label=gp.Label('Energy [eV]',font=4,charsize=1.5))
ax.legend(x=0.15,y=1.16,world_coords=False,font=4,vgap=1);
fig.redraw(force=True)

