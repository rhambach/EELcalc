from GracePlot import *

import math

#this is an example showing how grace can be run remotely over an ssh session.  
#Note that it isn't being run very remotely here, since the host is localhost
#also note that, as written, this works on MacOSX with fink, since it has a hard-wired 
#path to /sw/bin/xmgrace
 
class GPSSH(GracePlot):
    grace_command = 'ssh'
    command_args = ('-X', 'localhost', '/sw/bin/xmgrace', ) + GracePlot.command_args 
    
a=GPSSH(width=8, height=6, auto_redraw=True)
xvals=range(100)
yvals=[math.sin(x*1.0) for x in xvals]
y2vals=[math.cos(x*0.5) for x in xvals]
a.assign_color(colors.yellow, (128, 128, 0), "yellow-green")
a.assign_color(20, (64, 64, 0), "dark yellow-green")
g=a[0]

g.plot([
    Data(x=None, y=None, pairs=zip(xvals, yvals),
        line=Line(type=lines.none), symbol=Symbol(symbol=symbols.plus, color=colors.green4),
        errorbar=Errorbar(color=colors.green4), legend='hello'
    ), ])

g.title("Remote Control plot!")
g.legend()

