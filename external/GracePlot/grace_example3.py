from GracePlot import *
import math
import random

p = GracePlot(width=8, height=6) # A grace session opens

x=[1,2,3,4,5,6,7,8,9,10]
y=[1,2,3,4,5,6,7,8,9,10]
labels=['pt1','pt2','Iridium','Na','Ti','hydrogen','Mo '+format_scientific("1.23e3") ,'Ta','pokemon','digital']

dy=map(lambda x:random.random()*2.,x)

s1=Symbol(symbol=symbols.square,fillcolor=colors.cyan)
l1=Line(type=lines.none)

d1=DataXYDY(x=x,y=y,dy=dy,symbol=s1,line=l1)

g=p[0]
g.xaxis(xmin=0, xmax=12)
g.yaxis(ymin=0, ymax=12)

g.plot(d1, autoscale=False)

for i in range(len(labels)):
    g.text('  '+labels[i],x[i],y[i],color=colors.violet,charsize=1.2)

g.line(x1=3, y1=1, x2=8, y2=2, linewidth=3, color=colors.green4)
