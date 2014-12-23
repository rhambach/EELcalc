from GracePlot import *
import math

p = GracePlot() # A grace session opens

l1=Line(type=lines.none)


x1=map(lambda x:x/10.,range(0,100))
y1=map(math.sin,x1)
y2=map(math.cos,x1)

d2=Data(x=x1,y=y1,
        symbol=Symbol(symbol=symbols.circle,fillcolor=colors.red),
        line=l1)
d3=Data(x=x1,y=y2,
        symbol=Symbol(symbol=symbols.circle,fillcolor=colors.blue),
        line=l1)

g=p[0]

g.plot([d2,d3])

g.xaxis(label=Label('X axis',font=5,charsize=1.5),
        tick=Tick(majorgrid=True,majorlinestyle=lines.dashed,majorcolor=colors.blue,
                  minorgrid=True,minorlinestyle=lines.dotted,minorcolor=colors.blue))
g.yaxis(tick=Tick(majorgrid=True,majorlinestyle=lines.dashed,majorcolor=colors.blue,
                  minorgrid=True,minorlinestyle=lines.dotted,minorcolor=colors.blue))
