from GracePlot import *

p = GracePlot() # A grace session opens

x=[1,2,3,4,5,6,7,8,9,10]
y=[1,2,3,4,5,6,7,8,9,10]

s1=Symbol(symbol=symbols.circle,fillcolor=colors.red)
l1=Line(type=lines.none)

d1=Data(x=x,y=y,symbol=s1,line=l1)

g=p[0]

g.plot(d1)

g.text('test',.51,.51,color=2)

g.title('Graph Title')

g.yaxis(label=Label('Interesting Ydata',font=2,charsize=1.5))

g.xaxis(label=Label('X axis',font=5,charsize=1.5))
