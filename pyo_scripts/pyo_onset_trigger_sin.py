from pyo import *

s = Server().boot()
s.start()

a = Input()
m = AttackDetector(a, deltime=0.005, cutoff=10, maxthresh=0.5, minthresh=-30, reltime=0.001).play()

tr = TrigRand(m, 800, 800)  # random value between 800 and 800... yes 800
te = TrigEnv(m, table=HannTable(), dur=.25, mul=.2)
b = Sine(tr, mul=te).out()

# also want to record input
scope = Scope(a)

s.gui( locals() )
