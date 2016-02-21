
from pyo import *

s = Server(duplex=1).boot()
s.start()

a = Input()


# d = AttackDetector(a, deltime=0.005, cutoff=10, maxthresh=4, minthresh=-20, reltime=0.05)

# d = Phasor(freq=1, mul=1000, add=0)
# sine = Sine(freq=d, mul=d/1000/3).out()  #passing an array into an argument here duplicates another sin wave


# exc = TrigEnv(d, HannTable(), dur=0.005, mul=1)
# wgs = Waveguide(exc, freq=[100,200.1,300.3,400.5], dur=30).out()


# each time there is an onset, play a small sin wav

s.gui(locals())
