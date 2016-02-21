
from pyo import *

s = Server().boot()
s.start()


f = Phasor(freq=[1, 2], mul=1000, add=0)
sine = Sine(freq=f, mul=f/1000).out()  #passing an array into an argument here duplicates another sin wave


s.gui(locals())
