
from pyo import *

s = Server().boot()
s.start()

a = Sine(mul=0.01).out() # why does sine.play() not work?

s.gui(locals())

# try to make fade and replay
# example(Harmonizer)
