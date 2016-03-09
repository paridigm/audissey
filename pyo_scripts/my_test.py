from pyo import *
import time

s = Server().boot()
s.start()

a = Sine(mul=0.05).out() # why does sine.play() not work?

time.sleep(2)
s.stop()

s.gui(locals())

# try to make fade and replay
# example(Harmonizer)
