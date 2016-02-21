import time
from pyo import *

s = Server().boot()
s.start()

sf = SfPlayer(SNDS_PATH + '/transparent.aif', loop=True, mul=.3).out()
harm = Harmonizer(sf, transpo=-5, winsize=0.05).out(1)

time.sleep(5.000000) # does this pause the interpreter?
s.stop()

time.sleep(0.25)
s.shutdown()
