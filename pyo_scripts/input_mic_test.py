
from pyo import *

s = Server(duplex=1).boot()
s.start()
a = Input(chnl=0, mul=.7)
# b = Delay(a, delay=.1, feedback=.5, mul=.5).out()

a.out()

s.gui()
