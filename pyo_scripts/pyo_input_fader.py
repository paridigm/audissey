from pyo import *

s = Server().boot()
s.start()

a = SineLoop([449,450], feedback=0.05, mul=.2)
b = SineLoop([650,651], feedback=0.05, mul=.2)

c = InputFader(a).out()
# c.__setattr__("fadetime", 5) # didnt work, tyrintg to just fade a single audio signal

c.setInput(b, fadetime=1) # assign a new audio input to created a crossfade

s.gui(locals())
