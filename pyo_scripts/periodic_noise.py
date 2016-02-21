from pyo import *

s = Server().boot()
s.start()

f = Fader(fadein=0.1, fadeout=0.2, dur=0.3, mul=.5)
a = BrownNoise(mul=f).mix(2).out()                              # multiply by the fader

def repeat():
    f.play()            # causes the fader to play (back to beginning?)

pat = Pattern(function=repeat, time=0.5).play() # run pat.play()

s.gui( locals() )
