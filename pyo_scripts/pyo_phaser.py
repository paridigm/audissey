
from pyo import *

s = Server().boot()
s.start()

fade = Fader(fadein=.1, mul=.07).play()
a = Noise(fade).mix(2).out()

lf1 = Sine(freq=[.1, .15], mul=100, add=250)
lf2 = Sine(freq=[.18, .15], mul=.4, add=1.5)

b = Phaser(a, freq=lf1, spread=lf2, q=1, num=20, mul=.5).out(0)

s.gui( locals() )
