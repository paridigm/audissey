from pyo import *

s = Server().boot()
s.start()

mic = Input([0,1])
spec = Spectrum(mic, size=1024)

s.gui( locals() )
