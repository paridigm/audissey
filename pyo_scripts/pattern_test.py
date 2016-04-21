from pyo import *

s = Server().boot()
s.start()

t = HarmTable([1,0,.33,0,.2,0,.143,0,.111])
a = Osc(table=t, freq=[250,251], mul=.2).out()

def pat():
     f = random.randrange(200, 401, 25)
     a.freq = [f, f+1]

p = Pattern(pat, .125)
p.play()

s.gui(locals())

