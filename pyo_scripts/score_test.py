from pyo import *
s = Server().boot()
s.start()

a = SineLoop(freq=[200,300,400,500], feedback=0.05, mul=0.1).out()

def event_1():                  # order doesnt matter
    a.freq=[300,400,450,600]
def event_0():
    a.freq=[200,300,400,500]
def event_2():
    a.freq=[150,375,450,525]
def event_3():
    a.freq=[150,375,450,700]

m = Metro(1).play()             #
c = Counter(m, min=0, max=4)    #
sc = Score(c)


s.gui(locals())
