from pyo import *

s = Server().boot()
s.start()

'''
sf = SfPlayer(SNDS_PATH + "/transparent.aif").out()
trig = TrigRand(sf['trig'])
'''

sf = SfPlayer("./samples/beat_boxing_sample.wav")
sf = SfPlayer("./samples/snare_sample.wav")

def do():
    sf.out()
    trig = TrigRand(sf['trig'])

print(SNDS_PATH)

'''
snd = SNDS_PATH + "/transparent.aif"
sf = SfPlayer(snd, speed=[0.8, 1.2], loop=True, mul=.5).out()
'''

s.gui(locals() )
