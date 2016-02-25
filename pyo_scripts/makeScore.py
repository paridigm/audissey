from pyo import *

inputList = [['Bass', 0], ['Hi-Hat', 0.8], ['Snare', 1.6], ['Hi-Hat', 2.4]]

# initialize server
s = Server().boot()
s.start()

# load addresses of sounds
# Note: I put the sample sound files in the SNDS_PATH directory, you can print it out if you're
# unsure where it is.

# print SNDS_PATH
Bass = SNDS_PATH + '/Bass.wav'
Hi_Hat = SNDS_PATH + '/Closed-Hi-Hat.wav'
Snare = SNDS_PATH + '/Roland-R-8-Fat-Snare.wav'

# initialize the table
t = SndTable()

# populate the table with desired sounds and timings
for elt in inputList:
    if elt[0] == 'Bass':
        t.insert(Bass, pos=elt[1])
    elif elt[0] == 'Hi-Hat':
        t.insert(Hi_Hat, pos=elt[1])
    else:
        t.insert(Snare, pos=elt[1])

t.view()
s.gui(locals())
