from pyo import *

inputList = [['Bass', 0], ['Hi-Hat', 0.8], ['Snare', 1.6], ['Hi-Hat', 2.4]]

# initialize server
s = Server().boot()
s.start()

# load addresses of sounds
# Note: I put the sample sound files in the SNDS_PATH directory, you can print it out if you're
# unsure where it is.

# print SNDS_PATH
Bass = SNDS_PATH + '/bass_sample.wav'
Hi_Hat = SNDS_PATH + '/hihat_sample.wav'
Snare = SNDS_PATH + '/snare_sample.wav'

print("test")

# initialize the table
t = SndTable()

print("test2")

# populate the table with desired sounds and timings
for elt in inputList:

    # elt ~ element
    print(elt[0])

    if elt[0] is 'Bass':
        t.insert(Bass, pos=elt[1])
    elif elt[0] is 'Hi-Hat':
        t.insert(Hi_Hat, pos=elt[1])
    else:
        t.insert(Snare, pos=elt[1])

print("test3")

# a = Osc(table=t, freq=t.getRate(), mul=.3).out()

print("test4")

s.gui(locals())
