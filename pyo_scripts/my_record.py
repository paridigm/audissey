'''
script to record data and save it into a buffer.

call _r() to start recording
call _p() to stop recording and playback the recording
'''

from pyo import *

s = Server().boot()
s.start()

MAX_TIME = 4

mic = Input([0,1])
buf = NewTable(length = MAX_TIME, chnls = 2 )

# Table is different that these, cannot "out()" a buffer...
rec = TableRec(mic, table=buf)                #recorder object that fills in a buffer
playback = TableRead(buf, freq=1.0/MAX_TIME)  #reader object that can play a buffer

# Oscilloscope and spectrum analyzer
scope = Scope(mic)
#spec = Spectrum(playback, size=1024)


print(SNDS_PATH)

def _r():   #record
    rec.play()
    print("START RECORDING")

def _p():   # stop and perform a playback
    rec.stop()
    playback.out()

    print("DONE RECORDING")

'''
def plot():
    # how can I do this? --> I can't in pyo... need to save to wav file then use matplotlib?

IDEA for plotting:
    convert value into a control signal --> send control signal value to a live plotter window thing
    (Hopefully this will work!)

    need to grab features for classification
        - try MFCC's --> backup = spectral centroid + freaqnecy

'''

s.gui(locals())

