from pyo import *

s = Server().boot()
s.start()

# I think the call may have to exist in the file scope to work...
# Solution: load up an array with callbacks

# Start an oscillator with a frequency of 250 Hz
syn = SineLoop(freq=[250,251], feedback=0.0, mul=.2).out()

# change value
def changeVals(inc):
    syn.freq = [syn.freq[0] + inc, syn.freq[1] + inc]
    print(syn.freq)

# number of notes in song
pyo_calls = [None] * 10

print(pyo_calls)

def up():
    pyo_calls[0] = CallAfter(changeVals, 0, (25))
    pyo_calls[1] = CallAfter(changeVals, 1 / float(2), (25))
    pyo_calls[2] = CallAfter(changeVals, 2 / float(2), (25))
    pyo_calls[3] = CallAfter(changeVals, 3 / float(2), (25))

def down():
    pyo_calls[0] = CallAfter(changeVals, 0, (-25))
    pyo_calls[1] = CallAfter(changeVals, 1 / float(2), (-25))
    pyo_calls[2] = CallAfter(changeVals, 2 / float(2), (-25))
    pyo_calls[3] = CallAfter(changeVals, 3 / float(2), (-25))

s.gui(locals())

# ---------------------------------------------------------

# recursive playsong - didn't work
def playSong(i):

    # base case
    if(i > 10):
        return

    # change val
    syn.freq = [syn.freq[0] + 50, syn.freq[1] + 50]
    print(syn.freq)

    # recursive call
    call = CallAfter(playSong, 1, (i+1))
    print("playSong index: ", i)

# playSong(0)



