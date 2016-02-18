#####
#
# PyAudio Example with some added code for plotting.
#
#####

import pyaudio          # playback
import wave
import sys
import numpy            # plotting waveforms
import matplotlib.pyplot
from scipy.io import wavfile

# init
file_str = "test.wav"
p = pyaudio.PyAudio()
CHUNK = 1024

# one wav file for playing, the other for plotting
wf = wave.open(file_str, 'rb')
wf_plot_fs, wf_plot = wavfile.read(file_str)

print(wf_plot.dtype)
print(wf_plot.shape)  # shape = [frames, channels]

# plot
print( str(wf_plot[:,0]) + ' ' + str(wf_plot[:,1]) ) # sample, channel
s1 = wf_plot[:,0]

timeArray = numpy.arange(0, wf_plot.shape[0], 1)
timeArray = timeArray / wf_plot_fs
timeArray = timeArray * 1000  # scale to milliseconds

matplotlib.pyplot.plot(timeArray, s1, color='k')
matplotlib.pyplot.ylabel('Amplitude')
matplotlib.pyplot.xlabel('Time (ms)')
# matplotlib.pyplot.ion()  # prevents blocking in execution, but the plot is removed instantly :(
matplotlib.pyplot.show(block=False)

# open steam for playback
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

# play chunk by chunk
for i in range(0,  int(wf.getnframes()/CHUNK) ):
    data = wf.readframes(CHUNK)
    stream.write(data)

print("just about done")
stream.stop_stream()
stream.close()

p.terminate()  # close pyaudio

print("all done")

'''
while True:
    n = input("Done:")
    if n.strip() == "y":
        break
'''

# blocking code in the end (to show the plot)
matplotlib.pyplot.show()

sys.exit(1)

