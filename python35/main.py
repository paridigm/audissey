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
import onsetdetect_hfc as od


def main():
    
    # ---------------------- inits --------------------------------
    
    # init vars
    file_str = "test.wav"
    file_str = "beat_boxing_sample.wav"
    p = pyaudio.PyAudio()
    CHUNK = 1024
    
    # one wav file for playing, the other for plotting
    wf = wave.open(file_str, 'rb')
    wf_plot_fs, wf_plot = wavfile.read(file_str)
    
    # output some info
    print(wf_plot_fs)
    print("dtype:", wf_plot.dtype)
    print("Shape:", wf_plot.shape) # shape = [frames, channels]
    
    # ---------------------- get data --------------------------------
    
    s1 = None
    size = 0
    
    # if array is more than one dimensional    
    if( len(wf_plot.shape) > 1 ):
        s1 = numpy.array( wf_plot[:,0] )
    else:
        s1 = numpy.array( wf_plot )
    
    
    # ------------------------ filter ----------------------------    
    
    # s_float = s1.astype( numpy.float32 )
    
    # apply onset detector found on github
    onsets = od.detect_onsets(s1)
    
    # TODO plot the onsets
    onsets_visual = numpy.zeros(s1.size)
    
    for x in numpy.nditer(onsets):
        onsets_visual[x] = 1
    
    
    # ---------------------- plot --------------------------------
    
    # create time array in milliseconds
    timeArray = numpy.arange(start=0, stop=s1.size, step=1, dtype=float)
    # timeArray = timeArray / wf_plot_fs
    # timeArray = timeArray * 1000                          # scale to milliseconds
    
    # SIGNAL
    matplotlib.pyplot.figure(1)
    matplotlib.pyplot.subplot(211)
    matplotlib.pyplot.plot(timeArray, s1, color='k')
    matplotlib.pyplot.ylabel('Amplitude')
    matplotlib.pyplot.xlabel('Time (ms)')
    
    # ONSETS    
    matplotlib.pyplot.subplot(212)
    matplotlib.pyplot.plot(timeArray, onsets_visual, color='k')
    matplotlib.pyplot.ylabel('Onset')
    matplotlib.pyplot.xlabel('Time (ms)')
    
    
    '''
    plt.figure(1)
    plt.gcf().suptitle('Bold Title', fontsize=14, fontweight='bold')
    plt.gcf().canvas.set_window_title('Test')
    
    plt.subplot(321) # 3 row, 2 column, in plot location 1
    plt.plot(t, s1) 
    '''
    
    matplotlib.pyplot.show(block=False)
        
    
    # ---------------------- playback --------------------------------
    
    # open steam for playback
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    
    # play (chunk by chunk)
    for i in range(0,  int(wf.getnframes()/CHUNK) ):
        data = wf.readframes(CHUNK)
        stream.write(data)
    
    print("just about done")
    stream.stop_stream()
    stream.close()
    
    
    # ---------------------- terminate --------------------------------
    p.terminate()  # close pyaudio
    
    print("all done")
    
    '''
    # poll and wait for user to say they are done with program
    while True:
        n = input("Done:")
        if n.strip() == "y":
            break
    '''
    
    # blocking code in the end (to show the plot)
    matplotlib.pyplot.show()
    print()
    
    sys.exit(0)

main()