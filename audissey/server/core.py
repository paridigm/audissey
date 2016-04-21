#####
#
# The core functions for audio processing + file processing.
#   - onset detection
#   - segmentation (from onsets)
#   - feature extraction
#   - saving data to files
#
# also a singleton file ('fake singleton class') makes threading simple
#
#####

import wave

import matplotlib.pyplot as plt
import numpy as np                  # plotting waveforms
import pyaudio                      # playback
import scipy.signal as signal       # smoothing
from scipy.io import wavfile

import util.signal_utils as su
import util.feature_utils as fu
import util.onsetdetect_hfc as odhf


################### vars ######################

# generate plots of each segmented sound from onset detector (keep off)
gen_plots = False
stop_rec = False

# sounds variables
snds_path = "./my_sounds/"           # location of sample wav files
data_path = "./data/"                # location of training data
user_data_path = "./data/user/"       # location of training data
file_str = None                      # file to open

# current wav file vars
wf = None                   # wav file
wf_plot = None              # plotting wav file
fs = 1                      # sampling frequency

# numpy signal vars
sig = None                  # signal
s_smooth = None             # smoothed signal

# peak detector vars
thresh_env = 400            # threshold for new onset to be detected
cutoff = 50                 # size of smoothing window
peaks = []          # peaks final
peaks_env = []      # peaks from my algo
peaks_hfc = []      # peaks from high frequency content
starts = []         # traverse down peaks
ends = []           # traverse up peaks
ip_intervals = []   # inter-peak intervals

# MFCC FFT vars
FFT_BUFF = 512              # MFCC FFT size
FFT_STEP = FFT_BUFF/2       # how much the MFCC window traverses per read
FFT_MULT = 3                # minimum spacing from peaks (multiples of FFT_BUFF)

# feature vector
mfccs = (1, 5)
X = []

####################  FUNCTIONS  ##########################

# ----------- Functions that combine processes ---------------

""" opens the wav file and creates sig, the current signal """
def openwav(file_name_str=None):
    open_new_wav_file(file_name_str)
    convert_wf_to_signal()

""" smooth sig --> onset extraction --> featureextraction """
def process():
    smooth_signal()
    get_peaks_and_segment()
    get_ip_intervals()
    extract_features_from_signal()

# ------------------------------------------------------------

""" set both data and sound path """
def set_default_path(rel_path):
    global snds_path, data_path, user_data_path
    snds_path = rel_path + "my_sounds/"
    data_path = rel_path + "data/"
    user_data_path = rel_path + "data/user/"


def set_data_path(data_path_str): # set data path only
    global data_path
    data_path = data_path_str


def set_user_data_path(user_data_path_str): # set user data path only
    global user_data_path
    user_data_path = user_data_path_str


def set_snd_path(snd_path_str): # set sound path only
    global snds_path
    snds_path = snd_path_str




def get_snd_path():
    global snds_path
    return snds_path


def get_data_path():
    global data_path
    return data_path


def get_user_data_path():
    global user_data_path
    return user_data_path


def record_stop():
    global stop_rec
    stop_rec = True


def open_new_wav_file(file_name_str=None):
    global wf, fs, wf_plot
    global file_str

    # set a new target if specified
    if(file_name_str is not None):
        file_str = file_name_str

    # one wav file for playing, the other for plotting
    wf = wave.open(snds_path + file_str, 'rb')
    fs, wf_plot = wavfile.read(snds_path + file_str)


def convert_wf_to_signal():
    global sig, wf_plot

    # if array is more than one dimensional
    if( len(wf_plot.shape) > 1 ):
        sig = np.array( wf_plot[:,0] ) # get channel 1
    else:
        sig = np.array( wf_plot )      # otherwise, get content


def smooth_signal():
    global sig, s_smooth, cutoff

    # FIR window smoothing
    s1_scale = sig*1          # scale
    s_abs = np.abs(s1_scale)  # rectify
    fir = signal.firwin(1024, float(cutoff)/fs, window = "hamming") # win size, cutoff
    s_smooth = np.convolve(s_abs, fir, mode="same")


def get_peaks_and_segment():
    global peaks, peaks_env, peaks_hfc
    global thresh_env, cutoff
    global starts, ends

    # search peaks (with threshold for closest) and cutoff delay (for time shift)
    peaks_env = su.search_peaks(s_smooth, thresh_env, cutoff)

    # apply onset detector found on github
    peaks_hfc = odhf.detect_onsets(sig)

    # do matching for both types of onsets
    max_dist = 2000
    peaks = su.compare(peaks_env, peaks_hfc, max_dist)

    # traverse peaks to get starts and stops
    starts, ends = su.traverse_starts_ends(peaks, s_smooth, thresh_env)


def get_ip_intervals():
    global peaks, ip_intervals

    if( len(peaks) <= 1):
        print("ONE OR LESS PEAKS - NO INTERVALS TO BE FOUND")
        return

    ip_intervals = [1]*(len(peaks)-1)

    for i in range(0, len(peaks)-1):
        ip_intervals[i] = peaks[i+1] - peaks[i]

    # debug
    temp = ip_intervals
    print(temp)


def extract_features_from_signal():
    global gen_plots
    global X, mfccs

    # space out starts and ends to be at least 1 window of MFCC (512?)
    for i in range(0, len(peaks)):
        if(peaks[i] - starts[i] < FFT_BUFF*FFT_MULT):
            starts[i] = peaks[i] - FFT_BUFF*FFT_MULT
        if(ends[i] - peaks[i] < FFT_BUFF*FFT_MULT):
            ends[i] = peaks[i] + FFT_BUFF*FFT_MULT

    # iterate over each segmented sound
    X = []

    for i in range(0, len(peaks)):

        # get note segmentation and normalize
        seg = sig[ starts[i]:ends[i] ]
        prev = sig[ starts[i]:peaks[i] ]
        post = sig[ peaks[i]:ends[i] ]
        env = s_smooth[ starts[i]:ends[i] ]
        env_prev = s_smooth[ starts[i]:peaks[i] ]
        env_post = s_smooth[ peaks[i]:ends[i] ]

        # normalize segment
        seg_max = np.amax(seg)
        env_max = np.amax(env)
        env = env.astype(float)           / env_max
        env_prev = env_prev.astype(float) / env_max
        env_post = env_post.astype(float) / env_max
        seg = seg.astype(float)   / seg_max
        prev = prev.astype(float) / seg_max
        post = post.astype(float) / seg_max

        """ EXTRACT FEATURES """
        features = fu.extract_features(prev, post, env_prev, env_post,
                                       fs, FFT_BUFF, FFT_STEP,
                                       mfcc_range=mfccs)

        # populate a samples list (X)
        X.append(features)

    #endfor
    X = np.array(X)


def save_unclassified_data():
    global X, data_path, file_str

    # save (unclassified data) to a file
    X_non = np.column_stack( (np.zeros( np.shape(X)[0] ), X) )
    file_out_str = file_str.replace(".wav", "_")
    #np.savetxt(data_path + file_str_out + "data.csv", X, delimiter='\t')
    np.savetxt(data_path + file_out_str + "data.csv", X_non, fmt='%10f', delimiter='\t')


def save_user_training_data(class_training, user):
    global X, user_data_path

    # set data to desired class
    X_cls = np.column_stack( (np.full(np.shape(X)[0], class_training, dtype=np.int), X) )

    # append to tom training data
    f=open( get_user_data_path() + user + ".csv",'ab')
    np.savetxt(f, X_cls, fmt='%10f', delimiter='\t')
    f.write("\n")
    f.close()


""" saves a new wav file appropiately """
def record(wav_file_name):
    global stop_rec
    stop_rec = False
    p = pyaudio.PyAudio()
    CHANNELS = 1
    FORMAT = pyaudio.paInt16
    CHUNK = 1024
    RATE = 44100
    RECORD_SECONDS = 3

    # open steam for playback
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True, frames_per_buffer=CHUNK)

    # record chunk by chunk, until stop_rec flag is raised
    frames = []

    #for i in range(0,  int(RATE/CHUNK)*RECORD_SECONDS):

    while(not stop_rec):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()  # close pyaudio

    waveFile = wave.open(snds_path + wav_file_name + ".wav", 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(p.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()


""" should be called in a new thread """
def playback():
    global wf
    global sig  # debugging (don't need this)

    p = pyaudio.PyAudio()
    CHUNK = 1024

    # debug
    print("PLAYING BACK AUDIO: " + file_str)
    print(wf.getnframes())
    print(sig)

    # open steam for playback
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # play (chunk by chunk)
    for i in range(0,  int(wf.getnframes()/CHUNK) ):
        data = wf.readframes(CHUNK)
        stream.write(data)

    stream.stop_stream()
    stream.close()
    p.terminate()  # close pyaudio


""" show peaks, starts and ends """
def show_visual():
    global peaks, peaks_env, peaks_hfc
    global starts, ends
    global ip_intervals
    global sig, s_smooth, thresh_env

    # debugging
    peaks_local = peaks
    peaks_env_local = peaks_env
    peaks_hfc_local = peaks_hfc
    starts_local = starts
    ends_local = ends
    sig_local = sig

    """ plotting vars """
    thresh_visual = np.full(sig.size, thresh_env, dtype = float)

    peaks_env_visual = np.zeros(sig.size)
    peaks_hfc_visual = np.zeros(sig.size)
    peaks_visual = np.zeros(sig.size)
    starts_visual = np.zeros(sig.size)
    ends_visual = np.zeros(sig.size)
    ip_intervals_visual = np.array(ip_intervals)

    if(len(peaks_env) > 0):
        for x in peaks_env:
            if(x < sig.size):
                peaks_env_visual[x] = 1

    if(len(peaks_hfc) > 0):
        for x in peaks_hfc:
            if(x < sig.size):
                peaks_hfc_visual[x] = 1

    if(len(peaks) > 0):
        for x in peaks:
            if(x < sig.size):
                peaks_visual[x] = 1

    if(len(starts) > 0):
        for x in starts:
            if(x < sig.size):
                starts_visual[x] = 1

    if(len(ends) > 0):
        for x in ends:
            if(x < sig.size):
                ends_visual[x] = 1

    '''
    if(gen_plots):
            # plot each signal
            plt.figure(i+1)
            plt.plot( seg, color='y' )
            plt.plot( env, color='g' )
            plt.plot( res_visual[ starts[i]:ends[i] ]*1, color='k' )
    '''

    # create time array in milliseconds
    timeArray = np.arange(start=0, stop=sig.size, step=1, dtype=float)
    # timeArray = timeArray / fs
    # timeArray = timeArray * 1                          # scale to seconds

    HEIGHT = np.max(sig) * (0.5)
    HEIGHT_SMOOTH = np.max(s_smooth) * (0.5)

    plt_active = False
    if (len(plt.get_fignums()) > 0):   # window(s) open
        plt_active = True

    # SIGNAL
    plt.clf()
    plt.figure(1)
    plt.subplot(311)

    plt.subplot(311)
    plt.plot(timeArray, sig, color='y')
    plt.plot(timeArray, s_smooth, color='m')
    plt.plot(timeArray, peaks_env_visual*HEIGHT*(0.75), color='b')
    plt.plot(timeArray, peaks_hfc_visual*(-HEIGHT),     color='r')
    plt.plot(timeArray, thresh_visual,                  color='g')
    plt.plot(timeArray, peaks_visual*HEIGHT,            color='k')
    plt.ylabel('Peaks')
    plt.xlabel('Time (ms)')

    plt.subplot(312)
    plt.plot(timeArray, sig, color='y')
    plt.plot(timeArray, s_smooth*2,                 color='g')
    plt.plot(timeArray, peaks_visual*HEIGHT,      color='k')
    plt.plot(timeArray, -starts_visual*HEIGHT,    color='b')
    plt.plot(timeArray, -ends_visual*HEIGHT,      color='c')
    plt.ylabel('Peaks, Starts, Ends')
    plt.xlabel('Time (ms)')

    '''
    # ONSETS
    plt.subplot(313)
    plt.plot(timeArray, s_smooth, color='y')
    plt.plot(timeArray, peaks_visual*HEIGHT_SMOOTH*0.75,      color='k')
    plt.plot(timeArray, peaks_hfc_visual*(HEIGHT_SMOOTH*0.5), color='r')
    plt.plot(timeArray, peaks_env_visual*HEIGHT_SMOOTH*0.25,  color='b')
    plt.ylabel('LPF Display')
    plt.xlabel('Time (ms)')
    '''

    plt.subplot(313)
    plt.scatter(ip_intervals_visual,
                np.full(ip_intervals_visual.size, 1, dtype=np.int),
                marker='x')

    max_dot = 0
    if(ip_intervals_visual.size):
        max_dot = np.max(ip_intervals_visual)
    plt.scatter(np.array([0, max_dot*1.2]),
                np.array([1,1]),
                marker='o')
    plt.ylabel('inter-peak intervals')
    plt.xlabel('Time (ms)')


    # plt.show(block=False) # not happy without blocking people
    if (plt_active):   # window(s) open
        plt.draw()
    else:              # no windows
        plt.show()


############################################################

# runs twice if called locally - not sure why
def main_local(rel_path):

    # globals (will make class if necessary)
    global gen_plots, res_visual
    global snds_path, data_path, file_str
    global wf, wf_plot, fs
    global sig, s_smooth
    global peaks, peaks_env, peaks_hfc
    global thresh_env, cutoff
    global starts, ends
    global FFT_BUFF, FFT_STEP, FFT_MULT
    global X, mfccs

    # ---------------------- inits --------------------------------

    # init vars
    file_str = "beat_boxing_sample.wav"
    snds_path = rel_path + "my_sounds/"
    data_path = rel_path + "data/"

    # ---------------------- get data --------------------------------
    open_new_wav_file()

    print(fs)
    print("dtype:", wf_plot.dtype)
    print("Shape:", wf_plot.shape) # shape = [frames, channels]

    convert_wf_to_signal()

    # ---------------------- onsets/peaks ----------------------------
    smooth_signal()
    get_peaks_and_segment()

    # ------------------------ feature extraction ----------------------------
    extract_features_from_signal()

    # ---------------------- cluster data --------------------------------


    # ---------------------- save data --------------------------------
    save_unclassified_data()

    # ---------------------- playback --------------------------------
    playback()


    # ---------------------- plot --------------------------------
    show_visual()

