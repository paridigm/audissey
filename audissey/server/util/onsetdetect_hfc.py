import scipy
import numpy as np
import scipy.signal as signal

# References:
# Bello, Daudet, Abdallah, Duxbury, Davies, Sandler: A Tutorial on Onset Detection in Music Signals

#TODO add plotting in here to see what is going on

def detect_onsets(sig, fftwin = 512):
    spectrogram = generate_spectrogram(sig, fftwin)
    # hfcs = np.array(map(get_hfc, spectrogram))  # map() applies function to all elems
    
    # apply hfc function to every row of the spectrogram
    hfcs = np.apply_along_axis(get_hfc, 1, spectrogram)
    
    hfcs /= max(hfcs) # this line works with a numpy array, not a map object :(
    hfcs = filter_hfcs(hfcs) #filter_hfcs runs up and down peaks
    
    # multiplies by the window size (512) to scale it back to time scale
    peak_indices = np.array([i for i, x in enumerate(hfcs) if x > 0]) * fftwin
    1 == 1
    return peak_indices



def get_hfc(spectrum):
    hfc = np.sum(np.power(spectrum, 2) * np.arange(1, len(spectrum) + 1)) 
    return hfc # scalar, NOTE: the higer frequencies are multiplied by larger numbers



def generate_spectrogram(audio, window_size):
    # spectrogram = [0] * (1 + int((len(audio) / window_size)))
    spectrogram = np.zeros(( 1 + int((len(audio) / window_size)) , window_size / 2)) 
    
    for t in range(0, len(audio), window_size):
        if(len(audio) - t < window_size):   #added this case mod to ignore end
            break
        else:
            actual_window_size = min(window_size, len(audio) - t)
            
        windowed_signal = audio[t:(t + window_size)] * np.hanning(actual_window_size)
        spectrum = abs(scipy.fft(windowed_signal))
        spectrum = spectrum[0:len(spectrum) / 2]  # cutting the spectrum in half! 
        spectrogram[int(t / window_size)] = spectrum

    return spectrogram


"""
Apply a hamming window to smooth? and then climb_hill()
"""
def filter_hfcs(hfcs):
    fir = signal.firwin(11, 1.0 / 8, window = "hamming") # FIR window filter (smoothing?)
    filtered = np.convolve(hfcs, fir, mode="same")
    filtered = climb_hills(filtered)  # climbing hills is complaining :(
    1==1
    return filtered


# I believe this function will create an array  
# of zeros and 1 spikes that represent onsets --> yes
def climb_hills(vector):
    moving_points = list( range(len(vector)) ) # FIX: had to turn this into a list
    temp = np.array(moving_points)
    stable_points = []

    # got to understand this loop...
    while len(moving_points) > 0:
        for (i, x) in reversed(list(enumerate(moving_points))):

            # DEBUG PRINT
            # print(i,",",x)

            def stable():                       #adds a peak (stable point)
                stable_points.append(x)         # what is a moving_point?
                del moving_points[i]

            if x > 0 and x < len(vector) - 1:   #inside input vector
                if vector[x] >= vector[x - 1] and vector[x] >= vector[x + 1]:  #peak
                    stable()
                elif vector[x] < vector[x - 1]:                                #rising
                    moving_points[i] -= 1
                else:
                    moving_points[i] += 1                                      #falling

            elif x == 0:                        #iterating at beginning
                if vector[x] >= vector[x + 1]:
                    stable()                    #assume that a fall at the beginning is a peak
                else:
                    moving_points[i] += 1

            else:                               #iterating at end?
                if vector[x] >= vector[x - 1]:
                    stable()                    #assume that a rise at the end is a peak also
                else:
                    moving_points[i] -= 1

    filtered = [0] * len(vector)
    for x in set(stable_points):
        filtered[x] = vector[x]
    
    return filtered
