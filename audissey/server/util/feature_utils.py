#####
#
# Contains functions that assist with
# extracting features from audio segments.
#
#####

import numpy as np

from audissey.server.util.features import mfcc

""" specify desired feature function to use """
def desired_feature_function(sig, env, fs, fft_size, fft_step_size):
    return get_w_ave_mfccs_using_ave_w(sig, env, fs, fft_size, fft_step_size)



""" gets weighted average of MFCC's using averaged envelope amplitude as weights """
def get_w_ave_mfccs_using_ave_w(sig, env, fs, fft_size, fft_step_size):
    
    ## average MFCC during onset (ceplifter enabled --> does this matter?) ##
    mfccs = mfcc(sig, fs, winlen = float(fft_size)/fs,
                         winstep = float(fft_step_size)/fs,
                         appendEnergy = True)
                         # appendEnergy = False)
    
    # get average weight values (with end protection)
    i_w = np.arange(0, fft_step_size*np.shape(mfccs)[0], fft_step_size).astype(int)
    i_w_ends = np.zeros( np.shape(i_w)[0] ).astype(int)
    for i in range( np.shape(i_w)[0] ):
        if( i_w[i] + fft_size < np.shape(env)[0] ):
            i_w_ends[i] = i_w[i] + fft_size - 1
        else:
            i_w_ends[i] = np.shape(env)[0] - 1
    weights = (env[i_w] + env[i_w_ends]) / 2
    
    # multiply each row of MFCC by the envelope (importance scaling)    
    mfccs_w = (mfccs.T*weights).T
    
    # sum each (weighted) mfcc 
    sum_mfccs = np.sum(mfccs_w, axis=0)
    
    # divide by sum of weights
    w_ave_mfccs = sum_mfccs / np.sum(weights)

    return w_ave_mfccs



""" extract desired number of features from before and after peak with desired feature function """
def extract_features(prev, post, env_prev, env_post, fs, fft_size, fft_step_size, mfcc_range=(0, 12)):

    prev_mfccs =  desired_feature_function(prev, env_prev, fs, fft_size, fft_step_size)
    post_mfccs =  desired_feature_function(post, env_post, fs, fft_size, fft_step_size)

    prev_mfccs = prev_mfccs[ mfcc_range[0] : (mfcc_range[1]+1) ]
    post_mfccs = post_mfccs[ mfcc_range[0] : (mfcc_range[1]+1) ]    
    
    return np.append(prev_mfccs, post_mfccs)

    # return prev_mfccs




'''
# NOTE: only the first mfcc is different if I normalize each segment       
        # I think this is happening since the first coeff is ~= total energy
        # --> I will ignore the 0th coeff  

## average MFCC during onset (ceplifter enabled --> does this matter?) ##
mfccs = mfcc(prev, fs, winlen = float(FFT_BUFF)/fs,
                     winstep = float(FFT_STEP)/fs,
                     appendEnergy = True)
                     # appendEnergy = False)

# multiply each row of MFCC by the envelope (importance scaling)
i_w = np.arange(0, FFT_STEP*np.shape(mfccs)[0], FFT_STEP).astype(int)
weights = (env_prev[i_w] + env_prev[i_w + FFT_BUFF - 1]) / 2
mfccs_w = (mfccs.T*weights).T

# sum each mfcc ()
sum_mfccs = np.sum(mfccs_w, axis=0)

# divide by sum of weights
w_ave_mfccs = sum_mfccs / np.sum(weights)
'''
