from pyo import *
import time
import threading

# pyo backend vars
pyo_s = None        # pyo server
pyo_samples = None  # audio samples
pyo_calls = None    # callback functions for playback


def set_pyo_server(pyo_server_object):
    global pyo_s
    pyo_s = pyo_server_object


def load_samples_from(samples_list):
    global pyo_samples
    pyo_samples = [None] * len(samples_list)
    for i in range(0, len(pyo_samples)):
        pyo_samples[i] = SfPlayer(samples_list[i])


def play_song(notes, onsets):
    global pyo_calls, pyo_s

    # starting and stopping the pyo server in case it uses pyaudio
    pyo_s.start()

    pyo_calls = [None] * len(onsets)
    for i in range(0,len(onsets)):
        pyo_calls[i] = CallAfter(play_snd, onsets[i], notes[i])

    # BLOCKING CODE - pyo didn't like being called in the background
    time.sleep(onsets[len(onsets)-1] + 0.5)

    pyo_s.stop()


def play_snd(index):
    print(index)
    pyo_samples[index].out()


# local main function for testing
def main():

    # song notes
    my_samples = ["./samples/bass_sample.wav",
                  "./samples/hihat_sample.wav",
                  "./samples/snare_sample.wav"]

    # give the pyo side some information
    load_samples_from(my_samples)

    # my song
    my_onsets = [0,    0.5,    1,      1.5,    2,      2.25,   2.5,    3]
    my_notes =  [0,    1,      2,      1,      0,      0,      1,      2]

    # play my song
    play_song(my_notes, my_onsets)

    # test how many times I can loop it and not fill up memory
    my_notes = [0,0,0,0,2,0,2]
    my_onsets = [0,0.125,0.25,0.375,0.5,1.25,1.5]
    for j in range(1, 4):
        my_notes.extend([0,0,0,0,2,0,2])
        my_onsets.extend([x+2*j for x in [0,0.125,0.25,0.375,0.5,1.25,1.5]])

    # play big song
    play_song(my_notes, my_onsets)

    # pyo doesn't like doing stuff in another thread... will have to block in the function
    #threading.Thread(target=play_song, args=(my_notes, my_onsets, ))

