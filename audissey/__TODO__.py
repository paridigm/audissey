"""
- will we need real-time audio processing? or can the processing be done after recording?
- will have to see the time delay --> the solution might end up being a mix!
"""

'''
TODO:

- Pruning engine for unwanted onsets (per file)
- if already saved onset file for this name, reload ... else save new onset file

-onset file data:
    active, position
    1   15543 --> an agreed onset

- Data --> if training dat? classify with it --> cluster (no matter what)

- If a value gets changed to another class
    - If in cluster w/ no class --> change to that class, change class of everyone else in that cluster
    - If in class --> change single label (remove it from training data) (it was noise)

- If add new note DO NOT modify data. It may just be for artisitic purpose (not training)
'''


'''
# attempt at smoothing with a LPF
cutoff = 25
order = 2
nyq = 0.5 * fs
normal_cutoff = cutoff / nyq
b, a = butter(order, normal_cutoff, btype='low', analog=False)
s_smooth = lfilter(b, a, s_abs)
# attempt using hilbert transform
# s_smooth = np.abs( hilbert(s_abs) )
'''

'''
plt.figure(2)
plt.plot(timeArray, s_abs, color='b')

plt.figure(3)
plt.plot(timeArray, s_smooth, color='b')
'''

'''
plt.figure(1)
plt.gcf().suptitle('Bold Title', fontsize=14, fontweight='bold')
plt.gcf().canvas.set_window_title('Test')

plt.plot(t, sig)
'''


'''
# poll and wait for user to say they are done with program
while True:
n = input("Done:")
if n.strip() == "y":
break
'''


'''
sign in
load the class --> .wav map (contains the class labels for all the sounds you want to use) (any possible # of classes)

main loop {

 record: (ENTER for record, "filename" for file, "train" to train)

 if(train)
    select which sound you wish you train (class/sound map) - enter sound name
    record: (ENTER for record, "filename" for file)
 else
 if(record)
    do recording
    save recording as new wav - store it w/ unique ID for later use

 //either way, process
 process() --> segment and extract features

 if(train)
    append data to user_data with label for desired class
 else
 if(record)
    classify using training data

 playback()

}
'''

'''
get user name (this will be for saving all data)

main loop {

 record or load a wav

 if(wav loaded)
    upload saved onsets

 if(recorded)
    save recording as new wav - store it w/ unique ID for later use

 // by now we have a file no matter what...

 process() --> segment and extract features

 ### process unknown data ###
 cluster_labels = [peaks.size] --> in audissey core
 class_labels = [peaks.size]   --> in audissey core
 for each sound
    if(sound is unclustered)
        playback sound --> requires saving wav file
        get class label from user
        label cluster of all other sounds of that cluster with the input label

 if(user data is available)
    classify w/ users data

 problem: the unclustered data isn't matched with the labeled class :o
}
'''

##########################

'''
auds.openwav("beatbox_sample3.wav")

playback_thread = threading.Thread(target=auds.playback)
playback_thread.start()

auds.process()
auds.save_unclassified_data()
auds.show_visual()
'''

# OR test it with auds.main_local("./server/")

# do same with a GUI
