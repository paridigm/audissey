import os.path
import server as auds
import threading

auds.set_default_path("./server/")

done = False
current_file = None

user = raw_input("Enter user name : ")
user = "default" if(user == '') else user


# command line gui for now
while(not done):

    # record new wav or open file
    ans = raw_input("Press enter to record: ")
    ans = "default" if(ans == '') else ans

    class_training = None  # this will be the class label under training if training

    # if training --> set training class and ask again
    if(ans == "t" or ans == "train" or ans == "tr"):
        class_training = raw_input("Enter class to train: ")
        ans = raw_input("Press enter to record: ")
        ans = "default" if(ans == '') else ans

    # if playback, just playback and ask again..
    if(ans == "r" or ans == "replay" or ans =="play" or ans == "p"):

        auds.openwav(current_file + ".wav")    # fix for default (but shudnt need re-opening...)
        playback_thread = threading.Thread(target=auds.playback)
        playback_thread.start()

        while(playback_thread.isAlive()):  # printing (same duration every time)
            #print("... playing back")      # suggests that thread is at least alive
            pass
        continue

    # if file doesn't exist or recording new wav...
    if( ans == "default" or not os.path.exists( auds.get_snd_path() + ans + ".wav" ) ):

        # record in a new thread
        rec_thread = threading.Thread(target=auds.record, args=(ans,))
        rec_thread.start()

        # stop recording
        raw_input("Press enter to stop")
        auds.record_stop()

        # wait for recording to finish up saving wav file
        while( rec_thread.isAlive() ):
            pass

    # retrieve wav file (recorded or loaded)
    auds.openwav(ans + ".wav")
    current_file = ans
    print("saved")

    # playback
    playback_thread = threading.Thread(target=auds.playback)
    playback_thread.start()
    print("playback started")

    # segment and extract features
    auds.process()


    # save data
    auds.save_unclassified_data()
    print("***data saved ***")

    if(class_training != None):
        #training
        auds.save_user_training_data(class_training, user)
    else:
        #jamming - classify by uploading the training data
        pass

    # playback the transcribed audio


    # start a plot thread (ok, only the first thread will actually stay alive forever)
    vis_thread = threading.Thread(target=auds.show_visual)
    vis_thread.start()

#end while, main loop


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

auds.openwav("beatbox_sample3.wav")

playback_thread = threading.Thread(target=auds.playback)
playback_thread.start()

auds.process()
auds.save_unclassified_data()
auds.show_visual()

# OR test it with auds.main_local("./server/")


# do same with a GUI
