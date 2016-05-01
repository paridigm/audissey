import context  # for spyder... since the project dir isn't known
import os.path
import server as auds
import threading
from pyo import *

# pyo had to be instanitated in main
pyo_server = Server().boot()

# initialize server by giving it its relative path
auds.set_default_path("./server/", pyo_server)

# work around for python3.4 in sdyper
try:
    raw_input
except:
    raw_input = input

# program vars
done = False
current_file = None

# get user
user = raw_input("Enter user name : ")
user = "default" if(user == '') else user
auds.set_user(user)


# command line gui for now
while(not done):

    # record new wav or open file
    ans = raw_input("Press enter to record: ")
    ans = "default" if(ans == '') else ans

    class_training = None  # this will be the class label under training if training

    # exit program condition
    if(ans == "exit"):
        done = True
        break
    
    # if training --> set training class and ask again
    if(ans == "t" or ans == "train" or ans == "tr"):
        class_training = raw_input("Enter class to train: ")
        ans = raw_input("Press enter to record: ")
        ans = "default" if(ans == '') else ans

    # if playback, just playback and ask again..
    if(ans == "r" or ans == "replay" or ans =="play" or ans == "p"):

        # playback the current wav
        auds.openwav(current_file + ".wav")     # reopening wav was fix for default (but shudnt need re-opening... idk)
        playback_thread = threading.Thread(target=auds.playback)
        playback_thread.start()

        while(playback_thread.isAlive()):
            pass

        # playback the transcribed audio after actual playback (blocking function)
        auds.playback_with_pyo()

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
        auds.save_user_training_data(class_training)
    else:
        #jamming - classify by uploading the training data
        pass

    while( playback_thread.isAlive() ):
            pass

    # playback the transcribed audio after actual playback
    auds.playback_with_pyo()

    # start a plot thread (ok to recreate... only the first thread will actually stay alive forever)
    vis_thread = threading.Thread(target=auds.show_visual)
    vis_thread.start()


#end while, main loop
