import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav

from audissey.server.util.features import logfbank
from audissey.server.util.features import mfcc

sounds_dir = "../my_sounds/"

(rate,sig) = wav.read(sounds_dir + "file.wav")
sig = sig[:,0]

mfcc_feat = mfcc(sig, rate, appendEnergy = True)
fbank_feat = logfbank(sig,rate)

print(mfcc_feat[:,:])


# ----- plot -----
derired_range = range(0, 5)

plt.figure(1)
for i in derired_range:
    plt.plot(mfcc_feat[:,i], label="MFCC"+str(i))
plt.title("MFCC movement")
plt.xlabel("time (sample)")
plt.ylabel("MFCC value")
plt.legend()
plt.show()

print("__name__:" +  __name__)
print(np.shape(mfcc_feat))
