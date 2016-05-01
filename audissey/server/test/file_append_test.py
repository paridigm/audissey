
import numpy as np

#os.path.exists( auds.get_snd_path() + ans + ".wav" )

f=open('generated_test_data.dat','ab')

a = (np.random.rand(1,10)*10).astype(int)
np.savetxt(f, a, fmt='%3f')

f.close()
