import scipy.io as sio
import scipy
import numpy as np

dino = sio.loadmat('imgdata\dino_Ps.mat')
dino = dino['P']
C = np.asarray(dino[0,:]) # cameras, (R,t)
print(C)
