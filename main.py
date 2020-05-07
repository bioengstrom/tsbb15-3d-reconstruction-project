import scipy.io as sio
import scipy
from scipy import linalg
import scipy.cluster
import numpy as np
import fun

Dino_36C = sio.loadmat('imgdata\dino_Ps.mat')

Dino_36C = Dino_36C['P']
#print(Dino_36C)
C = np.asarray(Dino_36C.tolist())
K = np.zeros((C.shape[1],3,3))
R = np.zeros((C.shape[1],3,3))
t = np.zeros((C.shape[1],3))

for i in range(C.shape[1]):
    K[i,:,:], R[i,:,:], t[i,:] = fun.camera_resectioning(C[0,i,:,:])

#E = K.T*F*K
#E = np.matmul(np.transpose(K),np.matmul(F,K))
#print(K)

M = np.ones((3,3))
U, S, V = fun.specSVD(M)

R, t = fun.relative_camera_pose(M)
print(R)
print(t)