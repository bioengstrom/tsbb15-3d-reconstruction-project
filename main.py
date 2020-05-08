import scipy
from scipy import linalg
import scipy.cluster
import numpy as np
import fun
import scipy.io as sio
import numpy as np
import cv2 as cv
import scipy.io as sio
import lab3
print("Hello World")


img1 = cv.imread("images/viff.000.ppm", cv.IMREAD_COLOR)
img2 = cv.imread("images/viff.001.ppm", cv.IMREAD_COLOR)
img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

#F, y1, y2 = fun.f_matrix(img1, img2)
#np.save('F_matrix', F)
#np.save('y1', y1)
#np.save('y2', y2)
y1 = np.load('y1.npy')
y2 = np.load('y2.npy')
F = np.load('F_matrix.npy')

Dino_36C = sio.loadmat('imgdata\dino_Ps.mat')

Dino_36C = Dino_36C['P']
#print(Dino_36C)
C = np.asarray(Dino_36C.tolist())
K = np.zeros((C.shape[1],3,3))
R = np.zeros((C.shape[1],3,3))
t = np.zeros((C.shape[1],3))

for i in range(C.shape[1]):
    K, R[i,:,:], t[i,:] = fun.camera_resectioning(C[0,i,:,:])

#E = K.T*F*K
E = np.matmul(np.transpose(K),np.matmul(F,K))
#The second camera. This is always [I | 0]
C1, C2 = lab3.fmatrix_cameras(E)

R_est, t_est = fun.relative_camera_pose(E, C1, C2, y1[:,0], y2[:,0])
print(t_est)
t1 = t[0]
R1 = R[0]
print(R1.shape)

# map t_est to t, t = t_est + m
#translation
m = t1 - t_est
M = np.outer((t1 - m),t_est)

print(t1)
t_mapped = np.matmul(M,t_est) + m
print(t_mapped)
R_mapped = R_est@np.transpose(M)
print(R1)
print(R_mapped)

err = np.linalg.norm(t1 - t_mapped)
print(err)
