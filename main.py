import fun
import scipy.io as sio
import numpy as np
import cv2 as cv
import pickle

print("Hello World")

img1 = cv.imread("images/viff.000.ppm", cv.IMREAD_COLOR)
img2 = cv.imread("images/viff.001.ppm", cv.IMREAD_COLOR)
img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

#F, *_ = fun.f_matrix(img1, img2)

F = np.load('F_matrix.npy')

Dino_36C = sio.loadmat('imgdata\dino_Ps.mat')

Dino_36C = Dino_36C['P']
#print(Dino_36C)
C = np.asarray(Dino_36C.tolist())

K = np.zeros((C.shape[1],3,3))
R = np.zeros((C.shape[1],3,3))
t = np.zeros((C.shape[1],3))

for i in range(C.shape[1]):
    K[i,:,:], R[i,:,:], t[i,:] = fun.camera_resectioning(C[0,i,:,:])
#print(K)
#print(R)
#print(t)
