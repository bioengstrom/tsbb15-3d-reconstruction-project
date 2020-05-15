import fun
import scipy
import scipy.io as sio
import numpy as np

def getCameraMatrices():
    #Load cameras
    cameras = sio.loadmat('imgdata/dino_Ps.mat')
    cameras = cameras['P']
    return cameras

#Get images and camera matrices
#images = main.getImages()
cameras = getCameraMatrices()

C = np.asarray(cameras.tolist())
K = np.zeros((C.shape[1],3,3))
R = np.zeros((C.shape[1],3,3))
t = np.zeros((C.shape[1],3))

#Get K, R and t for each camera
for i in range(C.shape[1]):
    K, R[i,:,:], t[i,:] = fun.camera_resectioning(C[0,i,:,:])

R_est = np.load('estimated_R.npy')
t_est = np.load('estimated_t.npy')

"""
    EVAL TAJM
"""
t1 = t[0]
M,m = fun.estRigidTransformation(t1, t_est)

t_mapped = M@t_est + m
R_mapped = R_est@M.T

# individual errors for the camera positions
err = np.linalg.norm(t1 - t_mapped)
print("Position error: ")
print(err)
#individual angular error
ang_err = 2 * np.arcsin( np.linalg.norm(R_mapped - R[0]) / np.sqrt(8) )
print("Angular error: ")
print(ang_err)
