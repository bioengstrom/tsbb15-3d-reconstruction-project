import fun
import scipy
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

R_est = np.load('R_eval_clean.npy')
t_est = np.load('t_eval_clean.npy')

#Get K, R and t for each camera
for i in range(C.shape[1]):
    K, R[i,:,:], t[i,:] = fun.camera_resectioning(C[0,i,:,:])





#Get 3D coordinates for the cameras
coords_est = np.zeros([t.shape[0], 3])
coords_gt = np.zeros([t.shape[0], 3])
for i in range(t.shape[0]):
    coords_est[i] = -1.0*(R_est[i].T @ t_est[i])
    coords_gt[i] =  -1.0*(R[i].T @ t[i])

"""
#Plot the cameras
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for point in coords_est:
    ax.scatter(point[0], point[1], point[2], marker='o', color='blue')
for point in coords_gt:
    ax.scatter(point[0], point[1], point[2], marker='o', color='red')
plt.show()
"""

#Solve scaling problem
gt_unit = np.linalg.norm(coords_gt[0] - coords_gt[1])
est_unit = np.linalg.norm(coords_est[0] - coords_est[1])
#coords_gt = coords_gt
coords_est = (coords_est/est_unit)*gt_unit

#Solve scaling problem
gt_unit_t = np.linalg.norm(t[0] - t[1])
est_unit_t = np.linalg.norm(t_est[0] - t_est[1])
scaling = gt_unit_t/est_unit_t

"""
#Plot the cameras
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for point in coords_est:
    ax.scatter(point[0], point[1], point[2], marker='o', color='blue')
for point in coords_gt:
    ax.scatter(point[0], point[1], point[2], marker='o', color='red')
plt.show()
"""

"""
    EVAL TAJM
"""

M,m = fun.estRigidTransformation(coords_gt, coords_est)

t_mapped = np.zeros([t_est.shape[0],3])
R_mapped = np.zeros([t_est.shape[0],3,3])
for i in range(t_est.shape[0]):
    t_mapped[i] = scaling*M@t_est[i] + m
    R_mapped[i] = R_est[i]@M.T


#Get 3D coordinates for the cameras
coords_mapped = np.zeros([t.shape[0], 3])
for i in range(t.shape[0]):
    coords_mapped[i] = -1.0*(R_mapped[i].T @ t_mapped[i])

#Plot the cameras
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for point in coords_mapped:
    ax.scatter(point[0], point[1], point[2], marker='o', color='blue', alpha=0.2)
for point in coords_gt:
    ax.scatter(point[0], point[1], point[2], marker='o', color='red', alpha=0.2)
point = coords_gt[4]
ax.scatter(point[0], point[1], point[2], marker='^', color='red', alpha=1.0)
point = coords_mapped[4]
ax.scatter(point[0], point[1], point[2], marker='^', color='blue', alpha=1.0)
plt.show()


# individual errors for the camera positions
err = np.linalg.norm(coords_gt - coords_mapped, axis= 1)

print(coords_gt - coords_mapped)

#individual angular error
ang_err = np.zeros([R.shape[0]])
for i in range(R.shape[0]):
    ang_err[i] = 2 * np.arcsin( np.linalg.norm(R_mapped[i] - R[i]) / np.sqrt(8) )

#Calculate Mean Absolute Errors
err = np.mean(np.abs(err))
ang_err = np.mean(np.abs(ang_err))

#Print results
print("Position error: ")
print(err)
print("Angular error: ")
print(ang_err)
