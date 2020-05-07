import scipy
from scipy import linalg
import scipy.cluster
import numpy as np
import fun
import cv2 as cv
import scipy.io as sio
import lab3
import matplotlib.pyplot as plt

class CameraPose:
    def __init__(self, R = np.identity(3), t = np.array([0.0, 0.0, 0.0])):
        self.R = R
        self.t = t

"""
    Load data
"""
no_of_images = 36
img1 = cv.imread("../images/viff.000.ppm", cv.IMREAD_COLOR)
img1 = np.asarray(img1)


images = np.zeros([no_of_images, img1.shape[0],img1.shape[1],img1.shape[2]], dtype='int' )

for i in range(no_of_images):
    no = str(i)
    if i < 10:
        no = '0' + no
    #img1 = np.asarray(cv.cvtColor(images[0], cv.COLOR_BGR2GRAY)) # Grayscale
    #img2 = np.asarray(cv.cvtColor(images[1], cv.COLOR_BGR2GRAY))
    images[i] = np.asarray(cv.imread("../images/viff.0" + no + ".ppm", cv.IMREAD_COLOR))

Dino_36C = sio.loadmat('imgdata/dino_Ps.mat')
Dino_36C = Dino_36C['P']

"""
    INIT1: Choose initial views I1 & I2
"""
#Naive: choose 2 first img
#y1 and y2 are the consensus set C
#Fy1y2 = fun.f_matrix(images[0], images[1])
#np.save("Fmatrix", Fy1y2)
Fy1y2 = np.load("Fmatrix.npy", allow_pickle=True)

F = Fy1y2[0]
y1 = Fy1y2[1]
y2 = Fy1y2[2]

"""
    INIT2: Get E = R,t from the two intial views
"""

T_points = []
T_views = []
T_obs = []

C = np.asarray(Dino_36C.tolist())
K = np.zeros((C.shape[1],3,3))
R = np.zeros((C.shape[1],3,3))
t = np.zeros((C.shape[1],3))

for i in range(C.shape[1]):
    K, R[i,:,:], t[i,:] = fun.camera_resectioning(C[0,i,:,:])

#Calculate E = K.T*F*K
E = np.matmul(np.transpose(K),np.matmul(F,K))
R, t = fun.relative_camera_pose(E, y1[:,0], y2[:,0])
C1 = CameraPose()
C2 = CameraPose(R,t)

for i in range(y1.shape[1]):
    #Triangulate
    



"""
    INIT3: Triangulate points.
"""

"""
    Iterate through all images in sequence
"""

"""
    BA: Bundle Adjustment of all images so far
"""
"""
    WASH1: Remove bad 3D points
"""

"""
    EXT1: Choose new view C
"""

"""
    EXT2: Find 2D<->3D correspondences
"""

"""
    EXT3: PnP -> R,t of new view and consensus set C
"""

"""
    EXT4: Extend table with new row and insert image points in C
"""

"""
    EXT5: For each putative correspondence that satisfies E, extend table with column
"""

"""
    WASH2: Check elements not in C and remove either 3D points or observation
"""

"""
    After last iteration: Bundle Adjustment if outliers were removed since last BA
"""
