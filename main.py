import scipy.io as sio
import scipy
from scipy import linalg
import scipy.cluster
import numpy as np
import fun
"""
    Load data
"""
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
#print(R)
#print(t)
M = np.ones((3,3))
U, S, V = fun.specSVD(M)

W = fun.relative_camera_pose(M)

"""
    INIT1: Choose initial views I1 & I2
"""

"""
    INIT2: Get E = R,t from the two intial views
"""

"""
    INIT3: Triangulate points. Set l = 2
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
