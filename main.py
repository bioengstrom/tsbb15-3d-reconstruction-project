import scipy
from scipy import linalg
import scipy.cluster
import numpy as np
import fun
import cv2 as cv
import lab3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
from pnp import p3p
from help_classes import CameraPose, Point_3D, Observation, View
from tables import Tables
from correspondences import Correspondences
import fun as fun

"""
    Load data
"""
#Get images and camera matrices
print("Get images, camera matrices and correspondences...")
images = fun.getImages()
C = fun.getCameraMatrices()

#Get putative correspondence points
correspondences = fun.Correspondences()
y1, y2 = correspondences.getCorrByIndices(0,1)
#y2, y3 = correspondences.getNoOfCorrespondences(0,2)
#print(y2.shape())

"""
    INIT1: Choose initial views I1 & I2
"""
#Naive: choose 2 first img
#y1 and y2 are the consensus set C
#Fy1y2 = fun.f_matrix(images[0], images[1], y1, y2)
#np.save("Fmatrix", Fy1y2)
print("Initialize SfM pipeline...")
Fy1y2 = np.load("Fmatrix.npy", allow_pickle=True)

F = Fy1y2[0]
y1p = Fy1y2[1].T
y2p = Fy1y2[2].T

#Show the image with interest points

plt.imshow(images[0])
plt.scatter(y1p[:,0], y1p[:,1], color='orange')
plt.show()

"""
    INIT2: Get E = R,t from the two intial views
"""
#Declare the tables to store the data
print("Initialize tables...")
T_tables = Tables()

E, K = fun.getEAndK(C, F)
print("Vår K")
print(K)

K, R, t = fun.decomposeP(C[0,0,:,:])
print("Inte vår K")
print(K)

E = np.matmul(np.transpose(K),np.matmul(F,K))

#Get R and t from E
R, t = fun.relative_camera_pose(E, y1[0], y2[0])
"""
print(K.shape)
print(R.shape)
print(t.shape)
"""

#Make the image coordinates homogenous
y1 = fun.MakeHomogenous(K, y1p)
y2 = fun.MakeHomogenous(K, y2p)
#print(y1)

#Get first two camera poses
C1 = CameraPose()
C2 = CameraPose(R,t)

#Add the first two Views to the tables.
#Index 0 and C1 for image 1 and first camera pose. Same for second image and C2
view_index_1 = T_tables.addView(0,C1)
view_index_2 = T_tables.addView(1,C2)

"""
    INIT3: Triangulate points.
"""

for i in range(y1.shape[0]):
    #Triangulate points and add to tables
    new_3D_point = lab3.triangulate_optimal(C1.GetCameraMatrix(), C2.GetCameraMatrix(), y1[i], y2[i])
    point_index = T_tables.addPoint(new_3D_point)
    T_tables.addObs(y1[i], view_index_1, point_index)
    T_tables.addObs(y2[i], view_index_2, point_index)


T_tables.plot()


mask = T_tables.sparsity_mask()
T_tables.BundleAdjustment2()
T_tables.plot()

"""
    Iterate through all images in sequence
"""

#for i in range(images.shape[0]-1):
for i in range(1,5,1):
    #Select inlier 3D points T'points
    """
        BA: Bundle Adjustment of all images so far
    """
    #print("Bundle adjustment...")
    #T_tables.BundleAdjustment()
    T_tables.BundleAdjustment2()

    """
        WASH1: Remove bad 3D points. Re-triangulate & Remove outliers
    """
    #For each 3D points in T points that is not in T'points
    #Get corresponding observations y1 and y2 from T_obs and their camera poses C1 and C2 from T_views.
    #Triangulate x from y1, y2, C1 and C2. Update 3D point in T_points
    #Remove potential outliers from T_points after bundle adjustment

    """
        EXT1: Choose new view C
    """

    yp2, yp3 = correspondences.getCorrByIndices(i,i+1)
    #print(yp2.shape)
    lab3.show_corresp(images[i], images[i+1], yp2.T, yp3.T)
    plt.show()
    yp2_hom = fun.MakeHomogenous(K, yp2)
    yp3_hom = fun.MakeHomogenous(K, yp3)


    """
        EXT2: Find 2D<->3D correspondences. Algorithm 21.2
        EXT3: PnP -> R,t of new view and consensus set C
    """

    A_y1, A_y2 = T_tables.addNewView(K, images[1], images[2], 2, yp2_hom, yp3_hom, yp2, yp3)
    T_tables.plot()
    plt.show()


    """
        EXT4: Extend table with new row and insert image points in C. Algorithm 21.3
        EXT5: For each putative correspondence that satisfies E, extend table with column
    """
    #Add new 3D points
    T_tables.addNewPoints(A_y1, A_y2, i, i+1)



    """
        WASH2: Check elements not in C and remove either 3D points or observation
    """
    # Compute EpsilonBA
    # If error is large for several poses, remove 3D point.

    Rktk = np.empty((len(T_tables.T_views), 3,4))
    xj = np.empty((len(T_tables.T_points),3))
    yij = np.empty((len(T_tables.T_obs),3))

    for i,o in enumerate(T_tables.T_views):
        Rktk[i] = o.camera_pose.GetCameraMatrix()
    for i,o in enumerate(T_tables.T_points):
        xj[i] = o.point
    for i,o in enumerate(T_tables.T_obs):
        yij[i] = o.image_coordinates

    x0 = np.hstack([Rktk.ravel(), xj.ravel()])

    r = fun.EpsilonBA(x0, yij[0], yij[1], T_tables)
    print(r.shape)

"""
    After last iteration: Bundle Adjustment if outliers were removed since last BA
"""
