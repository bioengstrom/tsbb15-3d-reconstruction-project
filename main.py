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

lab3.show_corresp(images[0], images[1], y1.T, y2.T)
plt.show()

"""
    INIT1: Choose initial views I1 & I2
"""

"""
#Naive: choose 2 first img
#y1 and y2 are the consensus set C
Fy1y2 = fun.f_matrix(y1, y2)
np.save("Fmatrix", Fy1y2)
print("Initialize SfM pipeline...")
#Fy1y2 = np.load("Fmatrix.npy", allow_pickle=True)

F = Fy1y2[0]
#y1p = Fy1y2[1].T
#y2p = Fy1y2[2].T

"""

"""
Den andra gruppens F
F = np.array([[1.20899205e-08,  1.87079612e-07, 4.39313278e-04 ], [2.41307322e-07, -8.82954119e-09 , 7.81238182e-03 ], [ 5.75003645e-05 , -7.97164482e-03 , -1.74092676e-01 ]])
"""
#F, mask = cv.findFundamentalMat(y1,y2,cv.FM_RANSAC  )
#F = fun.getFFromLabCode(y1.T, y2.T)

#np.save("Fmatrix", F)
print("Initialize SfM pipeline...")
F = np.load("Fmatrix.npy", allow_pickle=True)

lab3.plot_eplines(F, y2.T, images[0].shape)
plt.show()

lab3.plot_eplines(F.T, y1.T, images[0].shape)
plt.show()

"""
    INIT2: Get E = R,t from the two intial views
"""
#Declare the tables to store the data
print("Initialize tables...")
T_tables = Tables()

#Vårt K
E, K = fun.getEAndK(C, F)

#Make the image coordinates homogenous
y1_hom = fun.MakeHomogenous(K, y1)
y2_hom = fun.MakeHomogenous(K, y2)

#Get R and t from E
R, t = fun.relative_camera_pose(E, y1_hom[0,:2].T, y2_hom[0,:2].T) #Inpute is C-normalized coordinates

#Get first two camera poses
C1 = CameraPose() #No rotation and translation
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
    new_3D_point = lab3.triangulate_optimal(C1.GetCameraMatrix(), C2.GetCameraMatrix(), y1_hom[i,:2], y2_hom[i,:2])
    point_index = T_tables.addPoint(new_3D_point)
    T_tables.addObs(y1_hom[i], view_index_1, point_index)
    T_tables.addObs(y2_hom[i], view_index_2, point_index)

T_tables.plotProjections(0, K, images[0])

"""
    Iterate through all images in sequence
"""
# Array for re-projection errors. (36, number of points)
#proj_err = np.empty(( ,images.shape[0]))
#print(proj_err.shape)

#for i in range(images.shape[0]-1):
for i in range(1,34,1):
    #Select inlier 3D points T'points
    """
        BA: Bundle Adjustment of all images so far
    """
    preBA = np.zeros([0,3])

    for p in T_tables.T_points.values():
        preBA = np.concatenate((preBA, [p.point]), axis=0)
    #print("Bundle adjustment...")
    #T_tables.plot()
    T_tables.BundleAdjustment2()
    #T_tables.plotProjections(i, K, images[i])
    #T_tables.plot()
    """
        WASH1: Remove bad 3D points. Re-triangulate & Remove outliers
    """
    #For each 3D points in T points that is not in T'points
    #Get corresponding observations y1 and y2 from T_obs and their camera poses C1 and C2 from T_views.
    #Triangulate x from y1, y2, C1 and C2. Update 3D point in T_points
    #Remove potential outliers from T_points after bundle adjustment

    # # Check for large changes in position before and after BA
    # delete_key = np.empty((0,1), dtype='int')
    # dist = np.empty((len(T_tables.T_points)))
    # for j,p in enumerate(T_tables.T_points.values()):
    #     dist[j] = np.linalg.norm(p.point - preBA[j])
    # #print(dist)
    # #print(dist.shape)
    # dist_idx = np.argwhere(dist > 1)
    # #print(dist_idx)
    # # Delete 3D point where larger than x
    # for j in range(dist_idx.shape[0]) :
    #     #print(dist_idx[j,0])
    #     #T_tables.deletePoint2(dist_idx[j,0])
    #     delete_key = np.append(delete_key, key)
    #     #print("Deleting point..."§)
    #     #dist_idx[:] = dist_idx[:] - 1
    #     #print(dist_idx)
    
    # Check for large reprojection errors
    delete_key = np.empty((0,1), dtype='int')
    for key in T_tables.T_points :
        #if(j == len(T_tables.T_points)-1) :
        #    break
        # Only check if p is outlier if it exists
        # in more than n views
        residuals = np.empty((T_tables.T_points[key].observations_index.shape[0],1))
        #yp = T_tables.T_obs[p.observations_index]
        for k,o in enumerate(T_tables.T_points[key].observations_index) :
            y = T_tables.T_obs[o]
            C = T_tables.T_views[y.view_index]
            p_homog = np.append(T_tables.T_points[key].point[:,np.newaxis], 1)

            P1 = C.camera_pose.GetCameraMatrix()
            p_proj = P1 @ p_homog
            p_proj = p_proj / p_proj[-1]
            #p_proj = np.dot(C.camera_pose.GetCameraMatrix(), p_homog) # Project p with C
            residuals[k] = np.linalg.norm(y.image_coordinates - p_proj)
        # if all the projection errors are larger than a threshold
        # delete 3D point p (outlier)
        r_bool = residuals > 0.01

        if(np.count_nonzero(r_bool) > 2) :
            print("Deleting points...")
            delete_key = np.append(delete_key, key)
            #T_tables.deletePoint2(j)
    for n in delete_key :
        T_tables.deletePoint2(n)
    
    """
        EXT1: Choose new view C
    """

    yp2, yp3 = correspondences.getCorrByIndices(i,i+1)
    #print(yp2.shape)
    #lab3.show_corresp(images[i], images[i+1], yp2.T, yp3.T)
    #plt.show()
    yp2_hom = fun.MakeHomogenous(K, yp2)
    yp3_hom = fun.MakeHomogenous(K, yp3)

    """
        EXT2: Find 2D<->3D correspondences. Algorithm 21.2
        EXT3: PnP -> R,t of new view and consensus set C
    """
    print("Adding view no:")
    print(i+1)
    A_y1, A_y2 = T_tables.addNewView(K, i+1, yp2_hom, yp3_hom, yp2, yp3)
    #T_tables.plotProjections(i+1, K, images[i+1])
    #T_tables.plot()


    """
        EXT4: Extend table with new row and insert image points in C. Algorithm 21.3
        EXT5: For each putative correspondence that satisfies E, extend table with column
    """
    print("Adding new 3D points...")
    #Add new 3D points
    A_y1_hom = fun.MakeHomogenous(K, A_y1)
    A_y2_hom = fun.MakeHomogenous(K, A_y2)
    noOfPointsAdded = T_tables.addNewPoints(A_y1_hom, A_y2_hom, i, i+1)
    print("Added n number of 3D points:")
    print(noOfPointsAdded)



    """
        WASH2: Check elements not in C and remove either 3D points or observation
    """
    T_tables.plot()
"""
    After last iteration: Bundle Adjustment if outliers were removed since last BA
"""
