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
#Declare the tables to store the data
print("Initialize tables...")
T_tables = Tables()
C = fun.getCameraMatrices()

#Get putative correspondence points
correspondences = fun.Correspondences()
y1, y2 = correspondences.getCorrByIndices(0,1)
"""
lab3.show_corresp(T_tables.images[0], T_tables.images[1], y1.T, y2.T)
plt.show()
"""
"""
    INIT1: Choose initial views I1 & I2
"""
#F, mask = cv.findFundamentalMat(y1,y2,cv.FM_RANSAC  )
#F = fun.getFFromLabCode(y1.T, y2.T)
#Pickle F
#np.save("Fmatrix", F)
F = np.load("Fmatrix.npy", allow_pickle=True)
"""
lab3.plot_eplines(F, y2.T, T_tables.images[0].shape)
plt.show()

lab3.plot_eplines(F.T, y1.T, T_tables.images[0].shape)
plt.show()
"""
"""
    INIT2: Get E = R,t from the two intial views
"""
print("Initialize SfM pipeline")
#Vårt K
E, K = fun.getEAndK(C, F)
T_tables.K = K

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
T_tables.triangulateAndAddPoints(view_index_1, view_index_2, C1, C2, y1_hom, y2_hom)
T_tables.plot()


T_tables.plotProjections(0, K, T_tables.images[0])
T_tables.plotProjections(1, K, T_tables.images[1])


data = T_tables.get3DPointsColorsAndNormals()
print(data[0].shape)
print(data[1].shape)
print(data[2].shape)

np.save("DinoVisalizationData", data)

"""
    Iterate through all images in sequence
"""

#for i in range(images.shape[0]-1):
for i in range(1,34,1):
    #Select inlier 3D points T'points
    """
        BA: Bundle Adjustment of all images so far
    """
    print("Bundle adjustment...")
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
    # Compute EpsilonBA
    # If error is large for several poses, remove 3D point.

    #err = np.empty((len(T_tables.T_points)))
    #y = T_tables.T_points[
    #print(y)
    """
    # for each 3D point in T_points
    for i,p in enumerate(T_tables.T_points) :
        # Only check if p is outlier if it exists
        # in more than n views
        n_views = len(T_tables.T_obs[p.observations_index])
        if(n_views > 3) :
            residuals = np.empty((n_views,1))
            yp = T_tables.T_obs[p.observations_index]
            for i,y in enumerate(yp) :
                C = T_tables.T_views[y.view_index]
                p_homog = np.append(p.point[:,np.newaxis], 1)
                p_proj = np.dot(C.camera_pose.GetCameraMatrix(), p_homog) # Project p with C

                residuals[i] = np.linalg.norm(y.image_coordinates - p_proj)
            # if all the projection errors are larger than a threshold
            # delete 3D point p (outlier)
            if(residuals.all() > 1.0) :
                print("Delete point not implemented. :(")
    """
T_tables.plot()
"""
    After last iteration: Bundle Adjustment if outliers were removed since last BA
"""
data = T_tables.get3DPointsColorsAndNormals()

np.save("DinoVisalizationData", data)
