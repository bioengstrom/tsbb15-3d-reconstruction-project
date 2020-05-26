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
    Choose initial views I1 & I2
"""

F = fun.getFFromLabCode(y1.T, y2.T)

#np.save("Fmatrix", F)
print("Initialize SfM pipeline...")
#F = np.load("Fmatrix.npy", allow_pickle=True)

lab3.plot_eplines(F, y2.T, images[0].shape)
plt.show()

lab3.plot_eplines(F.T, y1.T, images[0].shape)
plt.show()

"""
    Get E = R,t from the two intial views
"""
#Declare the tables to store the data
print("Initialize tables...")
T_tables = Tables()

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
    Triangulate points.
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

for i in range(1,35,1):
    #Select inlier 3D points T'points
    """
        Bundle Adjustment of all images so far
    """

    preBA = np.zeros([0,3])
    # Array for 3D point positions before BA.
    # Needed for outlier deletion
    for p in T_tables.T_points.values():
        preBA = np.concatenate((preBA, [p.point]), axis=0)

    T_tables.BundleAdjustment()

    """
        Remove bad 3D points. Re-triangulate & Remove outliers
    """
    T_tables.deleteOutliers(preBA)


    """
        Choose new view C
    """

    yp2, yp3 = correspondences.getCorrByIndices(i,i+1)

    yp2_hom = fun.MakeHomogenous(K, yp2)
    yp3_hom = fun.MakeHomogenous(K, yp3)

    """
        Find 2D<->3D correspondences. Algorithm 21.2
        PnP -> R,t of new view and consensus set C
    """
    print("Adding view no:")
    print(i+1)
    A_y1, A_y2 = T_tables.addNewView(K, i+1, yp2_hom, yp3_hom, yp2, yp3)


    """
        Extend table with new row and insert image points in C. Algorithm 21.3 in IREG compendium
        For each putative correspondence that satisfies E, extend table with column
    """
    print("Adding new 3D points...")
    #Add new 3D points
    A_y1_hom = fun.MakeHomogenous(K, A_y1)
    A_y2_hom = fun.MakeHomogenous(K, A_y2)
    noOfPointsAdded = T_tables.addNewPoints(A_y1_hom, A_y2_hom, i, i+1)
    print("Added n number of 3D points:")
    print(noOfPointsAdded)

    if i%5 == 1:
        T_tables.plot()
"""
    After last iteration: Bundle Adjustment if outliers were removed since last BA
"""
T_tables.BundleAdjustment2()
T_tables.plot()
#Save data for evaluation
R, t = T_tables.getCamerasForEvaluation()
#np.save("R_eval_clean", R)
#np.save("t_eval_clean", t)
#Save data for visualization
vis_data = T_tables.get3DPointsColorsAndNormals()
np.save("DinoVisalizationData_avgnormals", vis_data)
