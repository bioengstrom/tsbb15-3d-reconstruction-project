import scipy
from scipy import linalg
import scipy.cluster
import numpy as np
import fun
import cv2 as cv
import scipy.io as sio
import lab3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
from pnp import p3p
from help_classes import CameraPose, Point_3D, Observation, View
from tables import Tables
from correspondences import Correspondences

def MakeHomogenous(K, coord):
    #Normalizing corresponding points
    coord = coord.T
    coord_hom = np.zeros((3,coord.shape[1]), dtype='double')
    coord_hom[:2,:] = coord[:2,:]
    coord_hom[-1,:] = 1
    coord_hom = scipy.linalg.inv(K)@coord_hom
    return coord_hom.T

def reshapeToCamera3DPoints(x0):
    ratio = int((x0.shape[0]/16)*12)
    size = int(ratio/(3*4))
    Rktk = x0[:ratio]
    xj = x0[ratio:]
    Rktk = np.reshape(Rktk, [size, 3, 4])
    xj = np.reshape(xj, [size, 4])
    return Rktk, xj

def getImages():
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
    return images

def getCameraMatrices():
    #Load cameras
    cameras = sio.loadmat('imgdata/dino_Ps.mat')
    cameras = cameras['P']
    return cameras

"""
    Load data
"""
#Get images and camera matrices
images = getImages()
cameras = getCameraMatrices()

#Get putative correspondence points
correspondences = Correspondences()
y1, y2 = correspondences.getCorrByIndices(0,1)



"""
    INIT1: Choose initial views I1 & I2
"""
#Naive: choose 2 first img
#y1 and y2 are the consensus set C
#Fy1y2 = fun.f_matrix(images[0], images[1], y1, y2)
#np.save("Fmatrix", Fy1y2)
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
T_tables = Tables()

C = np.asarray(cameras.tolist())
K = np.zeros((C.shape[1],3,3))
R = np.zeros((C.shape[1],3,3))
t = np.zeros((C.shape[1],3))

#Get K, R and t for each camera
for i in range(C.shape[1]):
    K, R[i,:,:], t[i,:] = fun.camera_resectioning(C[0,i,:,:])

#Make the image coordinates homogenous
y1 = MakeHomogenous(K, y1p)
y2 = MakeHomogenous(K, y2p)

#Calculate essential matrix E = K.T*F*K
E = np.matmul(np.transpose(K),np.matmul(F,K))
#Get R and t from E
R, t = fun.relative_camera_pose(E, y1[:,0], y2[:,0])
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

"""
    Iterate through all images in sequence
"""
#T_tables.BundleAdjustment()



for i in range(images.shape[0]-1):
    #Select inlier 3D points T'points

    """
        BA: Bundle Adjustment of all images so far
    """


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
    lab3.show_corresp(images[i], images[i+1], yp2.T, yp3.T)
    plt.show()
    yp2_hom = MakeHomogenous(K, yp2)
    yp3_hom = MakeHomogenous(K, yp3)

    """
        EXT2: Find 2D<->3D correspondences. Algorithm 21.2
        EXT3: PnP -> R,t of new view and consensus set C
    """
    T_tables.addNewView(K, images[1], images[2], 2, yp2_hom[:100], yp3_hom[:100], yp2[:100], yp3[:100])
    T_tables.plot()
    """
        EXT4: Extend table with new row and insert image points in C. Algorithm 21.3
    """
    #Add new 3D points

    """
        EXT5: For each putative correspondence that satisfies E, extend table with column
    """

    """
        WASH2: Check elements not in C and remove either 3D points or observation
    """

"""
    After last iteration: Bundle Adjustment if outliers were removed since last BA
"""
