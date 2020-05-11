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

class CameraPose:
    def __init__(self, R = np.identity(3), t = np.array([0.0, 0.0, 0.0])):
        self.R = R
        self.t = t

    def __str__(self):
        the_print = "R: "
        the_print += str(self.R)
        the_print += " t: "
        the_print += str(self.t)

        return the_print

    def GetCameraMatrix(self):
        C1t1R1 = np.zeros((3,4), dtype='double')
        C1t1R1[:,-1] = self.t
        C1t1R1[:3,:3] = self.R

        return C1t1R1

class Point_3D:
    def __init__(self, point):
        self.point = point
        self.observations_index = np.array([], dtype = 'int')

    def __str__(self):
        the_print = "3D point: "
        the_print += str(self.point)
        the_print += " Observation index: "
        the_print += str(self.observations_index)

        return the_print

class Observation:
    def __init__(self, image_coordinates, view_index, point_3D_index):
        self.image_coordinates = image_coordinates
        self.view_index = view_index
        self.point_3D_index = point_3D_index
    def __str__(self):
        the_print = "OBSERVATION: "
        the_print += "Image coords: "
        the_print += str(self.image_coordinates)
        the_print += " View index: "
        the_print += str(self.view_index)
        the_print += " 3D point index "
        the_print += str(self.point_3D_index)

        return the_print

class View:
    def __init__(self, image, camera_pose):
        self.image = image
        self.camera_pose = camera_pose
        self.observations_index = np.array([], dtype = 'int')

    def __str__(self):
        the_print = "VIEW: "
        the_print += "Image index: "
        the_print += str(self.image)
        the_print += " Camera pose: "
        the_print += str(self.camera_pose)
        the_print += " Observations table "
        the_print += str(self.observations_index)

        return the_print

class Tables:

    def __init__(self):
        self.T_obs = np.array([], dtype = 'object')
        self.T_views = np.array([], dtype = 'object')
        self.T_points = np.array([], dtype = 'object')

    def addView(self,image, pose):
        new_view = np.array([View(image, pose)])
        self.T_views = np.append(self.T_views, new_view)
        return self.T_views.size-1 #Return index to added item

    def addPoint(self,coord):
        new_point = np.array([Point_3D(coord)])
        self.T_points = np.append(self.T_points, new_point)
        return self.T_points.size-1 #Return index to added item

    def addObs(self,coord, view_index, point_index):
        new_obs = np.array([Observation(coord, view_index, point_index)])
        self.T_obs = np.append(self.T_obs, new_obs)
        self.T_views[view_index].observations_index = self.T_obs.size - 1
        self.T_points[point_index].observations_index = self.T_obs.size - 1

    def __str__(self):
        print_array = np.vectorize(str, otypes=[object])
        the_print = "TABLES: \n"
        the_print += "3D points table: \n"
        the_print += str(print_array(self.T_points))
        the_print += "\nView table\n"
        the_print += str(print_array(self.T_views))
        the_print += "\nObservations table\n"
        the_print += str(print_array(self.T_obs))

        return the_print
    """
    def GetImgCoords(self):
        for obs in self.T_obs:
            self.T_obs
        return T_obs.
    """
    def BundleAdjustment(self):
        #y1 and y2 must be homogenous coordinates and normalized
        def distance(y1, y2):
            return np.abs(y1-y2)
        #Function to minimise in order to do bundle adjustment. Input should be
        #all observations so far
        def EpsilonBA(x0, yij):
            ratio = int((x0.shape[0]/16)*12)
            size = int(ratio/(3*4))
            Rktk = x0[:ratio]
            xj = x0[ratio:]
            Rktk = np.reshape(Rktk, [size, 3, 4])
            xj = np.reshape(xj, [size, 4])
            return np.sum(distance(yij,Rktk@xj.T)**2)

        size = self.T_obs.shape[0]

        yij = []
        Rktk = []
        xj = []

        for o in self.T_obs:
            for i in o.image_coordinates:
                yij.append(i)
                Rktk.append(self.T_views[o.view_index].camera_pose.GetCameraMatrix())
                point = self.T_points[o.point_3D_index].point
                xj.append([point[0],point[1],point[2],1.0] )

        yij = np.asarray(yij)
        Rktk = np.asarray(Rktk)
        xj = np.asarray(xj)

        x0 = np.hstack([Rktk.flatten(), xj.flatten()])

        result = scipy.optimize.least_squares(EpsilonBA, x0, args=([yij]))

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in self.T_points:
            ax.scatter(i.point[0], i.point[1], i.point[2], marker='o', color='orange')
        for i in self.T_views:
            ax.scatter(i.camera_pose.t[0], i.camera_pose.t[1], i.camera_pose.t[2], marker='^', color='black')
        plt.show()

def MakeHomogenous(K, coord):
    #Normalizing corresponding points
    coord_hom = np.zeros((3,coord.shape[1]), dtype='double')
    coord_hom[:2,:] = coord[:2,:]
    coord_hom[-1,:] = 1

    return scipy.linalg.inv(K)@coord_hom

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
y1p = Fy1y2[1]
y2p = Fy1y2[2]

plt.imshow(images[0])
plt.scatter(y1p[0], y1p[1], color='orange')
plt.show()

"""
    INIT2: Get E = R,t from the two intial views
"""

T_tables = Tables()

C = np.asarray(Dino_36C.tolist())
K = np.zeros((C.shape[1],3,3))
R = np.zeros((C.shape[1],3,3))
t = np.zeros((C.shape[1],3))

for i in range(C.shape[1]):
    K, R[i,:,:], t[i,:] = fun.camera_resectioning(C[0,i,:,:])

y1 = MakeHomogenous(K, y1p)
y2 = MakeHomogenous(K, y2p)

#Calculate E = K.T*F*K
E = np.matmul(np.transpose(K),np.matmul(F,K))
R, t = fun.relative_camera_pose(E, y1[:,0], y2[:,0])
C1 = CameraPose()
C2 = CameraPose(R,t)

#Add index 0 and C1 for image 1 and first camera pose. Same for second image and C2
view_index_1 = T_tables.addView(0,C1)
view_index_2 = T_tables.addView(1,C2)

"""
    INIT3: Triangulate points.
"""

for i in range(y1.shape[1]):
    #Triangulate points and add to tables
    new_3D_point = lab3.triangulate_optimal(C1.GetCameraMatrix(), C2.GetCameraMatrix(), y1[:,i], y2[:,i])
    point_index = T_tables.addPoint(new_3D_point)
    T_tables.addObs(y1[:,i], view_index_1, point_index)
    T_tables.addObs(y2[:,i], view_index_2, point_index)

#print(T_tables)
T_tables.plot()

result = T_tables.BundleAdjustment()
np.save("BAresult1", result.x)
#result = np.load("BAresult1.npy", allow_pickle=True)

print(result)

"""
    Iterate through all images in sequence
"""
#for img in images[:2]:

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
