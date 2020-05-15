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
    def getObsAsArrays(self):
        yij = []
        Rktk = []
        xj = []

        for o in self.T_obs:
                yij.append(o.image_coordinates)
                Rktk.append(self.T_views[o.view_index].camera_pose.GetCameraMatrix())
                point = self.T_points[o.point_3D_index].point
                xj.append([point[0],point[1],point[2],1.0])

        yij = np.asarray(yij)
        Rktk = np.asarray(Rktk)
        xj = np.asarray(xj)

        return yij, Rktk, xj

    def BundleAdjustment(self):
        #y1 and y2 must be homogenous coordinates and normalized
        def distance(y1, y2):
            return np.abs(y1-y2)
        #Function to minimise in order to do bundle adjustment. Input should be
        #all observations so far
        def EpsilonBA(x0, u, v, The_table):
            n_C = The_table.T_views.shape[0]
            n_P = The_table.T_points.shape[0]

            Rktk, xj = reshapeToCamera3DPoints(x0, n_C, n_P)
            xj_h = np.zeros((xj.shape[0], 4), dtype='double')
            xj_h[:,:3] = xj[:,:3]
            xj_h[:,-1] = 1

            r = np.empty(([len(u)*2]))
            #print((ck1*xj).sum(axis=1).shape)
            #print((ck1[0,:]*xj[0,:]).sum(axis=1))
            #print(c)
            #print(xj)
            #print(ck1*xj)
            #print(ukj)
            #print((ck1*xj).sum(axis=1)/(ck3*xj).sum(axis=1))

            for i,o in enumerate (self.T_obs):
                c = Rktk[o.view_index]
                c1 = c[0,:]
                c2 = c[1,:]
                c3 = c[2,:]
                x = xj_h[o.point_3D_index]

                r[i*2] = u[i] - (np.dot(c1,x)/np.dot(c3,x))
                r[(i*2)+1] = v[i] - (np.dot(c2,x)/np.dot(c3,x))
            #print(r.shape)
            #print(len(The_table.T_obs))
            return np.linalg.norm(r)  
            #return np.sum(distance(yij[:,:,None],Rktk @ xj[:,:,None])**2)

        #Get arrays from objects
        #yij, Rktk, xj = self.getObsAsArrays()
        #for o in self.T_views:
          #  Rktk = self.T_views[o.view_index].camera_pose.GetCameraMatrix()
        Rktk = np.empty((len(self.T_views), 3,4))
        xj = np.empty((len(self.T_points),3))
        yij = np.empty((len(self.T_obs),3))

        for i,o in enumerate(self.T_views):
            Rktk[i] = o.camera_pose.GetCameraMatrix()
        for i,o in enumerate(self.T_points):
            xj[i] = o.point
        for i,o in enumerate(self.T_obs):
            yij[i] = o.image_coordinates

        x0 = np.hstack([Rktk.flatten(), xj.flatten()])
        result = scipy.optimize.least_squares(EpsilonBA, x0, args=([yij[:,0],yij[:,1], self]))
        print(result.jac.shape)

        new_pose, new_points = reshapeToCamera3DPoints(result.x)

        self.updateCameras3Dpoints(new_pose, new_points)

    def updateCameras3Dpoints(self, new_pose, new_points):

        for i, o in enumerate(self.T_obs):
            t = new_pose[i,:,3]
            R = new_pose[i,:3,:3]
            self.T_views[o.view_index].camera_pose = CameraPose(R,t)
            self.T_points[o.point_3D_index].point = new_points[i]

    #Add new new view to T_views
    #coords1 & coords2 are the putative correspondeces
    def addNewView(self, K, img1, img2, img_index, y1, y2):
        print("Adding a view...")
        #image_coords, views, points_3D = self.getObsAsArrays()
        #Set D is the set that is containing matches with points already known
        D_3Dpoints = np.zeros([0,4])
        D_imgcoords = np.zeros([0,3])
        x_i = np.zeros([0], dtype='int')
        #A is the set of putative correspondences that do not find any match
        A_y1 = np.zeros([0,3])
        A_y2 = np.zeros([0,3])

        for i in range(y1.shape[0]):
            for o in self.T_obs:
                print(np.linalg.norm(o.image_coordinates-y1[i]))

                if np.linalg.norm(o.image_coordinates-y1[i]) < 0.01:
                    #There is a corresponding 3D point x in T_points! Add y2, x to D
                    j = o.point_3D_index
                    x_i = np.concatenate((x_i, [j]), axis = 0)
                    print(self.T_points[j].point.shape)
                    D_3Dpoints = np.concatenate((D_3Dpoints, [self.T_points[j].point]), axis = 0)
                    print(y2[i].shape)
                    D_imgcoords = np.concatenate((D_imgcoords, [y2[i]]), axis = 0)
                else:
                    #No correspondence found - add to A
                    A_y1 = np.concatenate((A_y1, [y1[i]]), axis = 0)
                    A_y2 = np.concatenate((A_y2, [y2[i]]), axis = 0)
        print(D_3Dpoints.shape)
        print(D_imgcoords.shape)
        print(x_i.shape)
        print("Doing ransac pnp with n many elements:")
        print()
        #Pnp Algorithm return consensus set C of correspondences that agree with the estimated camera pose
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        retval, R, t, inliers = cv.solvePnPRansac(D_3Dpoints[:,:3], D_imgcoords[:,:2], K, dist_coeffs)

        print("Ransac done!")
        #Make the rotation vector 3x3 matrix w open cv rodrigues method
        R, jacobian = cv.Rodrigues(R, R)
        consensus_coords = D_imgcoords[inliers[:,0]]
        consensus_x_i = x_i[inliers[:,0]]

        #Set Camera pose C = (R | t) for img2
        C = CameraPose(R, t)

        #Add new view to T_views
        view_index = self.addView(img_index, C)
        for y2, x in zip(consensus_coords, consensus_x_i):
            #Add all image points to T_obs.
            self.addObs(y2, view_index, x)

        #return a set of putative correspondences between the images so far without 3D points
        return A_y1, A_y2

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in self.T_points:
            ax.scatter(i.point[0], i.point[1], i.point[2], marker='o', color='orange')
        for i in self.T_views:
            ax.scatter(i.camera_pose.t[0], i.camera_pose.t[1], i.camera_pose.t[2], marker='^', color='black')
        plt.show()
    
    #function that computes a suitable sparsity pattern as a function of the number of point correspondences
    def sparsity_mask(self):
        
        Jc = np.zeros((len(self.T_obs)*2, (len(self.T_views)*12)))
        Jp = np.zeros((len(self.T_obs)*2, (len(self.T_points)*3)))
        
        k = 0
        for i,o in enumerate (self.T_obs):
            Jc[k:k+1,o.view_index*12:(o.view_index*12)+12] = 1
            Jp[k:k+1,o.point_3D_index*3:(o.point_3D_index*3)+3] = 1
            k = k+2
      
        J_mask = np.hstack((Jc,Jp))
        print(J_mask.shape)
        return J_mask
        
        

def MakeHomogenous(K, coord):
    #Normalizing corresponding points
    coord = coord.T
    coord_hom = np.zeros((3,coord.shape[1]), dtype='double')
    coord_hom[:2,:] = coord[:2,:]
    coord_hom[-1,:] = 1
    coord_hom = scipy.linalg.inv(K)@coord_hom
    return coord_hom.T

def reshapeToCamera3DPoints(x0, n_C, n_P):
    #ratio = int((x0.shape[0]/16)*12)
    #size = int(ratio/(3*4))
    Rktk = x0[:n_C*12]
    xj = x0[n_C*12:]
    Rktk = np.reshape(Rktk, [n_C, 3, 4])
    xj = np.reshape(xj, [n_P, 3])
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

class Correspondences:

    def __init__(self):
        self.points = np.loadtxt('imgdata/points.txt')

    def getCorrByIndices(self,i1, i2):
        y1 = self.points[:,i1*2:(i1*2)+2]
        y2 = self.points[:,i1*2:(i1*2)+2]

        #Make size same for both y1 and y2
        #size = np.minimum(y1.shape[0], y2.shape[0])
        #y1 = y1[:size,:]
        #y2 = y2[:size,:]

        #Remove all rows in y1 and y2 that has -1:s in any of y1 and y2
        is_correspondence_y1 = np.array([np.any(y1 != -1, axis=1)], dtype='bool')
        is_correspondence_y2 = np.array([np.any(y2 != -1, axis=1)], dtype='bool')
        correspondence = np.logical_and(is_correspondence_y1, is_correspondence_y2)
        y1 = y1[correspondence[0], :]
        y2 = y2[correspondence[0], :]

        return y1, y2

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
#plt.imshow(images[0])
#plt.scatter(y1p[0], y1p[1], color='orange')
#plt.show()

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

T_tables.sparsity_mask()
"""
    Iterate through all images in sequence
"""
T_tables.BundleAdjustment()

yp2, yp3 = correspondences.getCorrByIndices(1,2)
yp2 = MakeHomogenous(K, yp2)
yp3 = MakeHomogenous(K, yp3)
print(yp2.shape)
print(yp3.shape)

T_tables.addNewView(K, images[1], images[2], 2, yp2[:100], yp3[:100])
T_tables.plot()




for img in images[:1]:
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



    """
        EXT2: Find 2D<->3D correspondences. Algorithm 21.2
        EXT3: PnP -> R,t of new view and consensus set C
    """

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
