import numpy as np
from help_classes import CameraPose, Point_3D, Observation, View
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
from pnp import p3p
import cv2 as cv

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
        def EpsilonBA(x0, yij):
            Rktk, xj = reshapeToCamera3DPoints(x0)
            return np.sum(distance(yij[:,:,None],Rktk @ xj[:,:,None])**2)

        #Get arrays from objects
        yij, Rktk, xj = self.getObsAsArrays()

        x0 = np.hstack([Rktk.flatten(), xj.flatten()])
        result = scipy.optimize.least_squares(EpsilonBA, x0, args=([yij]))
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
        D_3Dpoints = np.zeros([0,3])
        D_imgcoords = np.zeros([0,3])
        x_i = np.zeros([0], dtype='int')
        #A is the set of putative correspondences that do not find any match
        A_y1 = np.zeros([0,3])
        A_y2 = np.zeros([0,3])

        for i in range(y1.shape[0]):
            for o in self.T_obs:
                if np.linalg.norm(o.image_coordinates-y1[i]) < 0.01:
                    #There is a corresponding 3D point x in T_points! Add y2, x to D
                    j = o.point_3D_index
                    x_i = np.concatenate((x_i, [j]), axis = 0)
                    #print(self.T_points[j].point.shape)
                    D_3Dpoints = np.concatenate((D_3Dpoints, [self.T_points[j].point]), axis = 0)
                    #print(y2[i].shape)
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
