import numpy as np
from help_classes import CameraPose, Point_3D, Observation, View
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
from pnp import p3p
import cv2 as cv
import fun as fun
from scipy.sparse import lil_matrix
import lab3 as lab3
from scipy.spatial.transform import Rotation as R

class Tables:

    def __init__(self):
        self.T_obs = {}
        self.T_views = {}
        self.T_points = {}
        self.o_key = 0
        self.v_key = 0
        self.p_key = 0
        #self.T_obs = np.array([], dtype = 'object')
        #self.T_views = np.array([], dtype = 'object')
        #self.T_points = np.array([], dtype = 'object')

    def addView(self,image, pose):
        self.T_views[self.v_key] = View(image, pose)
        self.v_key = self.v_key + 1
        #new_view = np.array([View(image, pose)])
        #self.T_views = np.append(self.T_views, new_view)
        return self.v_key-1 #Return index to added item

    def addPoint(self,coord):
        self.T_points[self.p_key] = Point_3D(coord)
        self.p_key = self.p_key + 1
        #new_point = np.array([Point_3D(coord)])
        #self.T_points = np.append(self.T_points, new_point)
        return self.p_key-1 #Return index to added item

    def addObs(self,coord, view_key, point_key):
        self.T_obs[self.o_key] = Observation(coord, view_key, point_key)
        self.o_key = self.o_key + 1
        self.T_views[view_key].observations_index = np.concatenate((self.T_views[view_key].observations_index, [self.o_key - 1]), axis = 0)
        self.T_points[point_key].observations_index = np.concatenate((self.T_points[point_key].observations_index, [self.o_key - 1]), axis = 0)

        #new_obs = np.array([Observation(coord, view_index, point_index)])
        #self.T_obs = np.append(self.T_obs, new_obs)
        #self.T_views[view_index].observations_index = np.concatenate((self.T_views[view_index].observations_index, [self.T_obs.size - 1]), axis = 0)
        #self.T_points[point_index].observations_index = np.concatenate((self.T_points[point_index].observations_index, [self.T_obs.size - 1]), axis = 0)

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

    def deletePoint2(self, key) :

        p_obs = self.T_points[key].observations_index

        for v in self.T_views.values() :
            for i in p_obs :
                v.observations_index = v.observations_index[v.observations_index != i]

        for i in p_obs:
            self.T_obs.pop(i)

        self.T_points.pop(key)

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

    def updateCameras3Dpoints(self, new_pose, new_points):
        print(new_pose[0,:,3].shape)
        for i, o in enumerate(self.T_obs):
            t = new_pose[i,:,3]
            R = new_pose[i,:3,:3]
            self.T_views[o.view_index].camera_pose = CameraPose(R,t)
            self.T_points[o.point_3D_index].point = new_points[i]

    #Add new new view to T_views
    #coords1 & coords2 are the putative correspondeces
    def addNewView(self, K, img_index, y1_hom, y2_hom, y1, y2):
        images = fun.getImages()

        #image_coords, views, points_3D = self.getObsAsArrays()
        #Set D is the set that is containing matches with points already known
        D_3Dpoints = np.zeros([0,3])
        D_imgcoords = np.zeros([0,2])
        D_imgcoordsy1 = np.zeros([0,2])
        D_imgcoords_hom = np.zeros([0,3])
        x_i = np.zeros([0], dtype='int')
        #A is the set of putative correspondences that do not find any match
        A_y1 = np.zeros([0,2])
        A_y2 = np.zeros([0,2])
        print("y1 shape add new view")
        print(y1.shape)
        last_C = self.T_views[img_index-1].camera_pose

        for i in range(y1.shape[0]):
            found = False
            match = 0
            #Got through all observations seen in last view added
            for v in self.T_views[len(self.T_views)-1].observations_index:
                match = self.T_obs[v]
                if np.linalg.norm(match.image_coordinates-y1_hom[i]) < 0.000001:
                    found = True
                    break
            #There is a corresponding 3D point x in T_points! Add y2, x to D
            if found == True:
                j = match.point_3D_index
                x_i = np.concatenate((x_i, [j]), axis = 0)
                D_3Dpoints = np.concatenate((D_3Dpoints, [self.T_points[j].point]), axis = 0)
                D_imgcoords = np.concatenate((D_imgcoords, [y2[i]]), axis = 0)
                D_imgcoords_hom = np.concatenate((D_imgcoords_hom, [y2_hom[i]]), axis = 0)
                D_imgcoordsy1 = np.concatenate((D_imgcoordsy1, [y1[i]]), axis = 0)
            #No correspondence found - add to A
            else:
                A_y1 = np.concatenate((A_y1, [y1[i]]), axis = 0)
                A_y2 = np.concatenate((A_y2, [y2[i]]), axis = 0)

        #lab3.show_corresp(images[img_index-1], images[img_index], D_imgcoordsy1.T, D_imgcoords.T, vertical= False)
        #plt.show()

        print("Doing ransac pnp with n many elements:")
        print(D_3Dpoints.shape[0])
        #Pnp Algorithm return consensus set C of correspondences that agree with the estimated camera pose
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        last_CRod = np.empty((3))
        cv.Rodrigues(last_C.R, last_CRod)
        lastCthomo = np.append(last_C.t[:,np.newaxis], 1)
        retval, R, t, inliers = cv.solvePnPRansac(D_3Dpoints[:,:3], D_imgcoords[:,:2], K, dist_coeffs, rvec = last_CRod, tvec = lastCthomo)

        print("Ransac done!")
        #Make the rotation vector 3x3 matrix w open cv rodrigues method
        R, jacobian = cv.Rodrigues(R, R)
        consensus_coords = D_imgcoords_hom[inliers[:,0]]
        consensus_x_i = x_i[inliers[:,0]]

        #Set Camera pose C = (R | t) for img2
        C = CameraPose(R, t[:,0])

        #Add new view to T_views
        view_index = self.addView(img_index, C)
        for y2, x in zip(consensus_coords, consensus_x_i):
            #Add all image points to T_obs.
            self.addObs(y2, view_index, x)

        print("A y1 y2")
        print(A_y1.shape)
        print(A_y2.shape)

        #return a set of putative correspondences between the images so far without 3D points
        return A_y1, A_y2

    def addNewPoints(self, A_y1, A_y2, view_index_1, view_index_2):
        C1 = self.T_views[view_index_1].camera_pose
        C2 = self.T_views[view_index_2].camera_pose

        E = fun.getEFromCameras(C1, C2)
        counter = 0
        for i in range(A_y1.shape[0]):
            #Check epipolar constraint
            #print(np.abs(y1.T @ E @ y2))
            if np.abs(A_y1[i].T @ E @ A_y2[i]) < 0.1:
                x = lab3.triangulate_optimal(C1.GetCameraMatrix(), C2.GetCameraMatrix(), A_y1[i], A_y2[i])
                point_index = self.addPoint(x)
                self.addObs(A_y1[i], view_index_1, point_index)
                self.addObs(A_y2[i], view_index_2, point_index)
                counter = counter + 1
        return counter

    def plotProjections(self, index, K, image):
        """
        the_points = np.zeros([0,3])
        for point in self.T_points:
            the_points = np.concatenate((the_points, [point.point]), axis = 0)

        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        last_camera = self.T_views[index].camera_pose
        R = last_camera.R
        t = last_camera.t
        rvec, jac = cv.Rodrigues(R)
        img_points, jac = cv.projectPoints(the_points[:,:3], rvec, t, K, dist_coeffs)
        img_points = np.array(img_points[:,0,:])
        #Show the image with interest points
        plt.imshow(image)
        plt.scatter(img_points[:,0], img_points[:,1], color='orange')
        plt.show()
        """
        plt.imshow(image)
        for point in self.T_points.values():
            P1 = K @ self.T_views[index].camera_pose.GetCameraMatrix()
            point3Dcoord = np.array([point.point[0], point.point[1], point.point[2],1.0])
            proj1 = P1 @ point3Dcoord
            proj1 = (proj1 / proj1[2])
            proj1 = proj1[0:2]
            plt.scatter(proj1[0], proj1[1], c= 'r', s = 40)
        plt.show()

    def getCamerasForEvaluation(self):
        Rs = np.zeros([0,3,3])
        ts = np.zeros([0,3])

        for v in self.T_views.values():
            Rs = np.concatenate((Rs, [v.camera_pose.R]), axis=0)
            ts = np.concatenate((ts, [v.camera_pose.t]), axis=0)

        return Rs, ts

    def getColorFor3DPoint(self, point):
        color = np.zeros([0,3], dtype = 'float')
        #Get all colors for the 3D point
        for o in point.observations_index:
            color = np.concatenate((color, [self.T_obs[o].color]), axis=0)

        #Take the mean of the colors
        if len(point.observations_index) != 1:
            color = np.mean(color, axis=0)
        return color/255.0

    def getNormalFor3DPoint(self, point):
        vectorsToCameras = np.zeros([0,3])

        #Get the vectors from the point to the cameras where it is visible
        for o in point.observations_index:
            view_index = self.T_obs[o].view_index
            cameraPosition = self.T_views[view_index].getWorldPosition()
            vector = cameraPosition - point.point
            vectorsToCameras = np.concatenate((vectorsToCameras, [vector]), axis=0)

        #Take the median of the colors
        if len(point.observations_index) != 1:
            normal = np.average(vectorsToCameras, axis=0)

        return normal

    def get3DPointsColorsAndNormals(self):
        points = np.zeros([0,3])
        colors = np.zeros([0,3])
        normals = np.zeros([0,3])

        for p in self.T_points.values():
            points = np.concatenate((points, [p.point]), axis=0)
            colors = np.concatenate((colors, [self.getColorFor3DPoint(p)]), axis=0)
            normals = np.concatenate((normals, [self.getNormalFor3DPoint(p)]), axis=0)

        return points, colors, normals

    def plot(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_zlim([0, 10])
        for i in self.T_points.values():
            ax.scatter(i.point[0], i.point[1], i.point[2], marker='o', color='orange')
        for i in self.T_views.values():
            position = -1.0*(i.camera_pose.R.T @ i.camera_pose.t)
            ax.scatter(position[0], position[1], position[2], marker='^', color='black')

        plt.show()

    def BundleAdjustment2(self):
        #y1 and y2 must be homogenous coordinates and normalized
        def distance(y1, y2):
            return np.abs(y1-y2)
        #Function to minimise in order to do bundle adjustment. Input should be
        #all observations so far
        def EpsilonBA(x0, u, v, The_table):
            n_C = len(The_table.T_views)
            n_P = len(The_table.T_points)

            Rktk, xj = fun.reshapeToCamera3DPoints2(The_table, x0, n_C, n_P)
            #xj_h = np.zeros((len(xj), 4), dtype='double')
            #xj_h[:,:3] = xj[:,:3]
            #xj_h[:,-1] = 1
            for i in xj :
                xj[i] = np.append(xj[i][:,np.newaxis], 1)

            r = np.empty(([len(u)*2]))

            for i,o in enumerate (The_table.T_obs.values()):
                c = Rktk[o.view_index]
                c1 = c[0,:]
                c2 = c[1,:]
                c3 = c[2,:]
                x = xj[o.point_3D_index]

                r[i*2] = u[i] - (np.dot(c1,x)/np.dot(c3,x))
                r[(i*2)+1] = v[i] - (np.dot(c2,x)/np.dot(c3,x))

            #print(len(The_table.T_obs))
            #print(r.shape)
            return r.ravel()
            #return np.sum(distance(yij[:,:,None],Rktk @ xj[:,:,None])**2)

        #Get arrays from objects
        #yij, Rktk, xj = self.getObsAsArrays()
        #for o in self.T_views:
          #  Rktk = self.T_views[o.view_index].camera_pose.GetCameraMatrix()
        Rktk = np.empty((len(self.T_views), 3,4))
        xj = np.empty((len(self.T_points),3))
        yij = np.empty((len(self.T_obs),3))

        #Rk = np.empty((len(self.T_views), 3,3))
        tk = np.empty((len(self.T_views), 3))
        Rq = np.empty((len(self.T_views), 7))

        for i,o in enumerate(self.T_views.values()):
            Rktk[i] = o.camera_pose.GetCameraMatrix()
            Rk = R.from_matrix(Rktk[i,:3,:3])
            #print(Rk)
            tk[i] = Rktk[i,:,-1]
            Rq[i] = np.hstack([Rk.as_quat(), tk[i]])
        for i,o in enumerate(self.T_points.values()):
            xj[i] = o.point
        for i,o in enumerate(self.T_obs.values()):
            yij[i] = o.image_coordinates

        x0 = np.hstack([Rq.ravel(), xj.ravel()])
        #print(x0.shape)
        r = EpsilonBA(x0, yij[:,0],yij[:,1], self)
        result = least_squares(EpsilonBA, x0, args=([yij[:,0],yij[:,1], self]), jac_sparsity=self.sparsity_mask(), verbose=2, x_scale='jac', ftol=1e-4, method='trf')
        #print(result.jac.shape)
        #J_mask = self.sparsity_mask()
        #plt.figure()
        #plt.imshow(J_mask)

        """
        plt.figure()
        plt.subplot(311)
        plt.plot(r)

        plt.subplot(312)
        plt.plot(result.fun)

        plt.show()

        """
        n_C = len(self.T_views)
        n_P = len(self.T_points)
        new_pose, new_points = fun.reshapeToCamera3DPoints2(self, result.x, n_C, n_P)
        #print(new_pose.shape)
        #print(new_points.shape)

        self.updateCameras3Dpoints2(new_pose, new_points)

    #function that computes a suitable sparsity pattern as a function of the number of point correspondences
    def sparsity_mask(self):
        """
        Jc = np.zeros((len(self.T_obs)*2, (len(self.T_views)*12)))
        Jp = np.zeros((len(self.T_obs)*2, (len(self.T_points)*3)))
        k = 0
        for i,o in enumerate (self.T_obs):
            #if(o.view_index == 0):
             #   continue
            Jc[k:k+1,o.view_index*12:(o.view_index*12)+12] = 1
            Jp[k:k+1,o.point_3D_index*3:(o.point_3D_index*3)+3] = 1
            k = k+2

        J_mask = np.hstack((Jc,Jp))
        """
        n_parameters = 7
        point_idx = np.empty((len(self.T_obs)))
        camera_idx = np.empty((len(self.T_obs)))

        for i,o in enumerate(self.T_obs.values()):
            search_key1 = o.point_3D_index
            point_idx[i] = list(self.T_points.keys()).index(search_key1)
            search_key2 = o.view_index
            camera_idx[i] = list(self.T_views.keys()).index(search_key2)

        m = len(self.T_obs) * 2
        n = len(self.T_views) * n_parameters + len(self.T_points) * 3
        A = lil_matrix((m, n), dtype = 'int')

        i = np.arange(len(self.T_obs))
        for s in range(n_parameters):
            A[2 * i, camera_idx * n_parameters + s] = 1
            A[2 * i + 1, camera_idx * n_parameters + s] = 1


        for s in range(3):
            A[2 * i, len(self.T_views) * n_parameters + point_idx * 3 + s] = 1
            A[2 * i + 1, len(self.T_views) * n_parameters + point_idx * 3 + s] = 1

        A[:,0:n_parameters] = 0
       # plt.spy(A)
        #plt.show()
        return A

        #return J_mask

    def updateCameras3Dpoints2(self, new_pose, new_points):
        for key in self.T_views:
            t = new_pose[key][:,3]
            R = new_pose[key][:3,:3]
            self.T_views[key].camera_pose = CameraPose(R,t)
        for key in self.T_points:
            self.T_points[key].point = new_points[key]
