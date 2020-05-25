import numpy as np

class CameraPose:
    def __init__(self, R = np.eye(3), t = np.array([0.0, 0.0, 0.0])):
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
        #self.observations_index = np.array([0], dtype = 'int')
        self.observations_index = np.zeros([0], dtype = 'int')

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
        self.color = color

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
        #self.observations_index = np.array([0], dtype = 'int')
        self.observations_index = np.zeros([0], dtype = 'int')

    def __str__(self):
        the_print = "VIEW: "
        the_print += "Image index: "
        the_print += str(self.image)
        the_print += " Camera pose: "
        the_print += str(self.camera_pose)
        the_print += " Observations table "
        the_print += str(self.observations_index)

        return the_print
