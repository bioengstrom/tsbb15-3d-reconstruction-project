import numpy as np
import scipy.io as sio

class Correspondences:

    def __init__(self):
        #load noisy
        
        self.points = np.loadtxt('imgdata/points.txt')
        
        """
        #load cleaned
        points = sio.loadmat('BAdino2.mat')
        newPoints2D = points['newPoints2D']
        newPoints2D = np.asarray(newPoints2D.tolist())
        self.points = newPoints2D[0,:,:,:]
        """

    def getCorrByIndices(self,i1, i2):
        
        #load noisy
        y1 = self.points[:,i1*2:(i1*2)+2]
        y2 = self.points[:,i2*2:(i2*2)+2]
        
        """
        #load cleaned
        y1 = self.points[i1,:,:].T
        y2 = self.points[i2,:,:].T
        """

        #Remove all rows in y1 and y2 that has -1:s in any of y1 and y2
        is_correspondence_y1 = np.array([np.any(y1 != -1, axis=1)], dtype='bool')
        is_correspondence_y2 = np.array([np.any(y2 != -1, axis=1)], dtype='bool')
        correspondence = np.logical_and(is_correspondence_y1, is_correspondence_y2)
        y1 = np.array(y1[correspondence[0], :])
        y2 = np.array(y2[correspondence[0], :])

        return y1, y2
"""
    def getNoOfCorrespondences(self, i1, i2):
        size = i2-i1
        correspondence = np.zeros([size, points.shape[0]], dtype='bool')
        for i in range(i1, 1, i2):
            y1 = self.points[:,i1*2:(i1*2)+2]
            y2 = self.points[:,i1*2:(i1*2)+2]
            #Remove all rows in y1 and y2 that has -1:s in any of y1 and y2
            correspondence = np.array([np.any(y1 != -1, axis=1)], dtype='bool')
            is_correspondence_y2 = np.array([np.any(y2 != -1, axis=1)], dtype='bool')
            corr_y1_y2 = np.logical_and(is_correspondence_y1, is_correspondence_y2)
            correspondence = np.logical_and(corr_y1_y2, is_correspondence_y2)
"""
