import lab3
import cv2 as cv
import numpy as np
import scipy
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from scipy.optimize import least_squares
import math

def matchingMatrix(roi1, roi2) :
    matrix = np.empty((len(roi1),len(roi2)))
    # the matching score is the MSE
    for i in range(len(roi1)) :
        for j in range(len(roi2)) :
            matrix[i,j] = np.linalg.norm(roi1[i] - roi2[j])
            #print(matrix[i,j])
    return matrix

def f_matrix(img1, img2) :

    # point = np.loadtxt('imgdata\points.txt')
    # points = point[:,:4]
    # coords1_t = points[:,0:2]
    # coords2_t = points[:,2:4]
    # coords1_t = coords1_t[np.any(coords1_t != -1, axis=1), :]
    # coords2_t = coords2_t[np.any(coords2_t != -1, axis=1), :]
    # coords2_t = coords2_t[:coords1_t.shape[0],:]
    # coords1 = coords1_t.T
    # coords2 = coords2_t.T
    # plt.imshow(img1)
    # plt.scatter(coords1[0], coords1[1])
    # plt.show()

    harris_1 = lab3.harris(img1, 7, 3)
    harris_2 = lab3.harris(img2, 7, 3)

    harris_1 = lab3.non_max_suppression(harris_1, 9)
    harris_2 = lab3.non_max_suppression(harris_2, 9)
    coords1 = np.vstack(np.nonzero(harris_1 > 0.001*np.max(harris_1)))
    coords2 = np.vstack(np.nonzero(harris_2 > 0.001*np.max(harris_2)))
    coords1 = np.flipud(coords1)
    coords2 = np.flipud(coords2)

    roi1 = lab3.cut_out_rois(img1, coords1[0], coords1[1], 7)
    roi2 = lab3.cut_out_rois(img2, coords2[0], coords2[1], 7)
    roi1 = np.asarray(roi1); roi2 = np.asarray(roi2)

    matrix = matchingMatrix(roi1, roi2)
    vals, ri, ci = lab3.joint_min(matrix)

    coords1 = coords1[:,ri]
    coords2 = coords2[:,ci]

    coords1_t = coords1.T
    coords2_t = coords2.T

    F, mask = cv.findFundamentalMat(coords1_t, coords2_t, cv.FM_RANSAC)

    # We select only inlier points
    coords1_t = coords1_t[mask.ravel()==1]
    coords2_t = coords2_t[mask.ravel()==1]

    inl1_coords1 = coords1_t.T
    inl2_coords2 = coords2_t.T

    lab3.show_corresp(img1, img2, inl1_coords1, inl2_coords2)
    plt.show()

    # camera 1 and 2
    C1, C2 = lab3.fmatrix_cameras(F)
    X = np.empty((3,inl1_coords1.shape[1]))
    for i in range(inl1_coords1.shape[1]) :
        X[:,i] = lab3.triangulate_optimal(C1, C2, inl1_coords1[:,i], inl2_coords2[:,i])

    # minimize using least_squares
    params = np.hstack((C1.ravel(), X.T.ravel()))
    solution = least_squares(lab3.fmatrix_residuals_gs, params, args=(inl1_coords1,inl2_coords2))

    C1 = solution.x[:12].reshape(3,4)
    F_gold = lab3.fmatrix_from_cameras(C1, C2)

    return F_gold, inl1_coords1, inl2_coords2

#input 3 x 4 camera matrix
def camera_resectioning(C):
    A = C[:,0:3]
    b = C[:,3]

    U, Q = specRQ(A)
    t = np.matmul(np.matrix.transpose(U), b)
    U = U/U[2,2]
    D = np.sign(U)
    K = U*D

    if np.linalg.det(D) == 1:
        R = np.matmul(D,Q)
        t = np.matmul(D,t)
    else:
        R = np.matmul(-1*D,Q)
        t = np.matmul(-1*D,t)
    return K,R,t
