import lab3
import cv2 as cv
import numpy as np
import scipy
from matplotlib import pyplot as plt
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

    point = np.loadtxt('imgdata\points.txt')
    points = point[:,:4]
    coords1_t = points[:,0:2]
    coords2_t = points[:,2:4]
    coords1_t = coords1_t[np.any(coords1_t != -1, axis=1), :]
    coords2_t = coords2_t[np.any(coords2_t != -1, axis=1), :]
    coords2_t = coords2_t[:coords1_t.shape[0],:]
    coords1 = coords1_t.T
    coords2 = coords2_t.T
    plt.imshow(img1)
    plt.scatter(coords1[0], coords1[1])
    plt.show()

    # harris_1 = lab3.harris(img1, 7, 3)
    # harris_2 = lab3.harris(img2, 7, 3)
    #
    # harris_1 = lab3.non_max_suppression(harris_1, 9)
    # harris_2 = lab3.non_max_suppression(harris_2, 9)
    # coords1 = np.vstack(np.nonzero(harris_1 > 0.001*np.max(harris_1)))
    # coords2 = np.vstack(np.nonzero(harris_2 > 0.001*np.max(harris_2)))
    # coords1 = np.flipud(coords1)
    # coords2 = np.flipud(coords2)
    #
    # roi1 = lab3.cut_out_rois(img1, coords1[0], coords1[1], 7)
    # roi2 = lab3.cut_out_rois(img2, coords2[0], coords2[1], 7)
    # roi1 = np.asarray(roi1); roi2 = np.asarray(roi2)
    #
    # matrix = matchingMatrix(roi1, roi2)
    # vals, ri, ci = lab3.joint_min(matrix)
    #
    # coords1 = coords1[:,ri]
    # coords2 = coords2[:,ci]
    #
    # coords1_t = coords1.T
    # coords2_t = coords2.T

    F, mask = cv.findFundamentalMat(coords1_t, coords2_t, cv.FM_RANSAC)

    # We select only inlier points
    coords1_t = coords1_t[mask.ravel()==1]
    coords2_t = coords2_t[mask.ravel()==1]

    inl_coords1 = coords1_t.T
    inl_coords2 = coords2_t.T

    lab3.show_corresp(img1, img2, inl_coords1, inl_coords2)
    plt.show()

    # camera 1 and 2
    C1, C2 = lab3.fmatrix_cameras(F)
    X = np.empty((3,inl_coords1.shape[1]))
    for i in range(inl_coords1.shape[1]) :
        X[:,i] = lab3.triangulate_optimal(C1, C2, inl_coords1[:,i], inl_coords2[:,i])

    # minimize using least_squares
    params = np.hstack((C1.ravel(), X.T.ravel()))
    solution = least_squares(lab3.fmatrix_residuals_gs, params, args=(inl_coords1,inl_coords2))

    C1 = solution.x[:12].reshape(3,4)
    F_gold = lab3.fmatrix_from_cameras(C1, C2)

    return F_gold, inl_coords1, inl_coords2

#input n x 3 matrix
def specRQ(M):
    #own implementation of special rq, doesnt work
    """
    m3 = M[-1, :]
    m2 = M[-2, :]
    #print(M)
    #print(m3)
    #print(m2)

    q3 = m3/np.linalg.norm(m3)
    print(np.dot(q3,m2))
    q2 = m2-(q3*np.dot(q3,m2))/(np.sqrt(np.linalg.norm(m2)**2-np.dot(q3,m2)**2))
    q1 = np.cross(q2,q3)

    Q = np.array((q1,q2,q3))
    U = M*np.matrix.transpose(Q)

    #test
    M1 = U*Q
    print(Q)
    print(U)
    """
    #maybe right implementation of special rq
    U, Q = scipy.linalg.rq(M)
    if np.linalg.det(Q) == -1:
        U[0,:] = U[0,:]*-1
        Q[0,:] = Q[0,:]*-1

    return U, Q

def specSVD(M):
    U, S, V = scipy.linalg.svd(M)

    un = U[:,-1]
    vm = V[:,-1]

    U[:,-1] = np.linalg.det(U)*un
    V[:,-1] = np.linalg.det(V)*vm

    s = S[-1]
    s1 = np.linalg.det(U)*np.linalg.det(V)*s
    S[-1] = s1

    return U, S, V

def relative_camera_pose(E, y1, y2):
    U,S,Vh = specSVD(E)
    W = np.zeros((3,3))
    W[0,1] = 1
    W[1,0] = -1
    W[2,2] = 1

    R1 = np.transpose(Vh)@W@np.transpose(U)
    R2 = np.transpose(Vh)@np.transpose(W)@np.transpose(U)

    V = np.transpose(Vh)
    t1 = V[:,-1]
    t2 = V[:,-1]*-1

    #C0 [I | 0]
    C0 = np.zeros((3,4), dtype='double')
    C0[:3,:3] = np.eye(3)

    C1t1R1 = np.zeros((3,4), dtype='double')
    C1t1R1[:,-1] = t1
    C1t1R1[:3,:3] = R1
    
    C1t1R2 = np.zeros((3,4), dtype='double')
    C1t1R2[:,-1] = t1
    C1t1R2[:3,:3] = R2

    C1t2R1 = np.zeros((3,4), dtype='double')
    C1t2R1[:,-1] = t2
    C1t2R1[:3,:3] = R1

    C1t2R2 = np.zeros((3,4), dtype='double')
    C1t2R2[:,-1] = t2
    C1t2R2[:3,:3] = R2

    #case1
    x1 = lab3.triangulate_optimal(C0, C1t1R1, y1, y2)
    x2 = (R1@x1)+t1
    if x1[-1] > 0 and x2[-1] > 0:
        return R1, t1
    #case2
    x1 = lab3.triangulate_optimal(C0, C1t2R1, y1, y2)
    x2 = (R1@x1)+t2
    if x1[-1] > 0 and x2[-1] > 0:
        return R1, t2
    #case3
    x1 = lab3.triangulate_optimal(C0, C1t1R2, y1, y2)
    x2 = (R2@x1)+t1
    if x1[-1] > 0 and x2[-1] > 0:
        return R2, t1
    #case4
    x1 = lab3.triangulate_optimal(C0, C1t2R2, y1, y2)
    x2 = (R2@x1)+t2
    if x1[-1] > 0 and x2[-1] > 0:
        return R2, t2

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
