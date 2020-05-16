import lab3
import cv2 as cv
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
import math
import scipy.io as sio
from correspondences import Correspondences
from help_classes import CameraPose, Point_3D, Observation, View

def getEFromCameras(C1, C2):

    R = C2.R @ C1.R.T
    t = C2.t - (C2.R @ C1.R.T @ C1.t)

    t_cross = crossProductMat(t)

    #Essential matrix E
    E = R.T @ t_cross
    return E


def crossProductMat(vec3):

    result = np.zeros([3,3])
    result[0,1] = -1*vec3[2]
    result[1,0] = vec3[2]
    result[0,2] = vec3[1]
    result[2,0] = -1*vec3[1]
    result[1,2] = -1*vec3[0]
    result[2,1] = vec3[0]

    return result

def decomposeP(P):

    M = P[0:3,0:3]
    Q = np.eye(3)[::-1]
    P_b = Q @ M @ M.T @ Q
    K_h = Q @ np.linalg.cholesky(P_b) @ Q
    K = K_h / K_h[2,2]
    A = np.linalg.inv(K) @ M
    l = (1.0/np.linalg.det(A)) ** (0.33333333)
    R = l * A
    t = l * np.linalg.inv(K) @ P[0:3,3]
    return K, R, t

def MakeHomogenous(K, coord):
    #Normalizing corresponding points
    coord = coord.T
    coord_hom = np.zeros((3,coord.shape[1]), dtype='double')
    coord_hom[:2,:] = coord[:2,:]
    coord_hom[-1,:] = 1
    coord_hom = scipy.linalg.inv(K)@coord_hom
    return coord_hom.T

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
    cameras = np.asarray(cameras.tolist())
    return cameras

def getEAndK(C, F):
    K = np.zeros((C.shape[1],3,3))
    R = np.zeros((C.shape[1],3,3))
    t = np.zeros((C.shape[1],3))

    #Get K, R and t for each camera
    for i in range(C.shape[1]):
        K, R[i,:,:], t[i,:] = camera_resectioning(C[0,i,:,:])

    #Calculate essential matrix E = K.T*F*K
    E = np.matmul(np.matmul(np.transpose(K),F),K)
    return E, K

def reshapeToCamera3DPoints(x0):
    ratio = int((x0.shape[0]/16)*12)
    size = int(ratio/(3*4))
    Rktk = x0[:ratio]
    xj = x0[ratio:]
    Rktk = np.reshape(Rktk, [size, 3, 4])
    xj = np.reshape(xj, [size, 4])
    return Rktk, xj

def matchingMatrix(roi1, roi2) :
    matrix = np.empty((len(roi1),len(roi2)))
    # the matching score is the MSE
    for i in range(len(roi1)) :
        for j in range(len(roi2)) :
            matrix[i,j] = np.linalg.norm(roi1[i] - roi2[j])
            #print(matrix[i,j])
    return matrix

def f_matrix(coords1_t, coords2_t) :


    """
    plt.imshow(img1)
    plt.scatter(coords1[0], coords1[1])
    plt.show()
    """
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

    F, mask = cv.findFundamentalMat(coords1_t, coords2_t, cv.FM_8POINT)

    # We select only inlier points
    coords1_t = coords1_t[mask.ravel()==1]
    coords2_t = coords2_t[mask.ravel()==1]

    inl_coords1 = coords1_t.T
    inl_coords2 = coords2_t.T
    """
    lab3.show_corresp(img1, img2, inl_coords1, inl_coords2)
    plt.show()
    """
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

    return F, inl_coords1, inl_coords2

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
        U[0,:] = U[0,:]*-1.0
        Q[:,0] = Q[:,0]*-1.0

    return U, Q

def specSVD(M):
    U, S, V = scipy.linalg.svd(M)

    det_U = np.linalg.det(U)
    det_V = np.linalg.det(V)

    un = U[:,-1]
    vm = V[:,-1]

    U[:,-1] = det_U * un
    V[:,-1] = det_V * vm

    s = S[-1]
    s1 = det_U*det_V*s
    S[-1] = s1

    return U, S, V


def relative_camera_pose(E, y1, y2):

    #Inpute is C-normalized coordinates
    U,S,Vh = specSVD(E)
    W = np.zeros((3,3))
    W[0,1] = 1
    W[1,0] = -1
    W[2,2] = 1

    R = np.transpose(Vh)@W@np.transpose(U)
    R_prim = np.transpose(Vh)@np.transpose(W)@np.transpose(U)

    V = np.transpose(Vh)
    t1 = V[:,-1]
    t2 = V[:,-1]*-1

    #C0 [I | 0]
    identity = CameraPose()
    C0 = np.zeros((3,4), dtype='double')
    C0[:3,:3] = np.eye(3)

    C1 = CameraPose(R, t1)
    C2 = CameraPose(R_prim, t1)
    C3 = CameraPose(R, t2)
    C4 = CameraPose(R_prim, t2)

    #case1
    x1 = lab3.triangulate_optimal(identity.GetCameraMatrix(), C1.GetCameraMatrix(), y1, y2)
    x2 = (R@x1)+t1
    if x1[-1] > 0 and x2[-1] > 0:

        return R, t1
    #case2
    x1 = lab3.triangulate_optimal(identity.GetCameraMatrix(), C2.GetCameraMatrix(), y1, y2)
    x2 = (R@x1)+t2
    if x1[-1] > 0 and x2[-1] > 0:

        return R, t2
    #case3
    x1 = lab3.triangulate_optimal(identity.GetCameraMatrix(), C3.GetCameraMatrix(), y1, y2)
    x2 = (R_prim@x1)+t1
    if x1[-1] > 0 and x2[-1] > 0:

        return R_prim, t1
    #case4
    x1 = lab3.triangulate_optimal(identity.GetCameraMatrix(), C4.GetCameraMatrix(), y1, y2)
    x2 = (R_prim@x1)+t2
    if x1[-1] > 0 and x2[-1] > 0:

        return R_prim, t2

def camera_resectioning(C):
    A = C[0:3,0:3]
    b = C[:,-1]

    U, Q = specRQ(A)
    t = np.matmul(scipy.linalg.inv(U), b)
    U = U/U[-1,-1]
    D = np.zeros([3,3])
    D[0,0] = np.sign(U[0,0])
    D[1,1] = np.sign(U[1,1])
    D[2,2] = np.sign(U[2,2])

    K = U@D

    if np.linalg.det(D) == 1:
        R = D@Q
        t = D@t
    else:
        R = -1*D@Q
        t = -1*D@t
    return K,R,t

def reshapeToCamera3DPoints2(x0, n_C, n_P):
    #ratio = int((x0.shape[0]/16)*12)
    #size = int(ratio/(3*4))
    Rktk = x0[:n_C*12]
    xj = x0[n_C*12:]
    Rktk = np.reshape(Rktk, [n_C, 3, 4])
    xj = np.reshape(xj, [n_P, 3])
    return Rktk, xj
