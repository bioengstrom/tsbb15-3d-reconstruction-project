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

# Get regions of interest
img = lab3.load_image("lab3img/indoors05.png")
img1 = lab3.load_image("lab3img/img1.png")
img2 = lab3.load_image("lab3img/img2.png")
img1 = lab3.rgb2gray(img1)
img2 = lab3.rgb2gray(img2)

harris_1 = lab3.harris(img1, 7, 3)
harris_2 = lab3.harris(img2, 7, 3)
plt.imshow(harris_1)

harris_1 = lab3.non_max_suppression(harris_1, 9)
harris_2 = lab3.non_max_suppression(harris_2, 9)
coords1 = np.vstack(np.nonzero(harris_1 > 0.001*np.max(harris_1)))
coords2 = np.vstack(np.nonzero(harris_2 > 0.001*np.max(harris_2)))
print(coords1.shape, coords2.shape)
coords1 = np.flipud(coords1)
coords2 = np.flipud(coords2)

roi1 = lab3.cut_out_rois(img1, coords1[0], coords1[1], 7)
roi2 = lab3.cut_out_rois(img2, coords2[0], coords2[1], 7)
roi1 = np.asarray(roi1); roi2 = np.asarray(roi2)
print(roi1.shape,roi2.shape)

matrix = matchingMatrix(roi1, roi2)
vals, ri, ci = lab3.joint_min(matrix)

coords1 = coords1[:,ri]
coords2 = coords2[:,ci]
print(coords1.shape, coords2.shape)
plt.figure()
plt.imshow(img1)
plt.scatter(coords1[0], coords1[1])
plt.show()

# Estimate F using 7-point algorithm
coords1_t = coords1.T
coords2_t = coords2.T
F, mask = cv.findFundamentalMat(coords1_t, coords2_t, cv.FM_RANSAC)

lab3.plot_eplines(F, coords2, img1.shape)
plt.figure()
lab3.plot_eplines(F.T, coords1, img2.shape)

# We select only inlier points
coords1_t = coords1_t[mask.ravel()==1]
coords2_t = coords2_t[mask.ravel()==1]

inl1_coords1 = coords1_t.T
inl2_coords2 = coords2_t.T

lab3.show_corresp(img1, img2, inl1_coords1, inl2_coords2)
plt.show()

#inl1_coords1 = coords1[:,total_inliers]
#inl2_coords2 = coords2[:,total_inliers]
# camera 1 and 2
C1, C2 = lab3.fmatrix_cameras(F)
X = np.empty((3,inl1_coords1.shape[1]))
for i in range(inl1_coords1.shape[1]) :
    X[:,i] = lab3.triangulate_optimal(C1, C2, inl1_coords1[:,i], inl2_coords2[:,i])
print(X.shape)

# minimize using least_squares
params = np.hstack((C1.ravel(), X.T.ravel()))
solution = least_squares(lab3.fmatrix_residuals_gs, params, args=(inl1_coords1,inl2_coords2))

C1 = solution.x[:12].reshape(3,4)
F_gold = lab3.fmatrix_from_cameras(C1, C2)
plt.figure()
lab3.plot_eplines(F_gold, inl2_coords2, img1.shape)
plt.figure()
lab3.plot_eplines(F_gold.T, inl1_coords1, img2.shape)
plt.show()
