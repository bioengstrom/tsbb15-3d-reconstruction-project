import cv2
import numpy as np

# p3p using openCV
def p3p(_3d_pts, img_pts, C):
    R, t = cv2.solvePnP(_3d_pts, img_pts, C, cv2.SOLVEPNP_ITERATIVE)
    
    return R, t



# Outline of the PnP, based on algebraic minimization
"""
Input: A set of m 3D points: {xk} in homogeneous coordinates,m≥6
Input: A set of m corresponding image points{yk}, in C-normalized homogeneous coordinates
Output: (R| t), minimizing the algebraic error corresponding to Equation (15.35)

1   foreach k=1, . . . , m do 
2       foreach row∗ r_l ∈ [yk]_x do 
3           a = vectorization of the 3×4 matrix r_l x_k^T, as a row vector
4           Append this row at the bottom of A: A = [A;a]
5       end
6   end
7   Determine C0 from data matrix A: (found in section 13.3)
8       Use either the homogeneous method, or the inhomogeneous method, to find c0 in the null space ofA
9       Reshape the vector c0 to 3×4 matrix C0 
10  Constraint enforcement of C0 = (A|b):
11      Set τ = sign(det(A))
12      SVD: τ*A = U @ S @ V^T 
13      Set R = U @ V^T, λ = 3 * τ / trace(S), t = λ * b
14      Return C = (R | t) 
15 ∗Possibly, use only two of the rows in [yk]_x since the 3 rows are linearly dependent

http://liu.diva-portal.org/smash/get/diva2:1136229/FULLTEXT03.pdf 

http://www.cvl.isy.liu.se/research/publications/PRE/0.40/main-pre-0.40.pdf 

http://openaccess.thecvf.com/content_ECCV_2018/papers/Mikael_Persson_Lambda_Twist_An_ECCV_2018_paper.pdf

"""


# TODO finish up this pnp
def pnp_minimize(_3d_pts, img_pts, m):

    """PnP based on algebraic minimization
    
    Parameters 
    ------------------
    3D points:
        A set of m 3D points: {_3d_pts} in homogeneous coordinates, m >= 6
    Image points:
        A set of m corresponding image points: {img_pts}, in C-normalized homogeneous coordinates
    m:
        m amount of 3D points and image points respectively

    Returns
    ------------------
    R, t:
        Which minimizies the algebraic error
    """
    A = []
    
    for k in range(m): 
        for i in range(len(img_pts[0])): #make sure it traversed along the rows. Maybe not the 3 rows
            a = img_pts[i,k] @ _3d_pts[k].T #should it be transpose on the 3d points??
            a = np.array(a).flatten() #Vectorize 3x4 matrix into a row vector
           
            A.append(a) #Append a to A

     




    return R_minimized, t_minimized