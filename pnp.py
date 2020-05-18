import cv2
import numpy as np
import math # math.nan
#import cmath # complex(x,y)

# p3p using openCV

def p3p(_3d_pts, img_pts, K):
    R, t = cv2.solvePnP(_3d_pts, img_pts, K, cv2.SOLVEPNP_ITERATIVE)

    return R, t


def mix(n, m):
    return (n, m, np.cross(n, m))


def normalize(v): 
    #norm = np.linalg.norm(v, 1)
    #print(v[:2])
    norm = np.linalg.norm(v[:2])
    
    if np.abs(norm) < np.finfo(float).eps:

        raise ValueError("Normilization failed due to division by 0")

    v[:2] = v[:2] / norm
    
    return v

def get_eig_vector(m, r):
    c = r*r + m[0]*m[4] - r*(m[0] + m[4]) - m[1]*m[1]

    a0 = (r*m[2] + m[1]*m[5] - m[2]*m[4]) / c
    a1 = (r*m[5] + m[1]*m[2] - m[0]*m[5]) / c

    v = [a0, a1, 1]
    v = normalize(v)

    return v

# Help function for p3p_twist
def eig_3x3_known_0(M):
    b2 = np.cross(M[:,1], M[:,2])
    b2 = normalize(b2)

    m = M.flatten()
    #print(M)
    #print(m)

    p = np.zeros((3,1))
    p2 = 1.0
    p1 = m[0] - m[4] - m[8]
    p0 = -m[1]*m[1] - m[2]*m[2] - m[5]*m[5] + m[0]*m[4] + m[8] + m[4]*m[8]
    p = [p2, p1, p0]

    sigma0, sigma1 = np.roots(p)
    #print(sigma0*sigma0 + p1*sigma0 + p0)
    #print(sigma1*sigma1 + p1*sigma1 + p0)
    b0 = get_eig_vector(m, sigma0)
    b1 = get_eig_vector(m, sigma1)

    if np.abs(sigma0) > np.abs(sigma1):
        return [b0, b1, b2], sigma0, sigma1
    else:
        return [b1, b0, b2], sigma1, sigma0


def isCollinear(p):
    arg0 = (p[1,1]-p[0,1])*(p[2,2]-p[0,2]) - (p[2,1]-p[0,1])*(p[1,2]-p[0,2])
    arg1 = (p[2,0]-p[0,0])*(p[1,2]-p[0,2]) - (p[1,0]-p[0,0])*(p[2,2]-p[0,2])
    arg2 = (p[1,0]-p[0,0])*(p[2,1]-p[0,1]) - (p[2,0]-p[0,0])*(p[1,1]-p[0,1]) 
    if np.abs(arg0) + np.abs(arg1) + np.abs(arg2) < np.finfo(float).eps:
        return True
    return False 


# p3p using twist
def p3p_twist(y, x):
    # img_pts == y
    # _3d_pts == x

    if isCollinear(y) == True:
        raise ValueError("y should not be collinear")
    if isCollinear(x) == True: 
        raise ValueError("x should not be collinear")
 
    #print(y[0,:])
    #print(x)
    y_normalized = np.zeros(y.shape)
    y_normalized[0,:] = normalize(y[0,:])
    y_normalized[1,:] = normalize(y[1,:])
    y_normalized[2,:] = normalize(y[2,:])

    #print(y_normalized[0,:])
    a = np.zeros(x.shape)
    b = np.zeros(y.shape)
        

    for i in range(3):
        for j in range(3):
            a[i,j] = np.dot(x[i,:] - x[j,:], x[i,:] - x[j,:]) #should be square(sqrt(dot())), but unnecessary
            b[i,j] = np.matmul(y_normalized[i,:].T, y_normalized[j,:])            
    #print(a)
    #print(b)

    M01 = np.asarray([[1, -b[0,1], 0], [-b[0,1], 1, 0], [0, 0, 0]])
    M02 = np.asarray([[1, 0, -b[0,2]], [0, 0, 0], [-b[0,2], 0, 1]])
    M12 = np.asarray([[0, 0, 0], [0, 1, -b[1,2]], [0, -b[1,2], 1]])

    #print(M01)
    #print(M02)
    #print(M12)

    D1 = M01*a[1,2] - M12*a[0,1]
    D2 = M02*a[1,2] - M12*a[0,2]

    # where dij is column j of the matrix Di.
    # Cumpute gamma by finding the solution to 0 = det(D_1 + gamma * D_2):
    c = np.zeros((4,1))
    c3 = np.linalg.det(D2)
    c2 = np.matmul(D2[:,0].T, np.cross(D1[:,1], D1[:,2])) + np.matmul(D2[:,1].T, np.cross(D1[:,2], D1[:,0])) + np.matmul(D2[:,2].T, np.cross(D1[:,0], D1[:,1]))
    c1 = np.matmul(D1[:,0].T, np.cross(D2[:,1], D2[:,2])) + np.matmul(D1[:,1].T, np.cross(D2[:,2], D2[:,0])) + np.matmul(D1[:,2].T, np.cross(D2[:,0], D2[:,1]))
    c0 = np.linalg.det(D1)
    c = np.array([c3, c2, c1, c0]).T
    
    roots = np.roots(c)
    #print(roots)

    gamma = math.nan
    for i in range(len(roots)):

        if np.isreal(roots[i]) == True: 
            print(roots[i])

            gamma = roots[i].real
            break

    if gamma == math.nan:
        raise ValueError("No real root found, therefore cannot determine D0")

    polSolution = c3*gamma*gamma*gamma + c2*gamma*gamma + c1*gamma + c0 
    #print(polSolution)    

    #print(gamma)
    D0 = D1 + gamma * D2
    #print(D0)
    print(np.linalg.det(D0))
    
    
    E, sigma0, sigma1 = eig_3x3_known_0(D0)

    E = np.array(E)

    S1 = E.T @ D0 @ E
    #print(S1)
    #print(sigma0,sigma1)
    #print(E)
    '''
    #s = np.sqrt((-sigma1)/sigma0)
    #s = [-s, s]

    '''



    #print(sum(y_normalized[1,:]))
    #print(x)
    return 0,0






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
