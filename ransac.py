import numpy as np
from pnp import p3p
from pnp import p3p_twist
import random


def calc_p(w, n, r):
    return 1-np.power(1-np.power(w, n), r)

def calc_r(w, n, p):
    return np.log(1-p)/np.log(1-np.power(w,n))

def gen_rnd_indices(set_length, n):
    if set_length < n: 
        raise ValueError("Cannot generate more indices than the amount of values in the set from which they are extracted. n should therefore be smaller or equal to set_length")
    
    shuffled_indices = list(range(set_length))
    random.shuffle(shuffled_indices)
    
    return shuffled_indices[0:n]

def norm_p(v):
    v=np.array(v)
    return np.ndarray.tolist(v / v[-1])

def cart(v):
    return norm_p(v)[0:-1]

def dpp(y1, y2):
    return np.sqrt(np.dot(norm_p(y1) - norm_p(y2), norm_p(y1) - norm_p(y2)))

def dpp_squared(y1, y2):
    return np.dot(norm_p(y1) - norm_p(y2), norm_p(y1) - norm_p(y2))

def calc_y_prim(x,R,t):
    return (R @ x) + t

def homogeneous_to_cartesian(D_part):
    if D_part.shape[1] == 7:
        temp1 = D_part[:,0:2] / D_part[:,[2]]
        temp2 = D_part[:,3:6] / D_part[:,[-1]]
        D_part = np.zeros((D_part.shape[0],5))
        D_part[:,0:2] = temp1
        D_part[:,2:] = temp2
    return D_part

def homogeneous_y(D_part, h):
    temp = D_part
    D_part = np.zeros((D_part.shape[0], 6))
    
    D_part[:,:2] = temp[:,:2] * h
    D_part[:,2] = h
    D_part[:,3:] = temp[:,2:]

    return D_part

def ransac_robust(D_med, D_high, r, thresh, n, pnp):
    """RANSAC algorithm for robust estimation of camera pose
    
    Parameters 
    ------------------
    D_med:
        Medium level correct correspondences between C-norm image points and 3D points 
    D_high:
        High level correct correspondences between C-norm image points and 3D points 
    r:
        the number of trials (section 17.3.3)
    t:
        threshold determining membership of a pair of points in the consensus set
    n: 
        number of 2D-3D correspondences
    pnp: 
        further detaljs on the chosen pnp

    Returns
    ------------------
    R_est, t_est:
        An estimated camera pose
    C_est:
        The corresponding consensus set, containing only correct corresp with probability p
    """    
    #Convert D to kartesian coordinates if they were given as homogeneous
    #print(D_med.shape[1])
    D_med = homogeneous_to_cartesian(D_med)
    D_high = homogeneous_to_cartesian(D_high)

    D_med = homogeneous_y(D_med, 1)
    D_high = homogeneous_y(D_high, 1)
    
    # Initialize the estimations as empty sets
    R_est = [] 
    t_est = []
    C_est = []
    
    # Extract the image points of D_med and D_high
    y_med = D_med[:,:3]
    y_high = D_high[:,:3] 

    x_med = D_med[:,3:]
    x_high = D_high[:,3:]

    #print(x_med.shape)
    
    # Iterate r times, i.e. the number of trials, which depends on the probability p
    for i in range(r):
        # Debugging 
        #print("initiate trial nr: " + str(i))

        # Pick a random subset T from D_high
        T_indices = gen_rnd_indices(len(D_high[0]), n)
        T = D_high[T_indices,:]

        #print(T[:,:3], T[:,3:] )
    
        # Decide which pnp to use

        R = []
        t = []
        
        if n == 3: 
            if pnp == "twist":
                R, t = p3p_twist(T[:,:3], T[:,3:])
            elif pnp == "opencv":
                R, t = p3p(T[:,:3], T[:,3:]) 
        elif n == 4:
            # TODO: implement p4p to easily combine with openCV
            raise ValueError("Not implemented yet")
        else: 
            raise ValueError("No PnP algorithm with the given n is implemented")
    
    '''
        # Iterate over the possible poses (maximum of 4 from p3p)
        for j in range(R[0]):
            C = [] # Initialize C as an empty set
            
            # Calculate the y_prim image points
            y_prim_med = calc_y_prim(D_med[:,1],R[j,:,:],t[j,:])
            y_prim_high = calc_y_prim(D_high[:,1],R[j,:,:],t[j,:])

            # Calculate the error
            e_med = dpp_squared(y_med, y_prim_med)
            e_high = dpp_squared(y_high, y_prim_high)

            # Append the point pair to C if the threshold is larger than the error
            C.append(D_med[thresh >= e_med])
            C.append(D_high[thresh >= e_high])

            # Keep the R_est, t_est and C_est correlated with the larges consensus set size 
            if len(C[0]) > len(C_est[0]):
                R_est = [R] 
                t_est = [t]
                C_est = [C]
    ''' 
    return R_est, t_est, C_est