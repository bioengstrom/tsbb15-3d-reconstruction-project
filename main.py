import scipy.io as sio
import scipy
from scipy import linalg
import scipy.cluster
import numpy as np

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



Dino_36C = sio.loadmat('imgdata\dino_Ps.mat')

Dino_36C = Dino_36C['P']
#print(Dino_36C)
C = np.asarray(Dino_36C.tolist())

K = np.zeros((C.shape[1],3,3))
R = np.zeros((C.shape[1],3,3))
t = np.zeros((C.shape[1],3))

for i in range(C.shape[1]):
    K[i,:,:], R[i,:,:], t[i,:] = camera_resectioning(C[0,i,:,:])
#print(K)
#print(R)
#print(t)