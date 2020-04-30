import scipy.io as sio
import scipy
from scipy import linalg
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
    #print(U)
    #print(Q)
    t = np.matmul(np.matrix.transpose(U), b)
    print(t)
    #Normalize U
    U = U/U[2,2]
    #print(U)
    D = np.sign(U)
    #print(D)
    K = U*D
    #print(K)

    return t



Dino_36C = sio.loadmat('imgdata\dino_Ps.mat')

Dino_36C = Dino_36C['P']
#print(Dino_36C)
C = np.asarray(Dino_36C.tolist())
#print(C)
#C = np.reshape(C, (3,4,36))
#print(C.shape)
C1 = C[:,:,:,0:3]
b = C[0,0,:,3]

U = np.zeros((3,3,C1.shape[1]))
Q = np.zeros((3,3,C1.shape[1]))
t = camera_resectioning(C[0,0,:,:])
#for i in range(C1.shape[1]):
    #C1 = 
    #print(C1.shape)
    #U[:,:,i], Q[:,:,i] = specRQ(C1[0,i,:,:])
#print(U.shape)
#print(b.shape)
#t = np.matrix.transpose(U)*b
#print(t)
#print(R)
#print(Q)
#K = C1[0,:,:,:]*np.matrix.transpose(Q)
#print(K)
#print(r)
#print(q)
#print(K)