import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import pandas as pd

def FractionRoot(x,y):
    if x >= 0:
        return -1.0 / x**(1.0/y)
    else:  # x < 0
        return -FractionRoot(-x,y)

def GetNormalOfPlaneFromPoints(pi):
    #Compute normal from points with help of eigenvalues
    k = pi.shape[0]
    pbar = (1/k)*np.sum(pi[:], axis=0) #TODO: Check so that outer product makes sense and compute M
    pi_piT = pi[:,:,None]*pi[:, None, :]
    pbar_pbarT = pbar*pbar[None, :,None]
    M = (1/k)*np.sum(pi_piT-pbar_pbarT,axis=0) #Pbar is ok

    #Compute eigenvalues of M
    eigenvalues, eigenvectors = np.linalg.eig(M)

    #Find smallest eigenvalue: this corresponds to the eigenvector that is the surface normal
    normal_NN_index = np.argmin(eigenvalues)
    normal = eigenvectors[normal_NN_index]

    return normal

def GetKNeighbours(points, tree, c1, c2, sigma, E=0.1, k0 = 15):

    k = np.full(points.shape[0], k0)
    NN_indices = np.zeros(points.shape[0], dtype='object')

    for i in range(10):
        for j in range(points.shape[0]):
            # Get K nearest neighbours
            NN_distance, NN_index = tree.query(points[None,j], k[j])
            NN_index = NN_index[0]
            NN_indices[j] = NN_index

            #Calculate r, rho and K
            r = np.max(NN_distance)
            rho = k[j]/(np.pi*(r**2)+0.0001)

            #Estimate local curvature kapha
            #Calculate distance from point P to plane
            #Let PQ be a vector from arbitrary point Q in plane to the point P
            i = 1
            if NN_index.shape[0] == 1:
                i = 0
            PQ = points[j]-points[NN_index[i]]

            # Get distance by projecting PQ onto normal of plane
            normal = GetNormalOfPlaneFromPoints(points[NN_index])
            d = np.dot(PQ, normal)
            mu = NN_distance.mean()
            kapha = (2*d)/(mu**2 + 0.001)
            value = ((1/(kapha+0.0001))*(c1*(sigma/(np.sqrt(E*rho))+0.0001) + c2*sigma**2))
            r_new = FractionRoot(value,3.0)

            k[j] = np.pi*(r_new**2)*rho


    NN_indices = np.array(NN_indices)
    return NN_indices

"""
    Prepare data from PCD (Point Cloud Data)
"""
#Get point cloud data (PCD)
points = np.genfromtxt("stanfordbunny.txt", delimiter=" ")

#Create a KDTree with the point cloud
tree = KDTree(points)

"""
    Calculate r for each point in point cloud
"""
"""
Stanford algorithm
c1 = 100
c2 = 100
sigma = 0
NN_indices = GetKNeighbours(points, tree, c1, c2, sigma)
"""

#Naive implementation of indices
r = 0.01
NN_indices = tree.query_radius(points, r)
NN_indices = np.array(NN_indices)

"""
    Calculate normals
"""
normals = np.zeros(points.shape)

for i in range(NN_indices.shape[0]):

    #Get normals for each set of neighbours for each point
    pi = points[NN_indices[i]]
    normals[i] = GetNormalOfPlaneFromPoints(pi)

"""
    Plot data
"""
# Plot vector and selected k points for last point
last_i = NN_indices.shape[0]-1
pi = points[NN_indices[last_i]]
not_selected_points = np.delete(points, NN_indices[last_i], axis=0)
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
ax.plot(not_selected_points[:,0], not_selected_points[:,1], not_selected_points[:,2], 'o', markersize=10, color='b', alpha=0.2)
ax.plot(pi[:,0], pi[:,1], pi[:,2], 'o', markersize=10, color='r', alpha=0.2)

#Plot vector for normal
ax.quiver(
        points[last_i,0], points[last_i,1], points[last_i,2], # <-- starting point of vector
        normals[last_i,0], normals[last_i,1], normals[last_i,2], # <-- directions of vector
        arrow_length_ratio=0.001, color = 'red', alpha = 0.3, lw = 1
    )
plt.show()

"""
    Print to file
"""

#Print points and normals to file
result = np.append(points, normals, axis=1)
np.savetxt("bunny_normals.txt", result, delimiter=' ')
print("Woop! Printed results to file :^)")
