import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import pandas as pd
"""
    Prepare data from PCD (Point Cloud Data)
"""
#Get point cloud data (PCD)
points = np.genfromtxt("stanfordbunny.txt", delimiter=" ")

#Create a KDTree with the point cloud
tree = KDTree(points)

"""
    Get K nearest neighbours from radius r
"""
#Initialize variables
c1 = 0.5
c2 = 0.5
k = np.full(points.shape[0], 5)
r = np.full(points.shape[0], 0.01)
rho = np.zeros(points.shape[0])

NN_indices = np.zeros(points.shape[0], dtype='object')

for i in range(1):
    for j in range(points.shape[0]):
    #for j in range(1):
        NN_distance, NN_index = tree.query(points[None,j], k[j])  # get K nearest neighbours
        NN_indices[j] = NN_index

        #r[j] = np.max(NN_distance)
        #rho[j] = k[j]/(np.pi*(r[j]**2))
NN_indices = np.array(NN_indices)

"""
    Calculate normals
"""
normals = np.zeros(points.shape)
pi = []
for i in range(NN_indices.shape[0]):

    #Get the k points that are in the radius
    pi = points[NN_indices[i]]
    pi = pi[0]

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
    normals[i] = normal

"""
    Plot data
"""
# Plot vector and selected k points for last point
last_i = NN_indices.shape[0]-1
pi = points[NN_indices[last_i]]
pi = pi[0]
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
print("Printed results to file")
