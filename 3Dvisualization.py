import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from tables import Tables

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

"""
    Prepare data from PCD (Point Cloud Data)
"""
#Get point cloud data (PCD)
dino_data = np.load("DinoVisalizationData_avgnormals.npy", allow_pickle=True)
points = dino_data[0]
colors = dino_data[1]
normals_sign = dino_data[2]

#Create a KDTree with the point cloud
tree = KDTree(points)

"""
    Calculate r for each point in point cloud
"""

#Naive implementation of indices
r = 0.1
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
    if np.dot(normals[i], normals_sign[i]) < 0:
        normals[i] = normals[i]*-1.0

"""
    Plot data
"""

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:,0], points[:,1], points[:,2], 'o', color=colors)

#Plot vector for normal
ax.quiver(
        points[:,0], points[:,1], points[:,2], # <-- starting point of vector
        normals[:,0], normals[:,1], normals[:,2], # <-- directions of vector
        arrow_length_ratio=0.001, color = colors, alpha = 0.3, lw = 1
    )
plt.show()

"""
    Print to file
"""
#Print points, reflectance, colors and normals
reflectance = np.ones([points.shape[0],1])
result = np.append(points, reflectance, axis=1)
result = np.append(result, colors, axis=1)
result = np.append(result, normals, axis=1)

#Save to file
np.savetxt("dino_normals.txt", result, delimiter=' ')
print("Woop! Printed results to file :^)")
