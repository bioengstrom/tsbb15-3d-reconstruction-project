import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Get point cloud data (PCD)
points = np.genfromtxt("stanfordbunny.txt", delimiter=" ")

#Create a KDTree with the point cloud
tree = KDTree(points)

#Look for the n nearest neighbours points in a sphere with r radius
r = 0.01
index = tree.query_radius(points[0:1], r)  # NNs within distance of r of point

# Plot selected points
pi = points[index[0]]
not_selected_points = np.delete(points, index[0], axis=0)
plt.figure()
ax = plt.subplot(111, projection='3d')
ax.scatter(not_selected_points[:,0], not_selected_points[:,1], not_selected_points[:,2], c='b')
ax.scatter(pi[:,0], pi[:,1], pi[:,2], c='r')
plt.show()

#Compute normal from points with help of eigen values
k = pi.shape[0]
p_bar = (1/k)*np.sum(pi, axis=0) #TODO: Check so that outer product makes sense and compute M
print(pi[:2])
print(pi[:2]*pi[:2,None])

M = (1/k)*np.sum((pi*pi[:,None])-(p_bar*p_bar[:,None]),axis=0)
print(M.shape)
