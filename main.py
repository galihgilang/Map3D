# -*- coding: utf-8 -*-

from scipy import io, sparse
import numpy as np
from mayavi import mlab
import os
import math

# set base_path and filename
base_path = os.path.expanduser(os.environ['CGPRAK'])
filename = os.path.join(base_path,'Tosca','non-rigid','cat1.mat')

def load_mesh(filename):
    surface = io.loadmat(filename)
    x,y,z,triv = surface['surface']['X'][0][0],surface['surface']['Y'][0][0],surface['surface']['Z'][0][0],surface['surface']['TRIV'][0][0]
    points = np.c_[x,y,z]   # points = vertices
    return points, triv-1   # triv = triangle

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
    
vert,triv = load_mesh(filename)

# properties per vertices
v1s = vert[triv[:,0],:]
v2s = vert[triv[:,1],:]
v3s = vert[triv[:,2],:]
center = (v1s+v2s+v3s)/3

e1s = v2s-v1s
e2s = v3s-v2s
e3s = v1s-v3s

norm_l = np.cross(e1s,-e3s,axis=1)
norm_mag = np.linalg.norm(norm_l,axis=1)
norm = norm_l/norm_mag[:,None]
triangle_area = norm_mag/2

#mlab.triangular_mesh(vert[:,0],vert[:,1],vert[:,2],triv)
#mlab.quiver3d(center[:,0],center[:,1],center[:,2],norm[:,0],norm[:,1],norm[:,2])

n_vert = vert.shape[0]
n_triv = triv.shape[0]

# matrix M
mat_M = np.zeros((n_vert,n_vert))
for i in range(n_vert):
    mat_M[i][i] = np.sum(triangle_area[np.argwhere(triv==i)[:,0]])/3

# matrix M-1
mat_M1 = np.zeros((n_vert,n_vert))
for i in range(n_vert):
    mat_M1[i][i] = 1/(np.sum(triangle_area[np.argwhere(triv==i)[:,0]])/3)
    
# matrix C
mat_C = np.zeros((n_vert,n_vert))
for j in range(n_triv):
    mat_C[triv[j][0]][triv[j][1]] += (1/math.tan(angle(-e2s[j],e3s[j])))/2
    mat_C[triv[j][1]][triv[j][0]] += (1/math.tan(angle(-e2s[j],e3s[j])))/2
    mat_C[triv[j][0]][triv[j][2]] += (1/math.tan(angle(-e1s[j],e3s[j])))/2
    mat_C[triv[j][2]][triv[j][0]] += (1/math.tan(angle(-e1s[j],e3s[j])))/2
    mat_C[triv[j][1]][triv[j][2]] += (1/math.tan(angle(-e3s[j],e1s[j])))/2
    mat_C[triv[j][2]][triv[j][1]] += (1/math.tan(angle(-e3s[j],e1s[j])))/2
    
for k in range(n_vert):
    mat_C[k][k] = -(np.sum(mat_C[k]))
    
# matrix L
mat_L = np.dot(mat_M1,mat_C)

# w,v is eigenvalue & eigenvector of L
w,v = np.linalg.eig(mat_L)
