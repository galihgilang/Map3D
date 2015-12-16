# -*- coding: utf-8 -*-

from scipy import io, sparse
import numpy as np
from mayavi import mlab

import os
base_path = os.path.expanduser(os.environ['CGprak'])
filename = os.path.join(base_path,'tosca_hires','cat0.mat')
    
def load_mesh(filename):

    surface = io.loadmat(filename)

    x,y,z,triv = surface['surface']['X'][0][0],surface['surface']['Y'][0][0],surface['surface']['Z'][0][0],surface['surface']['TRIV'][0][0]

    points = np.c_[x,y,z]
    return points, triv-1
    
    
points,triv = load_mesh(filename)

v1s = points[triv[:,0],:]
v2s = points[triv[:,1],:]
v3s = points[triv[:,2],:]

e1s = v2s-v1s
e2s = v3s-v2s
e3s = v1s-v3s

ns = np.cross(e1s,-e3s,axis=1)
center = (v1s+v2s+v3s)/3

triangle_area = np.linalg.norm(ns,axis=1)

norm_ns = ns/triangle_area[:,None]

mlab.triangular_mesh(points[:,0],points[:,1],points[:,2],triv)
mlab.quiver3d(center[:,0],center[:,1],center[:,2],norm_ns[:,0],norm_ns[:,1],norm_ns[:,2])
