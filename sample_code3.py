# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:09:00 2015

@author: alex
"""
from scipy import io,sparse
from scipy.sparse.linalg import eigsh
import numpy as np
from mayavi import mlab
import matplotlib.pylab as plt
from sklearn.neighbors import BallTree

import os
import time

start = time.time()
#base_path = os.path.expanduser(os.environ['CGPRAK'])
base_path = os.path.expanduser(os.environ['CGPRAK'])
source = os.path.join(base_path,'Tosca','hi-res','cat0.mat')
target = os.path.join(base_path,'Tosca','hi-res','cat1.mat')

def load_mesh(filename):

    surface = io.loadmat(filename)

    x,y,z,tris = surface['surface']['X'][0][0],surface['surface']['Y'][0][0],surface['surface']['Z'][0][0],surface['surface']['TRIV'][0][0]

    points = np.c_[x,y,z]
    return tris-1, points

def calc_tris_normals(tris, points):
    v1s = points[tris[:,0],:]
    v2s = points[tris[:,1],:]
    v3s = points[tris[:,2],:]

    e1s = v2s-v1s
    e2s = v3s-v2s
    e3s = v1s-v3s

    ns = np.cross(e1s,-e3s,axis=1)
    center = (v1s+v2s+v3s)/3

    triangle_area = np.linalg.norm(ns,axis=1)

    norm_ns = ns/triangle_area[:,None]
    return norm_ns, triangle_area, center

def calc_vertex_normals(tris, points):
    norm_ns, triangle_area, center = calc_tris_normals(tris, points)

    row_ind = tris.flatten()
    col_ind = np.arange(len(tris)).repeat(3)
    M = np.max(row_ind)+1
    N = np.max(col_ind)+1
    data = np.ones(len(row_ind))
    D = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(M,N))
    degree = D.sum(axis=1).A.flatten()

    diag = sparse.spdiags(1./degree,0,M,M)
    op = diag.dot(D)

    return op.dot(norm_ns)

def plot(tris, points, scalars=None,vectors=None):
    mlab.figure()
    plot1 = mlab.triangular_mesh(points[:,0],points[:,1],points[:,2],tris)

    if scalars is not None:
        plot1.mlab_source.scalars = scalars.flatten()
    if vectors is not None:
        if vectors.shape[0] == points.shape[0]:
            center = points
        elif vectors.shape[0] == tris.shape[0]:
            v1s = points[tris[:,0],:]
            v2s = points[tris[:,1],:]
            v3s = points[tris[:,2],:]

            center = (v1s+v2s+v3s)/3
        vecs = vectors
        mlab.quiver3d(center[:,0],center[:,1],center[:,2],vecs[:,0],vecs[:,1],vecs[:,2])

def get_vertex_area_matrix(tris, points):
    v1s = points[tris[:,0],:]
    v2s = points[tris[:,1],:]
    v3s = points[tris[:,2],:]

    e1s = v2s-v1s
    e2s = v3s-v2s
    e3s = v1s-v3s

    tris_normals = np.cross(e1s,-e2s)
    triangle_areas = np.linalg.norm(tris_normals,axis=1)/2.0

    M = points.shape[0]
    N = points.shape[0]
    data = (triangle_areas/3.0).repeat(3)
    row_ind = tris.flatten()
    col_ind = tris.flatten()
    A = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(M, N))
    return A.data

def normalize_vectors(vectors):
    length_vectors = np.linalg.norm(vectors,axis=1)
    return vectors/length_vectors[:,None]

def cotangent(angles):
    return np.cos(angles)/np.sin(angles)

def get_cotan_Laplacian(tris,points):
    v1s = points[tris[:,0],:]
    v2s = points[tris[:,1],:]
    v3s = points[tris[:,2],:]

    e1s = normalize_vectors(v2s-v1s)
    e2s = normalize_vectors(v3s-v2s)
    e3s = normalize_vectors(v1s-v3s)



    alphas = np.arccos((e1s*(-e3s)).sum(axis=1))
    betas = np.arccos((e2s*(-e1s)).sum(axis=1))
    gammas = np.arccos((e3s*(-e2s)).sum(axis=1))

    data = 0.5*cotangent([gammas,alphas,betas]).flatten()
    row_ind = np.array([tris[:,0],tris[:,1],tris[:,2]]).flatten()
    col_ind = np.array([tris[:,1],tris[:,2],tris[:,0]]).flatten()
    M = points.shape[0]
    N = points.shape[0]
    Cotan = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(M, N))
    Cotan = 0.5*(Cotan+Cotan.T)
    diagonal = Cotan.sum(axis=1).A.flatten()
    Cotan = Cotan-sparse.spdiags(diagonal,0,M,N)
    vertex_areas = get_vertex_area_matrix(tris, points)
    M = sparse.spdiags(vertex_areas,0,M,N)
    return M, Cotan

def heat_kernel_siugnature(evals,evecs,tris,points,times):
    vertex_areas = get_vertex_area_matrix(tris, points)
    k = np.zeros((evecs.shape[0],len(times)))
    for idx,t in enumerate(times):
        k[:,idx] = (np.exp(-t*evals)[None,:]*evecs*evecs).sum(axis=1)
    average_temperature = (vertex_areas[:,None]*k).sum(axis=0)/vertex_areas.sum()
    k = k/average_temperature
    return k

def get_coef(evecs,hks):
    coef = np.dot(evecs.T,MS.dot(hks))
    return coef

def get_funcMap(a,b):
    baT = b.dot(a.T)
    aaT = np.dot(a,a.T)
    inv_aaT = np.linalg.inv(aaT)
    funcMap = baT.dot(inv_aaT)
    return funcMap


from lib_rigid_ICP import compute_best_rigid_deformation

def extract_mapping_original(F, evecs_from, evecs_to):
    bt_ = BallTree(F.dot(evecs_from.T).T)
    dists, others = bt_.query(evecs_to)
    return others.flatten()
#%%


trisS, pointsS = load_mesh(source)
trisT, pointsT = load_mesh(target)

norm_nsS, triangle_areaS, centerS = calc_tris_normals(trisS, pointsS)
vertex_areasS = get_vertex_area_matrix(trisS, pointsS)
norm_nsT, triangle_areaT, centerT = calc_tris_normals(trisT, pointsT)
vertex_areasT = get_vertex_area_matrix(trisT, pointsT)

MS, CotanS = get_cotan_Laplacian(trisS,pointsS)
MT, CotanT = get_cotan_Laplacian(trisT,pointsT)

evalsS, evecsS = eigsh(CotanS,50,sigma=-1e-8,M=MS)
evalsT, evecsT = eigsh(CotanT,50,sigma=-1e-8,M=MT)

times = np.logspace(np.log(0.1),np.log(10),num=100)
hksS = heat_kernel_siugnature(evalsS[1:],evecsS[:,1:],trisS,pointsS,-times)
hksT = heat_kernel_siugnature(evalsT[1:],evecsT[:,1:],trisT,pointsT,-times)

coefS = get_coef(evecsS[:,1:],hksS)
coefT = get_coef(evecsT[:,1:],hksT)
funcMap = get_funcMap(coefS,coefT)

evecsS_mapped = (funcMap.dot(evecsS[:,1:].T)).T

from_mean, to_mean, rot, points_from_deformed = compute_best_rigid_deformation(evecsS_mapped,evecsT[:,1:],evecsS[:,1:],evecsT[:,1:])

deltaS = np.zeros(pointsS.shape[0])
deltaS[0] = 1
coefS_delta = evecsS[:,1:].T.dot(deltaS)

rec_deltaS = evecsS[:,1:].dot(coefS_delta)

plot(trisS,pointsS,scalars=rec_deltaS)

plot(trisT,pointsT,scalars=evecsT[:,1:].dot(funcMap.dot(coefS_delta)))

corrsS = points_from_deformed
corrsT = evecsT[:,1:]

for i in range(0,4):
    mapped_from_to_to = extract_mapping_original(funcMap,corrsS, corrsT)
    corrsS = points_from_deformed[mapped_from_to_to,:]
    _, _, funcMap, _ = compute_best_rigid_deformation(corrsS,corrsT,evecsS[:,1:],evecsT[:,1:])
    print i

plot(trisT,pointsT,scalars=rec_deltaS[mapped_from_to_to])
#%%
#diff = np.linalg.norm(hks[0,:][None,:]-hks,axis=1)
#where = np.zeros(points.shape[0])
#where[0] = 1
if False:
    plot(trisS,pointsS,scalars=hksS[:,0])
    plot(trisT,pointsT,scalars=evecsT.dot(funcMap.dot(coefS[:,0])))

    end = time.time()
    print(end-start)