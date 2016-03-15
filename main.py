from scipy.io import loadmat
from scipy.sparse import csr_matrix, dia_matrix
from scipy.sparse.linalg import eigsh, inv
import numpy as np
from numpy import cross, arccos, array
from numpy.linalg import norm
from mayavi import mlab
import matplotlib.pyplot as plt
import os
import time
from sklearn.neighbors import BallTree
from lib_rigid_ICP import compute_best_rigid_deformation

start = time.time()
#%% FUNCTIONS
def load_mesh(filename):
    surface = loadmat(filename)
    x,y,z,tri = surface['surface']['X'][0][0],surface['surface']['Y'][0][0],surface['surface']['Z'][0][0],surface['surface']['TRIV'][0][0]
    tri -= 1
    ver = np.c_[x,y,z]
    return tri, ver

def angle(e1,e2):
    return arccos((e1*e2).sum(axis=1)/(norm(e1,axis=1)*norm(e2,axis=1)))

def get_matM(tri,ver):
    v1 = ver[tri[:,0],:]
    v2 = ver[tri[:,1],:]
    v3 = ver[tri[:,2],:]        
    e1 = v2-v1
    e3 = v1-v3    
    
    norm_l = cross(e1,-e3,axis=1)
    norm_mag = norm(norm_l,axis=1)
#    norm_1 = norm_l/norm_mag[:,None]
    area = norm_mag/2
    
    data = (area/3).repeat(3)
    row_ind = col_ind = tri.flatten()
    M = N = ver.shape[0]
    matM = csr_matrix((data,(row_ind,col_ind)),shape=(M,N))
    return matM
    
def get_matC(tri,ver):
    v1 = ver[tri[:,0],:]
    v2 = ver[tri[:,1],:]
    v3 = ver[tri[:,2],:]        
    e1 = v2-v1
    e2 = v3-v2
    e3 = v1-v3  
    
    hcot_12 = 0.5/np.tan(angle(-e2,e3))
    hcot_13 = 0.5/np.tan(angle(-e1,e2))
    hcot_23 = 0.5/np.tan(angle(-e3,e1))
    
    data = array([hcot_12,hcot_23,hcot_13]).flatten()
    row_ind = array([tri[:,0],tri[:,1],tri[:,2]]).flatten()
    col_ind = array([tri[:,1],tri[:,2],tri[:,0]]).flatten()
    M = N = ver.shape[0]
    matC = csr_matrix((data,(row_ind,col_ind)),shape=(M,N))
    matC = matC + matC.T
    dia = -matC.sum(axis=1).A.flatten()
    matC = matC + dia_matrix((dia,0), shape=(M,N))
    return matC

def get_eigen(matM,matC,n):
    eival, eivec = eigsh(matC,n,sigma=-1e-8,M=matM)
    return eival,eivec

def get_hks(eival,eivec,tri,ver,times):
    vertex_areas = get_matM(tri, ver).data
    k = np.zeros((eivec.shape[0],len(times)))
    for idx,t in enumerate(times):
        k[:,idx] = (np.exp(-t*eival)[None,:]*eivec*eivec).sum(axis=1)
    average_temperature = (vertex_areas[:,None]*k).sum(axis=0)/vertex_areas.sum()
    hks = k/average_temperature
    return hks


def get_wks(eigen_vectors, eigen_values, energy_steps=None, absolute_sigma=None, num_steps=None, relative_sigma=None):
    eigen_values = np.abs(eigen_values)
    idx = np.argsort(eigen_values)
    eigen_values = eigen_values[idx[1:]]
    eigen_vectors = eigen_vectors[:, idx[1:]]

    if energy_steps is None:
        assert num_steps is not None
        energy_steps = np.log(np.linspace(eigen_values[1], eigen_values[-1], num_steps))

    if absolute_sigma is None:
        if relative_sigma is not None:
            absolute_sigma = (energy_steps.max() - energy_steps.min()) * relative_sigma
        else:
            # from paper
            delta = (energy_steps.max() - energy_steps.min()) / energy_steps.size
            absolute_sigma = 7 * delta

    nv = eigen_vectors.shape[0]
#    nev = eigen_vectors.shape[1]
    num_steps = energy_steps.size

    desc = np.zeros((nv, num_steps))
    for idx, e in enumerate(energy_steps):

        coeff = np.exp(-(e-np.log(eigen_values))**2/(2*absolute_sigma))
        desc[:, idx] = 1/coeff.sum() * (eigen_vectors**2).dot(coeff)

    return desc

def get_coef(eivec,matM,hks):
    coef = eivec.T.dot(matM.dot(hks))
    return coef

def get_funcMap(coefS,coefT,eivalS,eivalT):
    k = coefS.shape[0]
    t = coefS.shape[1]
    w_func = 1
    w_com = 1
    w_reg = 1

    data_1 = (coefS.T.repeat(k,axis=0).flatten())*w_func
    row_ind_1 = np.arange(k*t).repeat(k)
    col_ind_1 = np.tile(np.arange(k*k),t)    
    
    lS = np.tile(eivalS,(eivalS.shape[0],1))    
    lT = np.tile(eivalT,(eivalS.shape[0],1)).T
    matEival = (lT-lS) * (lT-lS)
    data_2 = (matEival.T.repeat(k,axis=0).flatten())*w_com
    row_ind_2 = (np.arange(k*k)+(k*t)).repeat(k)
    col_ind_2 = np.tile(np.arange(k*k),k)
        
    data_3 = (np.ones(k*k))*w_reg
    row_ind_3 = np.arange(k*k)+((k*t)+(k*k))
    col_ind_3 = np.arange(k*k)
    
    M = (k*t)+(k*k)+(k*k)
    N = k*k
    data = np.concatenate((data_1,data_2,data_3),axis=0)
    row_ind = np.concatenate((row_ind_1,row_ind_2,row_ind_3),axis=0)
    col_ind = np.concatenate((col_ind_1,col_ind_2,col_ind_3),axis=0)
    a = csr_matrix((data,(row_ind,col_ind)),shape=(M,N))
    b1 = (coefT.T.flatten())*w_func
    b2 = (np.zeros(k*k))*w_com
    b3 = (np.zeros(k*k))*w_reg
    b = (np.concatenate((b1,b2,b3),axis=0))[:,None]
    aTa = a.T.dot(a)
    aTb = a.T.dot(b)
    funcMap = (inv(aTa).dot(aTb)).reshape(k,k)
    return funcMap
    
#def get_funcMap(coefhS,coefhT,coefwS,coefwT,eivalS,eivalT):
#    k = coefhS.shape[0]
#    t = coefhS.shape[1]
#    w_hks = 1
#    w_wks = 1
#    w_com = 1
#    w_reg = 1
#
#    data_1 = (coefhS.T.repeat(k,axis=0).flatten())*w_hks
#    row_ind_1 = np.arange(k*t).repeat(k)
#    col_ind_1 = np.tile(np.arange(k*k),t)    
#    
#    data_2 = (coefwS.T.repeat(k,axis=0).flatten())*w_wks
#    row_ind_2 = np.arange(k*t).repeat(k)
#    col_ind_2 = np.tile(np.arange(k*k),t)    
#    
#    lS = np.tile(eivalS,(eivalS.shape[0],1))    
#    lT = np.tile(eivalT,(eivalS.shape[0],1)).T
#    matEival = (lT-lS) * (lT-lS)
#    data_3 = (matEival.T.repeat(k,axis=0).flatten())*w_com
#    row_ind_3 = (np.arange(k*k)+(k*t)).repeat(k)
#    col_ind_3 = np.tile(np.arange(k*k),k)
#        
#    data_4 = (np.ones(k*k))*w_reg
#    row_ind_4 = np.arange(k*k)+((k*t)+(k*k))
#    col_ind_4 = np.arange(k*k)
#    
#    M = (k*t)+(k*t)+(k*k)+(k*k)
#    N = k*k
#    data = np.concatenate((data_1,data_2,data_3,data_4),axis=0)
#    row_ind = np.concatenate((row_ind_1,row_ind_2,row_ind_3,row_ind_4),axis=0)
#    col_ind = np.concatenate((col_ind_1,col_ind_2,col_ind_3,col_ind_4),axis=0)
#    a = csr_matrix((data,(row_ind,col_ind)),shape=(M,N))
#    b1 = (coefhT.T.flatten())*w_hks
#    b2 = (coefwT.T.flatten())*w_wks
#    b3 = (np.zeros(k*k))*w_com
#    b4 = (np.zeros(k*k))*w_reg
#    b = (np.concatenate((b1,b2,b3,b4),axis=0))[:,None]
#    aTa = a.T.dot(a)
#    aTb = a.T.dot(b)
#    funcMap = (inv(aTa).dot(aTb)).reshape(k,k)
#    return funcMap

def extract_mapping_original(F, evecs_from, evecs_to):
    bt_ = BallTree(F.dot(evecs_from.T).T)
    dists, others = bt_.query(evecs_to)
    return others.flatten()
    
def plot(tri, ver, scalars=None,vectors=None):
    mlab.figure()
    plot1 = mlab.triangular_mesh(ver[:,0],ver[:,1],ver[:,2],tri)
    
    if scalars is not None:
        plot1.mlab_source.scalars = scalars.flatten()
    if vectors is not None:
        if vectors.shape[0] == ver.shape[0]:
            center = ver
        elif vectors.shape[0] == tri.shape[0]:
            v1 = ver[tri[:,0],:]
            v2 = ver[tri[:,1],:]
            v3 = ver[tri[:,2],:]   
            center = (v1+v2+v3)/3
        vecs = vectors
        mlab.quiver3d(center[:,0],center[:,1],center[:,2],vecs[:,0],vecs[:,1],vecs[:,2])

#%% PROGRAM
base_path = os.path.expanduser(os.environ['CGPRAK'])
source = os.path.join(base_path,'Tosca','hi-res','cat1.mat')
target = os.path.join(base_path,'Tosca','hi-res','cat0.mat')

triS, verS = load_mesh(source)
triT, verT = load_mesh(target)

#%% 1. LapBel Operator & Basis (eigenfunc)
matMS = get_matM(triS,verS)
matCS = get_matC(triS,verS)
matMT = get_matM(triT,verT)
matCT = get_matC(triT,verT)

n_eigen = 30
eivalS,eivecS = get_eigen(matMS,matCS,n_eigen)
eivalT,eivecT = get_eigen(matMT,matCT,n_eigen)
eivalS = eivalS[:-1]
eivecS = eivecS[:,:-1]
eivalT = eivalT[:-1]
eivecT = eivecT[:,:-1]

#flip eigen after turn it to positive
#eivalS = (np.fliplr(eivalS[None,:]))[0]
#eivecS = np.fliplr(eivecS)
#eivalT = (np.fliplr(eivalT[None,:]))[0]
#eivecT = np.fliplr(eivecT)

#%% 2. Function Representation (HKS & WKS) & its Coefficient
times = np.logspace(np.log(0.1),np.log(10),num=100)
hksS = get_hks(eivalS,eivecS,triS,verS,times)
hksT = get_hks(eivalT,eivecT,triT,verT,times)
wksS = get_wks(eivecS, eivalS,num_steps=100)
wksT = get_wks(eivecT, eivalT,num_steps=100)
#plot(triS,verS,scalars=hksS[:,0])
#plot(triS,verS,scalars=wksS[:,0])

coefhS = get_coef(eivecS,matMS,hksS)
coefhT = get_coef(eivecT,matMT,hksT)
coefwS = get_coef(eivecS,matMS,wksS)
coefwT = get_coef(eivecT,matMT,wksT)

funcMap = get_funcMap(coefhS,coefhT,eivalS,eivalT)
#funcMap = get_funcMap(coefhS,coefhT,coefwS,coefwT,eivalS,eivalT)

eivecS_mapped = (funcMap.dot(eivecS.T)).T
from_mean, to_mean, rot, points_from_deformed = compute_best_rigid_deformation(eivecS_mapped,eivecT,eivecS,eivecT)
corrsS = points_from_deformed
corrsT = eivecT

for i in range(0,8):
    mapped_from_to_to = extract_mapping_original(funcMap,corrsS, corrsT)
    corrsS = points_from_deformed[mapped_from_to_to,:]
    _, _, funcMap, _ = compute_best_rigid_deformation(corrsS,corrsT,eivecS,eivecT)
    print i

base_ver = 2000
delta = np.zeros(verS.shape[0])
delta[base_ver] = 1
coefS_delta = eivecS.T.dot(delta)
coefT_delta = funcMap.dot(coefS_delta)
recS_delta = eivecS.dot(coefS_delta)
recT_delta = eivecT.dot(coefT_delta)
plot(triS,verS,scalars=recS_delta)
plot(triT,verT,scalars=recT_delta)
plot(triT,verT,scalars=delta)

plot(triT,verT,scalars=recS_delta[mapped_from_to_to])
plot(triS,verS,scalars=hksS[:,0])
plot(triT,verT,scalars=(hksS[:,0])[mapped_from_to_to])

end = time.time()
print(end-start)