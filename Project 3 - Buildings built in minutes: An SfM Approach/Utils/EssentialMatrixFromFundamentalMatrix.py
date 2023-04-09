import numpy as np

def compute_essential_matrix(f,k1):
    k1_t = k1.T
    e = np.dot(k1_t,np.dot(f,k1))
    U,S,Vt = np.linalg.svd(e)
    S = [1,1,0]
    E = np.dot(U,np.dot(np.diag(S),Vt))
    return E