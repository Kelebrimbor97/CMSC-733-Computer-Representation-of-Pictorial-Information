import numpy as np
import random
from tqdm import tqdm

def homogeneouos(x):
    m, n = x.shape
    if (n == 3 or n == 2):
        x_new = np.hstack((x, np.ones((m, 1))))
    else:
        x_new = x
    return x_new

def LinearPnP(X, x, K):
    N = X.shape[0]
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    x = np.hstack((x, np.ones((x.shape[0], 1))))

    x = np.transpose(np.dot(np.linalg.inv(K), x.T))
    A = []
    for i in range(N):
        xt = X[i, :].reshape((1, 4))
        z = np.zeros((1, 4))
        p = x[i, :]

        a1 = np.hstack((np.hstack((z, -xt)), p[1] * xt))
        a2 = np.hstack((np.hstack((xt, z)), -p[0] * xt))
        a3 = np.hstack((np.hstack((-p[1] * xt, p[0] * xt)), z))
        a = np.vstack((np.vstack((a1, a2)), a3))

        if (i == 0):
            A = a
        else:
            A = np.vstack((A, a))

    _, _, v = np.linalg.svd(A)
    P = v[-1].reshape((3, 4))
    R = P[:, 0:3]
    t = P[:, 3]
    u, _, v = np.linalg.svd(R)

    R = np.matmul(u, v)
    d = np.identity(3)
    d[2][2] = np.linalg.det(np.matmul(u, v))
    R = np.dot(np.dot(u, d), v)
    C = -np.dot(np.linalg.inv(R), t)
    if np.linalg.det(R) < 0:
        R = -R
        C = -C
    return C, R

def proj_3D_to_2D(x3D, K, C, R):
 
    C = C.reshape(-1, 1)
    x3D = x3D.reshape(-1, 1)
    P = np.dot(np.dot(K, R), np.hstack((np.identity(3), -C)))
    X3D = np.vstack((x3D, 1))

    u_rprj = (np.dot(P[0, :], X3D)).T / (np.dot(P[2, :], X3D)).T
    v_rprj = (np.dot(P[1, :], X3D)).T / (np.dot(P[2, :], X3D)).T
    X2D = np.hstack((u_rprj, v_rprj))
    return X2D


def PnPRANSAC(X, x, K):

    count = 0
    M = x.shape[0]
    threshold = 5
    x_ = homogeneouos(x) # convert to homogenous coordinates

    Cnew = np.zeros((3, 1))
    Rnew = np.identity(3)

    for t in tqdm(range(500)):
        random_idx = random.sample(range(M), 6)
        C, R = LinearPnP(X[random_idx][:], x[random_idx][:], K)
        S = []
        for j in range(M):
            reprojection = proj_3D_to_2D(x_[j][:], K, C, R)
            e = np.sqrt(
                np.square((x_[j, 0]) - reprojection[0]) + np.square((x_[j, 1] - reprojection[1])))
            if e < threshold:
                S.append(j)
        countS = len(S)
        if (count < countS):
            count = countS
            Rnew = R
            Cnew = C

        if (countS == M):
            break
    return Cnew, Rnew
