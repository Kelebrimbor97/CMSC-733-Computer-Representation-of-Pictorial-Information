import numpy as np
import scipy.optimize as opt
from scipy.spatial.transform import Rotation as Rscipy


def reprojection_error(CQ, K, X, x):
   
    X = np.hstack((X, np.ones((X.shape[0], 1))))

    C = CQ[0:3]
    R = CQ[3:7]
    C = C.reshape(-1, 1)
    r = Rscipy.from_quat([R[0], R[1], R[2], R[3]])
    R = r.as_matrix()

    P = np.dot(np.dot(K, R), np.hstack((np.identity(3), -C)))
    u = (np.dot(P[0, :], X.T)).T / (np.dot(P[2, :], X.T)).T
    v = (np.dot(P[1, :], X.T)).T / (np.dot(P[2, :], X.T)).T
    error_1 = x[:, 0] - u
    error_2 = x[:, 1] - v
    total_e = error_1 + error_2

    return sum(total_e)


def NonLinearPnP(X, x, K, C0, R0):

    q_temp = Rscipy.from_matrix(R0)
    Q0 = q_temp.as_quat()
    CQ = [C0[0], C0[1], C0[2], Q0[0], Q0[1], Q0[2], Q0[3]]
    optimized_param = opt.least_squares(fun=reprojection_error, method="dogbox", x0=CQ, args=[K, X, x])
    Cnew = optimized_param.x[0:3]
    R = optimized_param.x[3:7]
    r = Rscipy.from_quat([R[0], R[1], R[2], R[3]])
    Rnew = r.as_matrix()

    return Cnew, Rnew
