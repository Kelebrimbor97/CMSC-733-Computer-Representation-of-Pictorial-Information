import numpy as np
import scipy.optimize as opt


def minimize_reproj_error(init, K, C1, R1, C2, R2, x1, x2):
    X = np.reshape(init, (x1.shape[0], 3))
    C2 = np.reshape(C2, (3,1))
    # make homogenous
    X = np.hstack((X, np.ones((x1.shape[0], 1))))
    # defining camera projection matrix
    P1 = np.dot(K, np.dot(R1, np.hstack((np.identity(3), -C1))))
    P2 = np.dot(K, np.dot(R2, np.hstack((np.identity(3), -C2))))
    error1 = 0
    error2 = 0
    error = []
    # calculating reprojection error
    u_reproj1 = np.divide((np.dot(P1[0, :], X.T).T), (np.dot(P1[2, :], X.T).T))
    v_reproj1 = np.divide((np.dot(P1[1, :], X.T).T), (np.dot(P1[2, :], X.T).T))
    u_reproj2 = np.divide((np.dot(P2[0, :], X.T).T), (np.dot(P2[2, :], X.T).T))
    v_reproj2 = np.divide((np.dot(P2[1, :], X.T).T), (np.dot(P2[2, :], X.T).T)) 

    error1 = ((x1[:, 0] - u_reproj1) + (x1[:, 1] - v_reproj2))
    error2 = ((x2[:, 0] - u_reproj2) + (x2[:, 1] - v_reproj2))
    error = sum(error1, error2)

    return sum(error)

def non_linear_triangulation(K, C1, R1, C2, R2, x1, x2, X_init):
    X = np.zeros((x1.shape[0], 3))
    init = X_init.flatten()
    params = opt.least_squares(fun=minimize_reproj_error,x0=init,method="dogbox",args=[K, C1, R1, C2, R2, x1, x2 ])

    X = np.reshape(params.x, (x1.shape[0], 3))

    return X

