import numpy as np

def linear_triangulation(K, R1, R2, C1, C2, x1, x2):
    X = []
    #By general mapping of a pinhole camera
    I = np.eye(3)
    C1 = C1.reshape((3,1))
    C2 = C2.reshape((3,1))
    P1 = K.dot(R1.dot(np.hstack((I, -C1))))
    P2 = K.dot(R2.dot(np.hstack((I, -C2))))

    p1T = P1[0,:].reshape(1,4)
    p2T = P1[1,:].reshape(1,4)
    p3T = P1[2,:].reshape(1,4)

    p_1T = P2[0,:].reshape(1,4)
    p_2T = P2[1,:].reshape(1,4)
    p_3T = P2[2,:].reshape(1,4)

    # constructing the A matrix of AX = 0
    for i in range(x1.shape[0]):
        x = x1[i,0]
        y = x1[i,1]
        x_dash = x2[i,0]
        y_dash = x2[i,1]
        A = [(y * p3T) -  p2T, p1T -  (x * p3T), (y_dash * p_3T) -  p_2T, p_1T -  (x_dash * p_3T) ]
        A = np.array(A).reshape(4,4)
        # Solving for A_triangulation*X = 0
        U , S, Vt = np.linalg.svd(A)
        x = x = Vt[-1] / Vt[-1, -1]
        X.append(x)
    return np.array(X)