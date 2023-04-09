import numpy as np

def estimate_camera_pose(E):
    U, D, Vt = np.linalg.svd(E)
    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])
    C = []
    R = []
    C.append(U[:,2])
    C.append(-U[:,2])
    C.append(U[:,2])
    C.append(-U[:,2])
    R.append(U.dot(W.dot(Vt)))
    R.append(U.dot(W.dot(Vt)))
    R.append(U.dot(W.T.dot(Vt)))
    R.append(U.dot(W.T.dot(Vt)))
    # correct the camera pose
    for i in range(len(R)):
        if(np.linalg.det(R[i]) < 0):
            R[i] = -R[i]
            C[i] = -C[i]

    return C, R