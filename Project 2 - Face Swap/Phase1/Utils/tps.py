import cv2
from cv2 import fillPoly
import numpy as np

def U(r):
    return r**2 * (np.log10(r**2))

def TPS(features_list_target, features_list_source, target, source, mask):
    # computing K matrix
    p = len(features_list_target)
    Lambda = 1e-4
    K  = np.zeros((p,p))
    for i in range(p):
        for j in range(p):
            K[i,j] = U(np.linalg.norm(features_list_target[i] - features_list_target[j]))
    K = np.nan_to_num(K)

    # compute P and V matrix
    P = np.ones((p,3))
    V_x = np.zeros(p+3)
    V_y = np.zeros(p+3)
    for i, (x,y) in enumerate(features_list_target):
        P[i] = (x,y,1)
        

    for i, (x,y) in enumerate(features_list_source):
        V_x[i] = x
        V_y[i] = y

    V_x = np.reshape(V_x, (p+3,1))
    V_y = np.reshape(V_y, (p+3,1))

    # compute final matrix
    final = np.hstack((K,P))
    final = np.vstack((final,np.hstack((P.T, np.zeros((3,3))))))
    final = final + Lambda*np.identity(p+3)

    # finding spline for x -coordinates
    spline_x = np.linalg.inv(final).dot(V_x)
    spline_y = np.linalg.inv(final).dot(V_y)
    w_x = spline_x[0:p]
    w_y = spline_y[0:p]
    ax_x = spline_x[p]
    ay_x = spline_x[p+1]
    a1_x = spline_x[p+2]
    ax_y = spline_y[p]
    ay_y = spline_y[p+1]
    a1_y = spline_y[p+2] 
 
    
    # cv2.imshow('mask',mask)
    # cv2.waitKey(0)
    face_index_x = np.where(mask == 255)[1]
    face_index_y = np.where(mask == 255)[0]
    pts =  np.vstack((face_index_x, face_index_y)).T
    points = np.array(features_list_target, np.int32)
    for point in pts:
        u = (points - point)
        u = np.linalg.norm(u, ord=2, axis=1)
        zero_index = np.where(u == 0)
        u[zero_index] = 1e-4
        u = U(u)
        u = np.nan_to_num(u)
        wUX = np.matmul(np.transpose(w_x), u)
        wUY = np.matmul(np.transpose(w_y), u)
        fX = int(a1_x + ax_x*point[0] + ay_x*point[1] + wUX)
        fY = int(a1_y + ax_y*point[0] + ay_y*point[1] + wUY)
        target[point[1]][point[0]] = source[fY][fX]

    return target