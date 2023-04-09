import numpy as np
import cv2


def construct_A(p1,p2): 
    A = []
    for i in range(p1.shape[0]):
        row = [p1[i,0]*p2[i,0], p1[i,0]*p2[i,1], p1[i,0], p1[i,1]*p2[i,0], p1[i,1]*p2[i,1], p1[i,1], p2[i,0], p2[i,1], 1]
        A.append(row) 
    return A

def compute_fundamental_matrix(A):
    U, S, Vt = np.linalg.svd(np.array(A))
    f= Vt[np.argmin(S)]
    F = np.array([[f[0], f[1],f[2]],
                    [f[3],f[4],f[5]],
                    [f[6],f[7],f[8]]])
    # enforcing rank 2 constraint
    u ,s, vt = np.linalg.svd(F)
    s = np.diag(s)
    s[2,2] = 0
    F = np.dot(u, np.dot(s,vt))
    return F

def compute_error(x1, x2, f_matrix):
    x_1 = np.hstack((x1,1))
    x_2 = np.hstack((x2, 1))
    e = np.dot(x_2.T, np.dot(f_matrix, x_1))
    return np.abs(e)

def correct_fundamental_matrix(p1,p2, n, threshold):
    inliers = []
    max_S = 0
    correct_f = None
    for i in range(n):
        points = []
        index = np.random.choice(p1.shape[0], 8, replace = 'False')
        x1_hat = p1[index,:]
        x2_hat = p2[index,:]
        S = 0
        a = construct_A(x1_hat, x2_hat)
        f = compute_fundamental_matrix(a)
        for j in range(p1.shape[0]):
            error = compute_error(p1[j,:], p2[j,:], f)
            if error < threshold:
                points.append(j)
                S+=1
        if S > max_S:
            max_S = S
            inliers = points
            correct_f = f
    if correct_f is None:
        correct_f = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC)
    
    return correct_f

def fund_mat(p1,p2):
    A = construct_A(p1,p2)
    F = compute_fundamental_matrix(A)

    return F


def EstimateFundamentalMatrix(points_a, points_b):
    """Function to calculate Fundamental Matrix
    Args:
        points_a (list): List of points from image 1
        points_b (list): List of points from image 2
    Returns:
        array: Fundamental Matrix
    """
    points_num = points_a.shape[0]
    A = []
    B = np.ones((points_num, 1))

    cu_a = np.sum(points_a[:, 0]) / points_num
    cv_a = np.sum(points_a[:, 1]) / points_num

    s = points_num / np.sum(
        ((points_a[:, 0] - cu_a)**2 + (points_a[:, 1] - cv_a)**2)**(1 / 2))
    T_a = np.dot(
        np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]]),
        np.array([[1, 0, -cu_a], [0, 1, -cv_a], [0, 0, 1]]))

    points_a = np.array(points_a.T)
    points_a = np.append(points_a, B)

    points_a = np.reshape(points_a, (3, points_num))
    points_a = np.dot(T_a, points_a)
    points_a = points_a.T

    cu_b = np.sum(points_b[:, 0]) / points_num
    cv_b = np.sum(points_b[:, 1]) / points_num

    s = points_num / np.sum(
        ((points_b[:, 0] - cu_b)**2 + (points_b[:, 1] - cv_b)**2)**(1 / 2))
    T_b = np.dot(
        np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]]),
        np.array([[1, 0, -cu_b], [0, 1, -cv_b], [0, 0, 1]]))

    points_b = np.array(points_b.T)
    points_b = np.append(points_b, B)

    points_b = np.reshape(points_b, (3, points_num))
    points_b = np.dot(T_b, points_b)
    points_b = points_b.T

    for i in range(points_num):
        u_a = points_a[i, 0]
        v_a = points_a[i, 1]
        u_b = points_b[i, 0]
        v_b = points_b[i, 1]
        A.append([
            u_a * u_b, v_a * u_b, u_b, u_a * v_b, v_a * v_b, v_b, u_a, v_a, 1
        ])


#     A = np.array(A)
#     F = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, -B))
#     F = np.append(F,[1])

    _, _, v = np.linalg.svd(A)
    F = v[-1]

    F = np.reshape(F, (3, 3)).T
    F = np.dot(T_a.T, F)
    F = np.dot(F, T_b)

    F = F.T
    U, S, V = np.linalg.svd(F)
    S = np.array([[S[0], 0, 0], [0, S[1], 0], [0, 0, 0]])
    F = np.dot(U, S)
    F = np.dot(F, V)

    F = F / F[2, 2]

    return F