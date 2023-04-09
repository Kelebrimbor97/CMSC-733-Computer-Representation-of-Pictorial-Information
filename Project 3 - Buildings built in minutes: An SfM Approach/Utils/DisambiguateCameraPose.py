import numpy as np

def cheirality_check(r3, X, c):
    # returns the number of points infront of the camera with positive depth
    count = 0
    for x in X:
        x = x.reshape((-1,1)) 
        if (r3.dot(x-c) > 0 and x[2]>0):
            count += 1
    return count

def find_correct_pose(R, C, X_points):
    correct_pose = 0
    max = 0
    for i in range(len(R)):
        R3 = R[i][2,:].reshape((1,-1))
        p_count = cheirality_check(R3, X_points[i][:,0:3], C[i].reshape((-1,1)))
        if p_count > max:
            correct_pose = i
            max = p_count
    
    return R[correct_pose], C[correct_pose], X_points[correct_pose]