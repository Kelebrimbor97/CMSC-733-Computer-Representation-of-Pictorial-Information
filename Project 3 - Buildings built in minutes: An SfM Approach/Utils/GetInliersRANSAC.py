import numpy as np
import cv2
def ransac(pts1, pts2, N=100, t=0.9, thresh=30):

    H_new = np.zeros((3,3))
    max_inliers = 0
    
    for j in range(N):
        index = []
        outlier = []
        pts = [np.random.randint(0,len(pts1)) for i in range(4)]
        p1 = pts1[pts]
        p2 = pts2[pts]
        H = cv2.getPerspectiveTransform( np.float32(p1), np.float32(p2))
        inLiers = 0
        for ind in range(len(pts1)):
            source = pts1[ind]
            target = np.array([pts2[ind][0],pts2[ind][1]])
            predict = np.dot(H, np.array([source[0],source[1],1]))
            if predict[2] != 0:
                predict_x = predict[0]/predict[2]
                predict_y = predict[1]/predict[2]
            else:
                predict_x = predict[0]/0.000001
                predict_y = predict[1]/0.000001

            predict = np.array([predict_x,predict_y])
            predict = np.float32([point for point in predict])

            a = np.linalg.norm(target-predict)
            if a < thresh:
                inLiers += 1
                index.append(ind)
            else:
                outlier.append(ind)

        if max_inliers < inLiers:
            max_inliers = inLiers
            H_new = H
            if inLiers > t*len(pts1):
                break
    return index, outlier