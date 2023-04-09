import dlib
import numpy as np
from Utils.misc import *


def convert_to_numpy(dlib_shape):
    """Converts dlib shape object to a list"""
    coordinates = []
    for i in range(0, dlib_shape.num_parts):
        coordinates.append(np.array([dlib_shape.part(i).x, dlib_shape.part(i).y], dtype='int'))
    # Return the list of (x,y) coordinates:
    return coordinates

def convert_to_list(dlib_shape):
    """Converts dlib shape object to a list"""
    coordinates = []
    for i in range(0, dlib_shape.num_parts):
        coordinates.append([dlib_shape.part(i).x, dlib_shape.part(i).y])
    # Return the list of (x,y) coordinates:
    return coordinates


def compute_landmarks(image, method):
    PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    face_features = detector(image, 0)
    features_list = []
    for (i, feature) in enumerate(face_features):
        features = predictor(image, feature)
        if method == 1:
            features_list.append(convert_to_list(features))
        else:
            features_list.append(convert_to_numpy(features))
    
    return features_list


def get_delaunay_triangles(hull,size, hull_source):
    delaunay_triangles = []
    pt=[]
    rect = (0, 0, size[1], size[0])
    subdiv_target = cv2.Subdiv2D(rect)

    for p in hull:
        subdiv_target.insert(p)

    triangles = subdiv_target.getTriangleList()
    for t in triangles :
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        
        
        if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
            ind = []
            #Get face-points (from 68 face detector) by coordinates
            for j in range(0, 3):
                for k in range(0, len(hull)):                    
                    if(abs(pt[j][0] - hull[k][0]) < 1.0 and abs(pt[j][1] - hull[k][1]) < 1.0):
                        ind.append(k)    
            # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph 
            if len(ind) == 3:                                                
                delaunay_triangles.append((ind[0], ind[1], ind[2]))
        
        pt = []  
    # print(delaunay_triangles)
    t1 = []
    t2 = []
    for i in range(0, len(delaunay_triangles)):
        t1.append([hull[delaunay_triangles[i][0]][0], hull[delaunay_triangles[i][0]][1], hull[delaunay_triangles[i][1]][0],hull[delaunay_triangles[i][1]][1],hull[delaunay_triangles[i][2]][0],hull[delaunay_triangles[i][2]][1]])
        t2.append([hull_source[delaunay_triangles[i][0]][0], hull_source[delaunay_triangles[i][0]][1], hull_source[delaunay_triangles[i][1]][0],hull_source[delaunay_triangles[i][1]][1],hull_source[delaunay_triangles[i][2]][0],hull_source[delaunay_triangles[i][2]][1]])
    
    return t1, t2

def get_correspondence(feature_t, triangle_t, feature_s):
    i = 1
    for t in triangle_t :
        pt1 = [int(t[0]), int(t[1])]
        pt2 = [int(t[2]), int(t[3])]
        pt3 = [int(t[4]), int(t[5])]
        index_1 = feature_t.index(pt1)
        index_2 = feature_t.index(pt2)
        index_3 = feature_t.index(pt3)
        if(i == 1):
            triangle_s = np.array([feature_s[index_1][0], feature_s[index_1][1], feature_s[index_2][0], feature_s[index_2][1], feature_s[index_3][0], feature_s[index_3][1]])
            i = i+1
        else:
            triangle_s = np.vstack((triangle_s, [feature_s[index_1][0], feature_s[index_1][1], feature_s[index_2][0], feature_s[index_2][1], feature_s[index_3][0], feature_s[index_3][1]]))


    return triangle_s

def swap_faces(triangle_t, triangle_s, target_copy, source):
    for k in range(len(triangle_t)):
        t = triangle_t[k]
        s = triangle_s[k]
        B = np.array([int(t[0]), int(t[2]), int(t[4])])
        B = np.vstack((B,[int(t[1]), int(t[3]), int(t[5])]))
        B = np.vstack((B, [1,1,1]))
        A = np.array([int(s[0]), int(s[2]), int(s[4])])
        A = np.vstack((A,[int(s[1]), int(s[3]), int(s[5])]))
        A = np.vstack((A, [1,1,1]))
        xmin, ymin, xmax, ymax = get_bb(B)
        for i in range(xmin, xmax+1):
            for j in range(ymin, ymax + 1):
                pixel_target = np.array([[i], [j], [1]])
                if np.linalg.det(B)!=0:
                    B_inv = np.linalg.inv(B)
                else:    
                    B_inv = np.linalg.pinv(B)
                barycentric_cord = np.matmul(B_inv, pixel_target)
                # print(barycentric_cord.shape)
                alpha = barycentric_cord[0,0]
                beta = barycentric_cord[1,0]
                gamma = barycentric_cord[2,0]
                # print(alpha, beta, gamma)
                if ((alpha >=-0.01 and alpha <= 1.01)and (beta >=-0.01 and beta <= 1.01) and (gamma >=-0.01 and gamma <= 1.01) ):
                    sum = alpha + beta + gamma
                    if (sum>0 and sum<=1):
                        pixel_source = np.matmul(A, barycentric_cord)
                        x_a = pixel_source[0,0]/pixel_source[2,0]
                        y_a = pixel_source[1,0]/pixel_source[2,0]
                        # target_copy[j, i, 0] = b(x_a, y_a)[0]
                        # target_copy[j, i, 1] = g(x_a, y_a)[0]
                        # target_copy[j, i, 2] = r(x_a, y_a)[0]
                        target_copy[j, i] = source[int(y_a),int(x_a)]
                        # target_copy[int(y_a),int(x_a),:] = source[j, i , :]
    return target_copy