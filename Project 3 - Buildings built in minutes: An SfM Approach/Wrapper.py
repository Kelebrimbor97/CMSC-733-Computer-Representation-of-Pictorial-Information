from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from Utils.NonlinearTriangulation import non_linear_triangulation
from Utils.PnPRANSAC import PnPRANSAC
from Utils.read_data import *
from Utils.GetInliersRANSAC import *
from Utils.EstimateFundamentalMatrix import *
from Utils.EssentialMatrixFromFundamentalMatrix import *
from Utils.ExtractCameraPose import *
from Utils.DisambiguateCameraPose import *
from Utils.LinearTriangulation import *
from Utils.NonlinearPnP import *
from Utils.draw_correspondence import *
from Utils.BundleAdjustment import *
from Utils.BuildVisibilityMatrix import *

K = np.array(([568.996140852, 0, 643.21055941], [0,  568.988362396, 477.982801038], [0, 0, 1]))

correspondences, indices = get_correspondence()
# print(indices)
inlier_correspondences  = []
outlier_correspondences = []
for i in range(len(correspondences)):
    if (len(correspondences[i][0])< 4):
        inlier_correspondences.append(correspondences[i])
    else:
        p1 = correspondences[i][0]
        p2 = correspondences[i][1]
        inlier_index, outlier_index  = ransac(p1, p2)
        if i == 0:
            inlier_idx = inlier_index
        inlier_correspondences.append([p1[inlier_index], p2[inlier_index]])
        outlier_correspondences.append([p1[outlier_index], p2[outlier_index]])

for i in range(1,6):
    for j in range(i+1,7):
        features = DrawCorrespondence(i, j, inlier_correspondences[i][0], inlier_correspondences[i][1], outlier_correspondences[i][0], outlier_correspondences[i][0],True)
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('image', 1000, 600)
        name = 'feature_correspondence_{}{}.png'.format(i, j)
        cv2.imwrite(name, features)
        # cv2.imshow('image', features)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

total_correspondences = 0
for i in range(0, len(inlier_correspondences)):
    total_correspondences += inlier_correspondences[i][0].shape[0]

visibility_matrix = np.zeros((total_correspondences, 6))

# print(inlier_idx)
F = EstimateFundamentalMatrix(inlier_correspondences[0][0], inlier_correspondences[0][1])
print('********** Fundamental Matrix **********')
pprint(F)
E = compute_essential_matrix(F, K)
print('********** Essential Matrix **********')
pprint(E)
C_set, R_set = estimate_camera_pose(E)
color = ['r', 'g', 'b', 'k']
X_set = []
for i in range(4):
    X = linear_triangulation(K, np.identity(3), R_set[i],  np.zeros((3, 1)),C_set[i].T, inlier_correspondences[0][0],inlier_correspondences[0][1])
    X_set.append(X)
    plt.scatter(X[:, 0], X[:, 2], c=color[i], s=1)
    ax = plt.gca()
    ax.set_xlabel('X')
    ax.set_ylabel('Z')

plt.savefig('linear_triangulation.png')
plt.close()

R, C , X_linear = find_correct_pose(R_set, C_set, X_set)
print(X_linear.shape)
X_nl = non_linear_triangulation(K, np.zeros((3,1)), np.identity(3), C, R,inlier_correspondences[0][0],inlier_correspondences[0][1], X_linear[:, 0:3])
plt.scatter(X_nl[:, 0], X_nl[:, 2], c=color[i], s=1)
ax = plt.gca()
ax.set_xlabel('X')
ax.set_ylabel('Z')
plt.savefig('non-linear.png')
plt.close()

plt.scatter(X_linear[:, 0], X_linear[:, 2], c='r', s=1)
plt.scatter(X_nl[:, 0], X_nl[:, 2], c='k', s=1)
ax = plt.gca()
ax.set_xlabel('X')
ax.set_ylabel('Z')
plt.savefig('non-linear_vs_linear.png')
plt.close()

Cset = []
Rset=[]
Cset.append(C)
Rset.append(R)

print('########## Starting PnPRANSAC ##########')
# print(indices[inlier_idx])
for i in range(3, 7):
    p12n, indices12n = find_common_points('Data/matching1.txt', i, indices[inlier_idx])
    # inlier_idx = np.array(inlier_idx)
    inter_indices = np.nonzero(np.in1d(indices, indices12n))[0]
    final_indices = np.nonzero(np.in1d(inlier_idx, inter_indices))[0]
    r_indx = final_indices
    if final_indices.shape[0] != 0:
        x = inlier_correspondences[0][0][final_indices]
        X = X_nl[final_indices]
        print(x.shape)
        print(X.shape)
        Cnew, Rnew = PnPRANSAC(X, x, K)
        Cnew, Rnew = NonLinearPnP(X, x, K, Cnew, Rnew)
        Cset.append(Cnew)
        Rset.append(Rnew)
        Xnew = linear_triangulation(K, np.identity(3), Rnew,  np.zeros((3, 1)),Cnew.T, inlier_correspondences[0][0],inlier_correspondences[0][1])
        Xnew = non_linear_triangulation(K, np.zeros((3,1)), np.identity(3), Cnew , Rnew ,inlier_correspondences[0][0],inlier_correspondences[0][1], Xnew[:, 0:3])
        X_nl = np.vstack((X_nl, Xnew))
        # V_bundle = BuildVisibilityMatrix(visibility_matrix, r_indx)   
        # points = np.hstack((inlier_correspondences[0][0].reshape((-1,1)), inlier_correspondences[0][1].reshape((-1,1))))
        # Rset, Cset, X_3D = BundleAdjustment(Cset, Rset, X_nl, K, points, V_bundle)
    else:
        continue



ax = plt.axes(projection='3d')
ax.scatter3D(X_nl[:, 0], X_nl[:, 1], X_nl[:, 2], s=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim([-0.5, 1])
ax.set_ylim([-0.5, 1])
ax.set_zlim([0, 1.5])

plt.show()
plt.close()

plt.scatter(X_nl[:, 0], X_nl[:, 2], c='r', s=1)
ax = plt.gca()
ax.set_xlabel('X')
ax.set_ylabel('Z')
plt.savefig('bundle_adjustment.png')
plt.close()