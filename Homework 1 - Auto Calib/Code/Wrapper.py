import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
import cv2

from scipy import optimize

#############################################################################################
"""
Read images and return color and gray set
"""
def img_set_reader(path):

    read_str = path+'*.jpg'
    img_set_color = []
    img_set_gray = []

    for filename in glob.glob(read_str):

        img = cv2.imread(filename)
        img_set_color.append(img)
        img_set_gray.append(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))

    return img_set_color,img_set_gray

#############################################################################################
"""
Get Chessboard corners
"""
def chessboard_corners(col_image,gray_image):
    
    imgcopy = col_image.copy()
    ret, corners = cv2.findChessboardCorners(gray_image,(9,6),None)

    if ret:
        fn1 = cv2.drawChessboardCorners(imgcopy,(9,6),corners,ret)
        return fn1, corners
    else:
        print("No corners detected")
        return col_image, corners

#############################################################################################
"""
find v components frm input H matrix matrix
"""
def find_v(H_mat):

    h1 = H_mat[:,0]
    h2 = H_mat[:,1]

    v12 = v_maker(h1,h2)
    v11 = v_maker(h1,h1)
    v22 = v_maker(h2,h2)

    return v12, np.subtract(v11,v22)

#############################################################################################
"""
Make V
"""
def v_maker(h_i,h_j):

    a1 = h_i[0]*h_j[0]
    a2 = h_i[0]*h_j[1] + h_i[1]*h_j[0]
    a3 = h_i[1]*h_j[1]
    a4 = h_i[2]*h_j[0] + h_i[0]*h_j[2]
    a5 = h_i[2]*h_j[1] + h_i[1]*h_j[2]
    a6 = h_i[2]*h_j[2]

    v = np.array([a1,a2,a3,a4,a5,a6])

    return v

#############################################################################################
"""
Solve for B
"""
def solve_B(V_set):

    _,__,Vt = np.linalg.svd(V_set)
    
    b = Vt[-1]
    
    B11 = b[0]
    B12 = b[1]
    B22 = b[2]
    B13 = b[3]
    B23 = b[4]
    B33 = b[5]

    B = np.array([[B11,B12,B13],[B12,B22,B23],[B13,B23,B33]])
    print('\nB Matrix:\n',B,'\n')

    v0 = (B12*B13 - B11*B23)/(B11*B22 - B12*B12)
    print('v0=',v0)

    l = B33 - (B13*B13 + v0*(B12*B13-B11*B23))/B11
    print('l=',l)
    alpha = np.sqrt(l/B11)
    print('alpha=',alpha)
    beta = np.sqrt(l*B11/(B11*B22 - B12*B12))
    print('beta=',beta)
    gamma = -B12*alpha*alpha*beta/l
    print('gamma=',gamma)
    u0 = gamma*v0/beta - B13*alpha*alpha/l
    print('u0=',u0)
    A = np.array([[alpha, gamma, u0],[0,beta,v0],[0,0,1]])

    return A

#############################################################################################
"""
Find Transformation matrix
"""
def transformer(H_mat,A):
    #print('Homography:\n',H_mat)

    h1 = H_mat[:,0]
    h2 = H_mat[:,1]
    h3 = H_mat[:,2]

    lam1 = 1/np.linalg.norm(np.matmul(np.linalg.inv(A),h1),ord=2)
    lam2 = 1/np.linalg.norm(np.matmul(np.linalg.inv(A),h2),ord=2)
    lamf = np.mean([lam1,lam2])

    r1 = np.multiply(np.matmul(np.linalg.inv(A),h1),lam1)
    r2 = np.multiply(np.matmul(np.linalg.inv(A),h2),lam1)
    r3 = np.cross(r1,r2)

    t = np.multiply(np.matmul(np.linalg.inv(A),h3),lam1)
    #print('t =',t[0])

    Q = np.array([r1.T,r2.T,r3.T])

    U,_,Vt = np.linalg.svd(Q)

    R = np.matmul(U,Vt)
    R_t = np.zeros((3,4))
    for i in range(3):
        for j in range(3):

            R_t[i,j] = R[i,j]

    R_t[2,0] = t[0]
    R_t[2,1] = t[1]
    R_t[2,2] = t[2]
    #print(np.shape(R_t))

    return R_t
#############################################################################################
"""
Estimate Initial reprojection error
"""
def error_init_calc(A,R_t,corners):

    world_points_x = np.arange(1,10)
    world_points_y = np.arange(1,7)
    world_points = []
    
    P = np.matmul(A,R_t)

    for i in range(len(world_points_x)):

        for j in range(len(world_points_y)):

            world_points.append([[(i+1)*21.5,(j+1)*21.5]])

    err_set = []

    for i in range(len(corners)):

        current_wcoord = np.array([world_points[i][0][0],world_points[i][0][1],0,1])
        current_corner = np.array([corners[i][0][0],corners[i][0][1],1])
        est_corner = np.matmul(P,current_wcoord)
        est_corner = np.divide(est_corner,est_corner[-1])

        err = np.linalg.norm(np.subtract(current_corner,est_corner))
        err_set.append(err)

    error_final = np.mean(err_set)

    return error_final

#############################################################################################
"""
minimizer function
"""

def minimizeFunction(A, imgpoints, homographies):
    
    K = np.zeros(shape=(3, 3))
    world_points_x = np.arange(1,10)
    world_points_y = np.arange(1,7)
    world_points = []

    for i in range(len(world_points_x)):

        for j in range(len(world_points_y)):

            world_points.append([(i+1)*21.5,(j+1)*21.5])
    
    K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[0, 1], K[2,2] = A[0], A[1], A[2], A[3], A[4], 1
    k1, k2 = A[5], A[6]
    u0, v0 = A[2], A[3]

    error = []
    i = 0
    for H in homographies:
        extrinsic = transformer(H,K)
        extrinsic = np.matmul(K,extrinsic)
        counter = 0
        for pt in imgpoints:
            model = np.array([world_points[counter][0], world_points[counter][1], 0, 1])
            
            proj_point = np.dot(extrinsic, model)
            proj_point = proj_point/proj_point[2]
            x, y = proj_point[0], proj_point[1]

            U = np.dot(K, proj_point)
            U = U/U[2]
            u, v = U[0], U[1]

            t = x**2 + y**2
            u_cap = u + (u-u0)*(k1*t + k2*(t**2))
            v_cap = v + (v-v0)*(k1*t + k2*(t**2))
            
            error.append(pt[0]-u_cap)
            error.append(pt[1]-v_cap)

            counter+=1
          

    return np.float64(error).flatten()

#############################################################################################
"""
Error optimization
"""    
def optimizer(K,corners_set,H_set):

    alpha = K[0,0]
    beta = K[1,1]
    u0 = K[0,2]
    v0 = K[1,2]
    gamma = K[0,1]

    K_init = [alpha,beta,u0,v0,gamma,0,0]

    opt_k_components = optimize.least_squares(fun=minimizeFunction, x0=K_init, method="lm", args=[corners_set,H_set])

    [alpha,beta,u0,v0,gamma,k1,k2] = opt_k_components.x

    K = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])

    return K,k1,k2

#############################################################################################
def main():

    Parser = argparse.ArgumentParser(description='Texture Editing by PRN')

    Parser.add_argument('-i',default='Calibration_Imgs/',type=str,
                        help='path to input images')
    Parser.add_argument('-o',default='Output/',type=str,
                        help='path to output images')

    Args = Parser.parse_args()
    path = Args.i
    out_path = Args.o

    img_set_color,img_set_gray = img_set_reader(path)

    check_corners = np.array([[21.5,21.5],[21.5*9,21.5],[21.5*9,21.5*6],[21.5,21.5*6]])     #Order [0,0],[1,0],[1,1],[0,1]
    V_set = []
    H_set = []
    corners_set = []
    cornerized_imgs = []

    for i in range(len(img_set_gray)):

        img_tmp,corners = chessboard_corners(img_set_color[i],img_set_gray[i])
        cornerized_imgs.append(img_tmp)
        corners_set.append(corners)
        tmp_corners_xtreme = np.array([corners[0],corners[8],corners[53],corners[45]])
        
        H_tmp,_ = cv2.findHomography(check_corners,tmp_corners_xtreme)
        H_set.append(H_tmp)
        
        vi2,vid = find_v(H_tmp)
        V_set.append(vi2)
        V_set.append(vid)

    A = solve_B(V_set)

    print('\nK Matrix before optimization:\n',A)

    R_t_set = []
    errors_total = []

    for H in H_set:
        R_t = transformer(H, A)
        R_t_set.append(R_t)
        error_tmp = error_init_calc(A,R_t,corners)
        errors_total.append(error_tmp)
    
    error_init_total = np.mean(errors_total)
    #print('\ninitial error:',error_init_total)

    K_opt,k1,k2 = optimizer(A,corners_set,H_set)
    print("\nOptimized K:\n",K_opt,'\n\nk1 =',k1,'k2 =',k2)

    distortion = np.array([k1,k2,0,0,0])

    undistorted_images = []

    for i in range(len(img_set_color)):

        undimg = cv2.undistort(img_set_color[i],A,distortion)

        plt.figure(i+1)
        
        plt.subplot(1,2,1)
        plt.axis("off")
        plt.imshow(cv2.cvtColor(cornerized_imgs[i],cv2.COLOR_BGR2RGB))

        plt.subplot(1,2,2)
        plt.axis("off")
        plt.imshow(cv2.cvtColor(undimg,cv2.COLOR_BGR2RGB))

    plt.show()

    R_t_set = []
    errors_total = []

    for H in H_set:
        R_t = transformer(H, A)
        R_t_set.append(R_t)
        error_tmp = error_init_calc(A,R_t,corners)
        errors_total.append(error_tmp)
    
    error_init_total = np.mean(errors_total)
    #print('\nFinal error:',error_init_total)

if __name__=='__main__':
    main()