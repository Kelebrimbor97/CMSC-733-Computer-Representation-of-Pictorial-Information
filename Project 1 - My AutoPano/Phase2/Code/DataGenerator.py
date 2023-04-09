import numpy as np
import pandas as pd
import cv2
import glob
import os

def patch_generator(image, patch_size, margin_size, p):
  height = image.shape[0]
  width = image.shape[1]
  final_margin = patch_size + margin_size
  # minimum dimension of the input image to generate a patch
  min_dim = patch_size + margin_size
  #condition to check if a patch can be genrated on the input image
  if ((height > min_dim) and (width > min_dim)):
    #select the topmost left corner of the patch
    left_x = np.random.randint(margin_size, width - final_margin)
    top_y = np.random.randint(margin_size, height - final_margin)
    right_x = left_x + patch_size
    bot_y = top_y + patch_size
    # 4 points of the patch P_A
    C_A = np.array([[left_x, top_y],[left_x, bot_y],[right_x, top_y],[right_x, bot_y]], dtype='float32')
    C_B = np.zeros(C_A.shape)
    # Performing a random perturbation to find P_B
    for i in range(C_A.shape[0]):
      C_B[i][0] = C_A[i][0] + np.random.randint(-p, p)
      C_B[i][1] = C_A[i][1] + np.random.randint(-p, p)
    #Finding the homography matrix H_BA
    H_BA = np.linalg.inv(cv2.getPerspectiveTransform(np.float32(C_A), np.float32(C_B)))
    # Using H_BA to warp input image and obtain I_B
    I_B = cv2.warpPerspective(image, H_BA, (width, height))
    # Extracting P_A and P_B
    P_A = image[top_y:bot_y, left_x:right_x]
    P_B = I_B[top_y:bot_y, left_x:right_x]
    H_4pt = C_B - C_A
    
    return I_B,P_A, P_B, H_4pt, np.dstack((C_A,C_B))

  else:
    return None

def data_generator(base_dir):
  patch_size = 128
  p = 32
  # image_resize = [320,240]
  dataset_type = ['Train', 'Val', 'Test']
  for type in dataset_type:
    if(type == 'Train'):
      print('Generating synthetic training data')
      image_files  = glob.glob(base_dir + '/Train' + '/*.jpg')
      print(image_files)
      save_path = base_dir + '/S_train'
      print(save_path)
    elif(type == 'Val'):
      print('Generating synthetic validation data')
      image_files  = glob.glob(base_dir + '/Val' + '/*.jpg')
      save_path = base_dir+ '/S_val'
    else:
      print('Generating synthetic test data')
      image_files  = glob.glob(base_dir + '/Test' + '/*.jpg')
      save_path = base_dir + '/S_test'
      patch_size = 256
      p = 64
      image_resize = [640, 480]
    if(not (os.path.isdir(save_path))):
            print(save_path, "Creating the folder")
            os.makedirs(save_path)
    
    

    H_4pt_list = []
    points_list = []
    for image in image_files:
      img = cv2.imread(image) # read the image in grayscale
      img = cv2.resize(img, (320,240), interpolation = cv2.INTER_AREA) # resize the image
      I_B, P_A, P_B, H_4pt, points = patch_generator(img, patch_size, 50, p)
      img_name = str.split(image,'/')
      if(not (os.path.isdir(save_path +'/P_A/'))):
        os.makedirs(save_path +'/P_A/')
        os.makedirs(save_path +'/P_B/')
        os.makedirs(save_path +'/I_A/')
        os.makedirs(save_path +'/I_B/')
      cv2.imwrite((save_path + '/P_A/' + str.split(img_name[-1],'.')[0] + '.jpg'), P_A)
      cv2.imwrite((save_path + '/P_B/' + str.split(img_name[-1],'.')[0] + '.jpg'), P_B)
      cv2.imwrite((save_path + '/I_A/' + str.split(img_name[-1],'.')[0] + '.jpg'), img)
      cv2.imwrite((save_path + '/I_B/' + str.split(img_name[-1],'.')[0] + '.jpg'), I_B)
      H_4pt_list.append(np.hstack((H_4pt[:,0] , H_4pt[:,1])))
      points_list.append(points) 

    df = pd.DataFrame(H_4pt_list)
    df.to_csv(save_path+"/H_4pt.csv", index=False)
    print("saved H_4pt data in:  ", save_path)

    np.save(save_path+"/points_list.npy", np.array(points_list))
    print("saved points data in:  ", save_path)