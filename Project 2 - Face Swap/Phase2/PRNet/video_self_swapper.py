import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast
import matplotlib.pyplot as plt
import argparse

from api_sameswap import PRN
from utils.render import render_texture
import cv2

#######################################################################################################

def vid_swapper(args,prn):

    cap = cv2.VideoCapture(args.video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(args.output_path,cv2.VideoWriter_fourcc('M','J','P','G'),fps,(w,h))

    if(not cap.isOpened()):print("Error in openeing video")

    frame_counter = 0

    while(cap.isOpened()):

        ret,frame = cap.read()

        if ret:

            abomination = texture_editing(prn,args,frame)
            print("Frame",frame_counter,"done...")
            frame_counter+=1

            #cv2.imshow("Feast Your eyes",abomination)
            out.write(abomination)

            if cv2.waitKey(30) & 0XFF == ord('q'):break

        else:break

    cap.release()
    out.release()

    cv2.destroyAllWindows()

#######################################################################################################

def texture_editing(prn, args,frame):
    # read image
    image = frame
    [h, w, _] = image.shape

    #-- 1. 3d reconstruction -> get texture. 
    pos1, pos2 = prn.process(image)
    
    if(np.shape(pos1)==() or np.shape(pos2)==()):
        
        return frame

    #Extract Face1 location
    vertices1 = prn.get_vertices(pos1)
    image1 = frame/255.
    texture1 = cv2.remap(image1, pos1[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))

    #Extract Face2 location
    vertices2 = prn.get_vertices(pos2)
    image2 = frame/255.
    texture2 = cv2.remap(image2, pos2[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    
    #-- 2. Texture Editing
    Mode = args.mode
    # change part of texture(for data augumentation/selfie editing. Here modify eyes for example)
    if Mode == 0: 
        # load eye mask
        uv_face_eye = imread('Data/uv-data/uv_face_eyes.png', as_grey=True)/255. 
        uv_face = imread('Data/uv-data/uv_face.png', as_grey=True)/255.
        eye_mask = (abs(uv_face_eye - uv_face) > 0).astype(np.float32)

        # texture from another image or a processed texture
        ref_image1 = frame
        ref_pos1 = pos2
        ref_image1 = ref_image1/255.
        ref_texture1 = cv2.remap(ref_image1, ref_pos1[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))

        # modify texture
        new_texture1 = texture1*(1 - eye_mask[:,:,np.newaxis]) + ref_texture1*eye_mask[:,:,np.newaxis]

        ref_image2 = frame
        ref_pos2 = pos1
        ref_image2 = ref_image2/255.
        ref_texture2 = cv2.remap(ref_image2, ref_pos2[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))

        # modify texture
        new_texture2 = texture2*(1 - eye_mask[:,:,np.newaxis]) + ref_texture2*eye_mask[:,:,np.newaxis]

        
    # change whole face(face swap)
    elif Mode == 1: 
        # texture from another image or a processed texture
        ref_image1 = frame
        ref_pos1 = pos2
        ref_image1 = frame/255.
        ref_texture1 = cv2.remap(ref_image1, ref_pos1[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
        ref_vertices1 = prn.get_vertices(ref_pos1)
        new_texture1 = ref_texture1#(texture + ref_texture)/2.

        ref_image2 = frame
        ref_pos2 = pos1
        ref_image2 = frame/255.
        ref_texture2 = cv2.remap(ref_image2, ref_pos2[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
        ref_vertices2 = prn.get_vertices(ref_pos2)
        new_texture2 = ref_texture2#(texture + ref_texture)/2.

    else:
        print('Wrong Mode! Mode should be 0 or 1.')
        exit()


    #-- 3. remap to input image.(render)
    vis_colors1 = np.ones((vertices1.shape[0], 1))
    face_mask1 = render_texture(vertices1.T, vis_colors1.T, prn.triangles.T, h, w, c = 1)
    face_mask1 = np.squeeze(face_mask1 > 0).astype(np.float32)
    
    new_colors1 = prn.get_colors_from_texture(new_texture1)
    new_image1 = render_texture(vertices1.T, new_colors1.T, prn.triangles.T, h, w, c = 3)

    # Possion Editing for blending image

    vis_colors2 = np.ones((vertices2.shape[0], 1))
    face_mask2 = render_texture(vertices2.T, vis_colors2.T, prn.triangles.T, h, w, c = 1)
    face_mask2 = np.squeeze(face_mask2 > 0).astype(np.float32)

    new_colors2 = prn.get_colors_from_texture(new_texture2)
    new_image2 = render_texture(vertices2.T, new_colors2.T, prn.triangles.T, h, w, c = 3)

    
    new_image1 = image*(1 - face_mask1[:,:,np.newaxis]) + new_image1*face_mask1[:,:,np.newaxis]   
    vis_ind = np.argwhere(face_mask1>0)
    vis_min = np.min(vis_ind, 0)
    vis_max = np.max(vis_ind, 0)
    center = (int((vis_min[1] + vis_max[1])/2+0.5), int((vis_min[0] + vis_max[0])/2+0.5))
    output = cv2.seamlessClone((new_image1*255).astype(np.uint8), (image).astype(np.uint8), (face_mask1*255).astype(np.uint8), center, cv2.NORMAL_CLONE)
    
    new_image2 = output*(1 - face_mask2[:,:,np.newaxis]) + new_image2*face_mask2[:,:,np.newaxis] 
    vis_ind = np.argwhere(face_mask2>0)
    vis_min = np.min(vis_ind, 0)
    vis_max = np.max(vis_ind, 0)
    center = (int((vis_min[1] + vis_max[1])/2+0.5), int((vis_min[0] + vis_max[0])/2+0.5))
    output = cv2.seamlessClone((new_image2*255).astype(np.uint8), (output).astype(np.uint8), (face_mask2*255).astype(np.uint8), center, cv2.NORMAL_CLONE)

    # save output
    #imsave(args.output_path, output) 
    #print('Done.')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Texture Editing by PRN')

    parser.add_argument('-v', '--video_path', default='TestSet_P2/Test1.mp4', type=str,
                        help='path to input image')
    parser.add_argument('-r1', '--ref_path1', default='TestSet_P2/Rambo.jpg', type=str, 
                        help='path to reference image(texture ref)')
    parser.add_argument('-r2', '--ref_path2', default='TestSet_P2/Scarlett.jpg', type=str, 
                        help='path to reference image(texture ref)')
    parser.add_argument('-o', '--output_path', default='TestImages/swap.avi', type=str, 
                        help='path to save output')
    parser.add_argument('--mode', default=1, type=int, 
                        help='ways to edit texture. 0 for modifying parts, 1 for changing whole')
    parser.add_argument('--gpu', default='0', type=str, 
                        help='set gpu id, -1 for CPU')

    # ---- init PRN
    os.environ['CUDA_VISIBLE_DEVICES'] = parser.parse_args().gpu # GPU number, -1 for CPU
    prn = PRN(is_dlib = True) 

    #texture_editing(prn, parser.parse_args())
    vid_swapper(parser.parse_args(),prn)