import numpy as np
import cv2
from Utils.face_features import *
from Utils.tps import *
from Utils.misc import *
import os


def main():
    choice = int(input('Enter the choice of swapping technique: 1-Delaunay Triangulation 2-TPS '))
    cv2.namedWindow("video_frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("video_frame", 700, 700)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video1 = os.path.join(base_dir, 'Data/Data1.mp4')
    source = cv2.imread(os.path.join(base_dir, 'Data/vk.jpeg'))
    gray2 = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    source_landmarks =  compute_landmarks(gray2,2)[0]
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    tps_video = cv2.VideoWriter('Data1OutputTPS.mp4',fourcc, 20, (270,480))
    dt_video = cv2.VideoWriter('Data1OutputTri.mp4',fourcc, 20, (270,480))
    cap = cv2.VideoCapture(video1)
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    while True:
        ret, target = cap.read()
        
        if ret == True:
            
            # target = cv2.resize(target, (240,426), interpolation = cv2.INTER_AREA)
            target_copy = target.copy()
            print(target.shape)
            if (choice == 2):
                
                gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
                face_landmarks = compute_landmarks(gray,2)[0]
                mask = create_mask(target_copy, face_landmarks)
                final= TPS(face_landmarks, source_landmarks, target_copy, source, mask)
                blend_final = blending(final, target, mask)
                cv2.imshow('video_frame', blend_final)
                tps_video.write(blend_final)   
            else:
                
                gray1 = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
                features_list_target = compute_landmarks(gray1,1)[0]
                features_list_source = compute_landmarks(gray2,1)[0]
                size = target.shape
                hull1 = []
                hull2 = []

                hullIndex = cv2.convexHull(np.array(features_list_target), returnPoints = False)
                    
                for i in range(0, len(hullIndex)):
                    hull1.append(features_list_target[int(hullIndex[i])])
                    hull2.append(features_list_source[int(hullIndex[i])])

                triangleList_target, triangleList_source = get_delaunay_triangles(hull1, size, hull2)
                delaunay_img_t = draw_delaunay( target, triangleList_target[:1], (255, 255, 255) )
                delaunay_img_s = draw_delaunay( source, triangleList_source[:1], (255, 255, 255) )
                swapped_img = swap_faces(triangleList_target,triangleList_source, target_copy, source)

                mask  = create_mask(target, features_list_target)                 
                blend_swap = blending(swapped_img, target, mask)
                cv2.imshow('video_frame', blend_swap)
                dt_video.write(blend_swap)
            if cv2.waitKey(1) & 0xFF ==ord('q'):
                break
        else:
            break

    dt_video.release()
    tps_video.release()
    cap.release()
    video2 = os.path.join(base_dir, 'Data/Data2.mp4')
    tps_video2 = cv2.VideoWriter('Data2OutputTPS.mp4',fourcc, 20, (640,360))
    dt_video2 = cv2.VideoWriter('Data2OutputTri.mp4',fourcc, 20, (640,360))
    cap = cv2.VideoCapture(video2)
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    while True:
        ret, target = cap.read()
        
        if ret == True:
            # target = cv2.resize(target, (240,426), interpolation = cv2.INTER_AREA)
            print(target.shape)
            target_copy = target.copy()
            if (choice == 2):
                
                gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
                face_landmarks = compute_landmarks(gray,2)
                # print(len(face_landmarks))
                if len(face_landmarks)!= 2:
                    print('Two faces not detected')
                else:
                    target_landmarks = face_landmarks[0]
                    source_landmarks = face_landmarks[1]
                    mask = create_mask(target_copy, target_landmarks)
                    swap1= TPS(target_landmarks, source_landmarks, target_copy, target, mask)
                    blend_swap1 = blending(swap1, target, mask)
                    # cv2.imshow('video_frame', blend_swap1)
                    # cv2.waitKey(0)

                    target_landmarks = face_landmarks[1]
                    source_landmarks = face_landmarks[0]
                    mask = create_mask(target, target_landmarks)
                    swap2= TPS(target_landmarks, source_landmarks, blend_swap1, target, mask)
                    blend_swap2 = blending(swap2, blend_swap1, mask)
                    cv2.imshow('video_frame', blend_swap2)
                    tps_video2.write(blend_swap2)   
            else:
                
                gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
                face_landmarks = compute_landmarks(gray,1)
                # print(len(face_landmarks))
                if len(face_landmarks)!= 2:
                    print('Two faces not detected')
                else:
                    target_landmarks = face_landmarks[0]
                    source_landmarks = face_landmarks[1]
                    mask = create_mask(target_copy, target_landmarks)
                    size = target.shape
                    hull1 = []
                    hull2 = []

                    hullIndex = cv2.convexHull(np.array(target_landmarks), returnPoints = False)
                        
                    for i in range(0, len(hullIndex)):
                        hull1.append(target_landmarks[int(hullIndex[i])])
                        hull2.append(source_landmarks[int(hullIndex[i])])

                    triangleList_target, triangleList_source = get_delaunay_triangles(hull1, size, hull2)
                    swapped_img = swap_faces(triangleList_target,triangleList_source, target_copy, target)
                    blend_swap1 = blending(swapped_img, target, mask)
                    # cv2.imshow('video_frame', blend_swap1)
                    # cv2.waitKey(0)

                    target_landmarks = face_landmarks[1]
                    source_landmarks = face_landmarks[0]
                    mask = create_mask(target, target_landmarks)
                    hull1 = []
                    hull2 = []

                    hullIndex = cv2.convexHull(np.array(target_landmarks), returnPoints = False)
                        
                    for i in range(0, len(hullIndex)):
                        hull1.append(target_landmarks[int(hullIndex[i])])
                        hull2.append(source_landmarks[int(hullIndex[i])])
                    triangleList_target, triangleList_source = get_delaunay_triangles(hull1, size, hull2)
                    swapped_img2 = swap_faces(triangleList_target,triangleList_source, blend_swap1, target)
                    blend_swap2 = blending(swapped_img2, blend_swap1, mask)
                    cv2.imshow('video_frame', blend_swap2)
                    dt_video2.write(blend_swap2)
            if cv2.waitKey(1) & 0xFF ==ord('q'):
                break
        else:
            break
    dt_video2.release()
    tps_video2.release()
    cap.release()

if __name__ == '__main__':
    main()