import numpy as np
import os
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp
from time import time

from predictor import PosPrediction

class PRN:
    ''' Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network
    Args:
        is_dlib(bool, optional): If true, dlib is used for detecting faces.
        prefix(str, optional): If run at another folder, the absolute path is needed to load the data.
    '''
    def __init__(self, is_dlib = False, prefix = '.'):

        # resolution of input and output image size.
        self.resolution_inp = 256
        self.resolution_op = 256

        #---- load detectors
        if is_dlib:
            import dlib
            detector_path = os.path.join(prefix, 'Data/net-data/mmod_human_face_detector.dat')
            self.face_detector = dlib.cnn_face_detection_model_v1(
                    detector_path)

        #---- load PRN 
        self.pos_predictor = PosPrediction(self.resolution_inp, self.resolution_op)
        prn_path = os.path.join(prefix, 'Data/net-data/256_256_resfcn256_weight')
        if not os.path.isfile(prn_path + '.data-00000-of-00001'):
            print("please download PRN trained model first.")
            exit()
        self.pos_predictor.restore(prn_path)

        # uv file
        self.uv_kpt_ind = np.loadtxt(prefix + '/Data/uv-data/uv_kpt_ind.txt').astype(np.int32) # 2 x 68 get kpt
        self.face_ind = np.loadtxt(prefix + '/Data/uv-data/face_ind.txt').astype(np.int32) # get valid vertices in the pos map
        self.triangles = np.loadtxt(prefix + '/Data/uv-data/triangles.txt').astype(np.int32) # ntri x 3
        
        self.uv_coords = self.generate_uv_coords()        

    def generate_uv_coords(self):
        resolution = self.resolution_op
        uv_coords = np.meshgrid(range(resolution),range(resolution))
        uv_coords = np.transpose(np.array(uv_coords), [1,2,0])
        uv_coords = np.reshape(uv_coords, [resolution**2, -1]);
        uv_coords = uv_coords[self.face_ind, :]
        uv_coords = np.hstack((uv_coords[:,:2], np.zeros([uv_coords.shape[0], 1])))
        return uv_coords

    def dlib_detect(self, image):
        return self.face_detector(image, 1)

    def net_forward(self, image):
        ''' The core of out method: regress the position map of a given image.
        Args:
            image: (256,256,3) array. value range: 0~1
        Returns:
            pos: the 3D position map. (256, 256, 3) array.
        '''
        return self.pos_predictor.predict(image)

    def process(self, input, image_info = None):
        ''' process image with crop operation.
        Args:
            input: (h,w,3) array or str(image path). image value range:1~255. 
            image_info(optional): the bounding box information of faces. if None, will use dlib to detect face. 

        Returns:
            pos: the 3D position map. (256, 256, 3).
        '''
        if isinstance(input, str):
            try:
                image = imread(input)
            except IOError:
                print("error opening file: ", input)
                return None
        else:
            image = input

        if image.ndim < 3:
            image = np.tile(image[:,:,np.newaxis], [1,1,3])

        if image_info is not None:
            if np.max(image_info.shape) > 4: # key points to get bounding box
                kpt = image_info
                if kpt.shape[0] > 3:
                    kpt = kpt.T
                left = np.min(kpt[0, :]); right = np.max(kpt[0, :]); 
                top = np.min(kpt[1,:]); bottom = np.max(kpt[1,:])
            else:  # bounding box
                bbox = image_info
                left = bbox[0]; right = bbox[1]; top = bbox[2]; bottom = bbox[3]
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
            size = int(old_size*1.6)
        else:
            detected_faces = self.dlib_detect(image)
            
            if len(detected_faces) < 2:
                print('warning: not enough faces detected')
                return None, None

            d1 = detected_faces[0].rect ## only use the first detected face (assume that each input image only contains one face)
            left1 = d1.left(); right1 = d1.right(); top1 = d1.top(); bottom1 = d1.bottom()
            old_size1 = (right1 - left1 + bottom1 - top1)/2
            center1 = np.array([right1 - (right1 - left1) / 2.0, bottom1 - (bottom1 - top1) / 2.0 + old_size1*0.14])
            size1 = int(old_size1*1.58)

            d2 = detected_faces[1].rect ## use the second detected face (assume that each input image contains more than one face)
            left2 = d2.left(); right2 = d2.right(); top2 = d1.top(); bottom2 = d1.bottom()
            old_size2 = (right2 - left2 + bottom2 - top2)/2
            center2 = np.array([right2 - (right2 - left2) / 2.0, bottom2 - (bottom2 - top2) / 2.0 + old_size2*0.14])
            size2 = int(old_size2*1.58)


        # crop image
        src_pts1 = np.array([[center1[0]-size1/2, center1[1]-size1/2], [center1[0] - size1/2, center1[1]+size1/2], [center1[0]+size1/2, center1[1]-size1/2]])
        DST_PTS1 = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform1 = estimate_transform('similarity', src_pts1, DST_PTS1)
        
        image1 = image/255.
        cropped_image1 = warp(image1, tform1.inverse, output_shape=(self.resolution_inp, self.resolution_inp))

        # run our net
        #st = time()
        cropped_pos1 = self.net_forward(cropped_image1)
        #print 'net time:', time() - st

        # restore 
        cropped_vertices1 = np.reshape(cropped_pos1, [-1, 3]).T
        z1 = cropped_vertices1[2,:].copy()/tform1.params[0,0]
        cropped_vertices1[2,:] = 1
        vertices1 = np.dot(np.linalg.inv(tform1.params), cropped_vertices1)
        vertices1 = np.vstack((vertices1[:2,:], z1))
        pos1 = np.reshape(vertices1.T, [self.resolution_op, self.resolution_op, 3])

        src_pts2 = np.array([[center2[0]-size2/2, center2[1]-size2/2], [center2[0] - size2/2, center2[1]+size2/2], [center2[0]+size2/2, center2[1]-size2/2]])
        DST_PTS2 = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform2 = estimate_transform('similarity', src_pts2, DST_PTS2)

        image2 = image/255.
        cropped_image2 = warp(image2, tform2.inverse, output_shape=(self.resolution_inp, self.resolution_inp))

        # run our net
        #st = time()
        cropped_pos2 = self.net_forward(cropped_image2)
        #print 'net time:', time() - st

        # restore 
        cropped_vertices2 = np.reshape(cropped_pos2, [-1, 3]).T
        z2 = cropped_vertices2[2,:].copy()/tform2.params[0,0]
        cropped_vertices2[2,:] = 1
        vertices2 = np.dot(np.linalg.inv(tform2.params), cropped_vertices2)
        vertices2 = np.vstack((vertices2[:2,:], z2))
        pos2 = np.reshape(vertices2.T, [self.resolution_op, self.resolution_op, 3])

        #print("pos1==pos2?",pos1==pos2)
        return pos1,pos2
            
    def get_landmarks(self, pos):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            kpt: 68 3D landmarks. shape = (68, 3).
        '''
        kpt = pos[self.uv_kpt_ind[1,:], self.uv_kpt_ind[0,:], :]
        return kpt


    def get_vertices(self, pos):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            vertices: the vertices(point cloud). shape = (num of points, 3). n is about 40K here.
        '''
        all_vertices = np.reshape(pos, [self.resolution_op**2, -1])
        vertices = all_vertices[self.face_ind, :]

        return vertices

    def get_colors_from_texture(self, texture):
        '''
        Args:
            texture: the texture map. shape = (256, 256, 3).
        Returns:
            colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
        '''
        all_colors = np.reshape(texture, [self.resolution_op**2, -1])
        colors = all_colors[self.face_ind, :]

        return colors


    def get_colors(self, image, vertices):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
        '''
        [h, w, _] = image.shape
        vertices[:,0] = np.minimum(np.maximum(vertices[:,0], 0), w - 1)  # x
        vertices[:,1] = np.minimum(np.maximum(vertices[:,1], 0), h - 1)  # y
        ind = np.round(vertices).astype(np.int32)
        colors = image[ind[:,1], ind[:,0], :] # n x 3

        return colors