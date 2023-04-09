#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code

"""


import tensorflow as tf
import cv2
import os
import sys
import glob
import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import *
from Misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
import string
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *

def load_data_unsupervised(test_files_pa,test_files_pb,IA_files, points_list, shuffle = True):

    patches_ab = []
    ca = []
    patches_b = []
    I_A = []

    for i in range(len(test_files_pa)):
        patch_a = cv2.imread(test_files_pa[i], cv2.IMREAD_GRAYSCALE)
        patch_b = cv2.imread(test_files_pb[i], cv2.IMREAD_GRAYSCALE)
        image_a = cv2.imread(IA_files[i], cv2.IMREAD_GRAYSCALE)

        patch_a = np.float32(patch_a)
        patch_b = np.float32(patch_b) 
        image_a = np.float32(image_a)   

        #combine images along depth
        patch_ab = np.dstack((patch_a, patch_b))     
        corner1 = points_list[i, :, :, 0]
        
        
        patches_ab.append(patch_ab)
        ca.append(corner1)
        patches_b.append(patch_b.reshape(128, 128, 1))

        I_A.append(image_a.reshape(image_a.shape[0], image_a.shape[1], 1))

    patch_indices = getPatchIndices(np.array(ca))    
    return np.array(patches_ab), np.array(ca), np.array(patches_b), np.array(I_A), patch_indices

def Test_unsupervised(PatchPairsPH, CornerPH, Patch2PH, Image1PH,patchIndicesPH, ModelPath, test_files_pa, test_files_pb, image_files, pointsList, SavePath, NumTestSamples):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    DataPath - Paths of all images where testing will be run on
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to ./TxtFiles/PredOut.txt
    """
    
    if(not (os.path.isdir(SavePath))):
        print(SavePath, "  was not present, creating the folder...")
        os.makedirs(SavePath)

    # Create the graph
    # Predict output with forward pass, MiniBatchSize for Test is 1
    _, _, H_batches = unsupervised_HomographyNet(PatchPairsPH, CornerPH, Patch2PH, Image1PH, patchIndicesPH, NumTestSamples)

    # Setup Saver
    # load session and run
    Saver = tf.train.Saver()
    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
            
        PatchPairsBatch, Corner1Batch, patch2Batch, Image1Batch, patchIndicesBatch = load_data_unsupervised(test_files_pa,test_files_pb,image_files, pointsList, shuffle = True)
        FeedDict = {PatchPairsPH: PatchPairsBatch, CornerPH: Corner1Batch, Patch2PH: patch2Batch, Image1PH: Image1Batch, patchIndicesPH: patchIndicesBatch}            
            
        H_pred = sess.run(H_batches, FeedDict)
        np.save(SavePath+'H_Pred.npy', H_pred)

def GenerateBatch(test_files_pa, test_files_pb, TrainLabels, MiniBatchSize):

  patches_ab = []
  labels = []

  for i in range(MiniBatchSize):
    index = random.randint(0, len(test_files_pa)-1) 
    patch_a = cv2.imread(test_files_pa[i], cv2.IMREAD_GRAYSCALE)
    patch_b = cv2.imread(test_files_pb[i], cv2.IMREAD_GRAYSCALE)

    patch_a = np.float32(patch_a)
    patch_b = np.float32(patch_b)  

    #combine images along depth
    patch_ab = np.dstack((patch_a, patch_b))     
    labels.append(TrainLabels[i,:].reshape(8,))
    patches_ab.append(patch_ab)

  return np.array(patches_ab), np.array(labels)

def Test_sup(ImgPH,ModelPath, test_files_PA, test_files_PB, train_labels, image_files, SavePath):


    imgData,labels = GenerateBatch(test_files_PA, test_files_PB, train_labels, 1)

    H4pt = homography_model(ImgPH, [128,128,2], 1)
    Saver = tf.train.Saver()
  
    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        imgData=np.array(imgData).reshape(1,128,128,2)
        
        FeedDict = {ImgPH: imgData}
        Predicted = sess.run(H4pt,FeedDict)
        h4pt_pred = np.array(Predicted)
        
    np.save(SavePath+"h4pt_Pred.npy",h4pt_pred)
def main():
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    # Parser.add_argument('--ModelPath', dest='ModelPath', default='../Checkpoints/unsupervised/99model.ckpt', help='Path to load latest model from, Default:ModelPath')
    # Parser.add_argument('--CheckPointPath', dest='CheckPointPath', default= '../Checkpoints/unsupervised/', help='Path to load latest model from, Default:CheckPointPath')
    # Parser.add_argument('--BasePath', dest='BasePath', default='../Data/Test_synthetic', help='Path to load images from, Default:BasePath')
    # Parser.add_argument('--SavePath', dest='SavePath', default='./Results/', help='Path of labels file, Default: ./Results/')
    Parser.add_argument('--ModelType', default='Unsup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
    base_dir = os.getcwd()
    base_path = base_dir + '/Data'
    Args = Parser.parse_args()
    ModelType = Args.ModelType
    test_files_PA = glob.glob( base_path + '/S_test/P_A'+ '/*.jpg')
    test_files_PB = glob.glob( base_path + '/S_test/P_B'+ '/*.jpg')
    image_files = glob.glob( base_path + '/S_test/I_A'+ '/*.jpg')
    NumTestSamples = len(test_files_PA)
    if ModelType == 'Unsup':
        ModelPath = base_path + '/unsupervised_checkpoints/74model.ckpt'
        SavePath = base_path + '/unsupervised_test_results/'
        print('The number of test samples is ' ,NumTestSamples)
        pointsList = np.load(base_path + '/S_test/points_list.npy')

        CornerPH = tf.placeholder(tf.float32, shape=(NumTestSamples, 4,2))
        PatchPairsPH = tf.placeholder(tf.float32, shape=(NumTestSamples, 128, 128 ,2))
        Patch2PH = tf.placeholder(tf.float32, shape=(NumTestSamples, 128, 128, 1))
        Images1PH = tf.placeholder(tf.float32, shape=(NumTestSamples, 240, 320, 1))
        patchIndicesPH = tf.placeholder(tf.int32, shape=(NumTestSamples, 128, 128 ,2))

        Test_unsupervised(PatchPairsPH, CornerPH, Patch2PH, Images1PH,patchIndicesPH, ModelPath, test_files_PA, test_files_PB, image_files, pointsList, SavePath+"unsupervised/", NumTestSamples)
         
        rand_i = np.random.randint(0,NumTestSamples-1, size=5)
        for i in rand_i:
            image = image_files[i]
            comparison = unsupervised_visualisation(i, base_path + '/S_test', SavePath, pointsList, image)
            cv2.imwrite(SavePath+'/comparison'+ str(i)+'.png',comparison)        
        print('Test results for unsupervised model are stored')
        
    else:
        ModelPath = base_path + '/supervised_checkpoints/19model.ckpt'
        SavePath = base_path + '/supervised_test_results/'
        train_labels = pd.read_csv(base_path + '/S_test/H_4pt.csv', index_col = False)
        train_labels = train_labels.to_numpy()
        ImgPH=tf.placeholder(tf.float32, shape=(len(test_files_PA), 128, 128, 2))
        Test_sup(ImgPH,ModelPath, test_files_PA, test_files_PB, train_labels, image_files, SavePath)
        pointsList = np.load(base_path + '/S_test/points_list.npy')
        rand_i = np.random.randint(0,len(test_files_PA)-1, size=5)
        for i in rand_i:
            image = image_files[i]
            comparison = supervised_visualisation(i, base_path + '/S_test', SavePath, train_labels, pointsList, image)
            cv2.imwrite(SavePath+'/comparison'+ str(i)+'.png',comparison)
        
        print('Results generated in the supervised test folder')     
if __name__ == '__main__':
    main()
 
