#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code

"""

import tensorflow as tf
import cv2
import sys
import os
import glob
import random
import matplotlib.pyplot as plt
from Network.Network import *
from Misc.MiscUtils import *
from Misc.DataUtils import *
import numpy as np
import time
import argparse
import shutil
import math as m
from tqdm import tqdm
from DataGenerator import data_generator

#function to load generated synthetic data for unsupervised training
def load_data_unsupervised(train_files_pa,train_files_pb,IA_files, points_list, batch_size, shuffle = True):

    patches_ab = []
    ca = []
    patches_b = []
    I_A = []

    for i in range(batch_size):
        index = random.randint(0, len(train_files_pa)-1) 
        patch_a = cv2.imread(train_files_pa[i], cv2.IMREAD_GRAYSCALE)
        patch_b = cv2.imread(train_files_pb[i], cv2.IMREAD_GRAYSCALE)
        image_a = cv2.imread(IA_files[i], cv2.IMREAD_GRAYSCALE)

        patch_a = np.float32(patch_a)
        patch_b = np.float32(patch_b) 
        image_a = np.float32(image_a)   

        #combine images along depth
        patch_ab = np.dstack((patch_a, patch_b))     
        corner1 = points_list[index, :, :, 0]
        
        
        patches_ab.append(patch_ab)
        ca.append(corner1)
        patches_b.append(patch_b.reshape(128, 128, 1))

        I_A.append(image_a.reshape(image_a.shape[0], image_a.shape[1], 1))

    patch_indices = getPatchIndices(np.array(ca))    
    return np.array(patches_ab), np.array(ca), np.array(patches_b), np.array(I_A), patch_indices
def TrainOperation_unsup(PatchPairsPH, CornerPH, Patch2PH, Images1PH, patchIndicesPH, train_files_pa, train_files_pb, image_files, pointsList, NumTrainSamples, ImageSize,NumEpochs, batch_size, SaveCheckPoint, CheckPointPath, LatestFile,LogsPath):
 
    # Predict output with forward pass
    patchb_pred, patchb_true, _ = unsupervised_HomographyNet(PatchPairsPH, CornerPH, Patch2PH, Images1PH,patchIndicesPH, batch_size)

    with tf.name_scope('Loss'):
        loss =  tf.reduce_mean(tf.abs(patchb_pred - patchb_true))

    with tf.name_scope('Adam'):
        Optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

    epoch_loss_ph = tf.placeholder(tf.float32, shape=None)
    loss_summary = tf.summary.scalar('Loss_per_iteration', loss)
    epoch_loss_summary = tf.summary.scalar('loss_per_epoch', epoch_loss_ph)
    # tf.summary.image('Anything you want', AnyImg)

    # Merge all summaries into a single operation
    MergedSummaryOP1 = tf.summary.merge([loss_summary])
    MergedSummaryOP2 = tf.summary.merge([epoch_loss_summary])
    # MergedSummaryOP = tf.summary.merge_all()

    # Setup Saver
    Saver = tf.train.Saver()

    with tf.Session() as sess:  

        if LatestFile is not None:
            Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
          # Extract only numbers from the name
            StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
            print('Latest checkpoint loaded ' + LatestFile + '....')
        else:
            sess.run(tf.global_variables_initializer())
            StartEpoch = 0
            print('New model initialized')

        # Tensorboard
        Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())
        
        L1_loss = []
        for Epochs in tqdm(range(StartEpoch, NumEpochs)):

            NumIterationsPerEpoch = int(NumTrainSamples/batch_size)
            Loss=[]
            epoch_loss=0

            for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):

                PatchPairsBatch, Corner1Batch, patch2Batch, Image1Batch, patchIndicesBatch = load_data_unsupervised(train_files_pa,train_files_pb,image_files, pointsList, batch_size, shuffle = True)
                FeedDict = {PatchPairsPH: PatchPairsBatch, CornerPH: Corner1Batch, Patch2PH: patch2Batch, Images1PH: Image1Batch, patchIndicesPH: patchIndicesBatch}

                _, LossThisBatch, Summary = sess.run([Optimizer, loss, MergedSummaryOP1], feed_dict=FeedDict)
                Loss.append(LossThisBatch)
                epoch_loss = epoch_loss + LossThisBatch

          # Tensorboard
            Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
            epoch_loss = epoch_loss/NumIterationsPerEpoch
            print('Completed epoch: ', Epochs)
            print("Epoch loss:  ",  np.mean(Loss), "\n")
            L1_loss.append(np.mean(Loss))
            SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
            Saver.save(sess, save_path=SaveName)
            print('\n' + SaveName + ' Model Saved')
            Summary_epoch = sess.run(MergedSummaryOP2,feed_dict={epoch_loss_ph: epoch_loss})
            Writer.add_summary(Summary_epoch,Epochs)
            Writer.flush()

        np.savetxt(LogsPath + "losshistory_unsup.txt", np.array(L1_loss), delimiter = ",")

# geerating batch data for supervised learning
def GenerateBatch(train_files_pa, train_files_pb, TrainLabels, MiniBatchSize):

    patches_ab = []
    labels = []

    for i in range(MiniBatchSize):
        index = random.randint(0, len(train_files_pa)-1) 
        patch_a = cv2.imread(train_files_pa[i], cv2.IMREAD_GRAYSCALE)
        patch_b = cv2.imread(train_files_pb[i], cv2.IMREAD_GRAYSCALE)

        patch_a = np.float32(patch_a)
        patch_b = np.float32(patch_b)  

        #combine images along depth
        patch_ab = np.dstack((patch_a, patch_b))     
        labels.append(TrainLabels[i,:].reshape(8,))
        patches_ab.append(patch_ab)

    return np.array(patches_ab), np.array(labels)

def TrainOperation_Sup(ImgPH, LabelPH,CornerPH, I2PH, I1FullPH, train_files_PA, train_files_PB, train_labels, NumTrainSamples, ImageSize,
  NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
  DivTrain, LatestFile, LogsPath):
 
    H4pt = homography_model(ImgPH, ImageSize, MiniBatchSize)

    with tf.name_scope('Loss'):
        loss = tf.sqrt(tf.reduce_sum((tf.squared_difference(H4pt,LabelPH))))

    with tf.name_scope('Adam'):
        Optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    # Tensorboard
    # Create a summary to monitor loss tensor
    EpochLossPH = tf.placeholder(tf.float32, shape=None)
    loss_summary = tf.summary.scalar('LossEveryIter', loss)
    epoch_loss_summary = tf.summary.scalar('LossPerEpoch', EpochLossPH)
    # tf.summary.image('Anything you want', AnyImg)

    # Merge all summaries into a single operation
    MergedSummaryOP1 = tf.summary.merge([loss_summary])
    MergedSummaryOP2 = tf.summary.merge([epoch_loss_summary])
    # MergedSummaryOP = tf.summary.merge_all()

    # Setup Saver
    Saver = tf.train.Saver()
    with tf.Session() as sess:       
        if LatestFile is not None:
            Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
        # Extract only numbers from the name
            StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
            print('Latest checkpoint loaded ' + LatestFile)
        else:
            sess.run(tf.global_variables_initializer())
            StartEpoch = 0
            print('Initialized New model')

    # Tensorboard
        Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())

        for Epochs in tqdm(range(StartEpoch, NumEpochs)):
            NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
            Loss=[]
            epoch_loss=0
            for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                ImgBatch,LabelBatch = GenerateBatch(train_files_PA, train_files_PB, train_labels, MiniBatchSize)
                FeedDict = {ImgPH: ImgBatch, LabelPH: LabelBatch}

                _, LossThisBatch, Summary = sess.run([Optimizer, loss, MergedSummaryOP1], feed_dict=FeedDict)
                Loss.append(LossThisBatch)
                epoch_loss = epoch_loss + LossThisBatch
                if PerEpochCounter % SaveCheckPoint == 0:
                    SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
                    Saver.save(sess,  save_path=SaveName)
                    print('\n' + SaveName + ' Model Saved...')
        Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)


        epoch_loss = epoch_loss/NumIterationsPerEpoch

        print(np.mean(Loss))
        SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
        Saver.save(sess, save_path=SaveName)
        print('\n' + SaveName + ' Model Saved...')
        Summary_epoch = sess.run(MergedSummaryOP2,feed_dict={EpochLossPH: epoch_loss})
        Writer.add_summary(Summary_epoch,Epochs)
        Writer.flush()

def main():
    base_dir = os.getcwd()
    data_dir = base_dir + '/Data'
    # print(data_dir)
    data_generator(data_dir)
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelType', default='Unsup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
    Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=10, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    
    
    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    DivTrain = float(Args.DivTrain)
    batch_size = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    ModelType = Args.ModelType
    ImageSize = [128,128,1]
    SaveCheckPoint = True
    patchSize = 128
    train_files_PA = glob.glob( data_dir + '/S_train/P_A'+ '/*.jpg')
    train_files_PB = glob.glob( data_dir + '/S_train/P_B'+ '/*.jpg')
    image_files = glob.glob( data_dir + '/S_train/I_A'+ '/*.jpg')
    NumTrainSamples = len(train_files_PA)

    if ModelType == 'Unsup':
        tf.reset_default_graph()
        print(" UnSupervised  Model Training using Tensorflow ...")
        CheckPointPath = data_dir + '/unsupervised_checkpoints/'
        LogsPath = data_dir + '/unsupervised_logs/'
        if(not (os.path.isdir(LogsPath))):
            print(LogsPath, "Creating this folder")
            os.makedirs(LogsPath)

        if(not (os.path.isdir(CheckPointPath))):
            print(CheckPointPath, "Creating this folder")
            os.makedirs(CheckPointPath)
        # train_files_PA = glob.glob( data_dir + 'S_train/P_A'+ '/*.jpg')
        # train_files_PB = glob.glob( data_dir + '/S_train/P_B'+ '/*.jpg')
        # image_files = glob.glob( data_dir + '/S_train/I_A'+ '/*.jpg')
        # NumTrainSamples = len(train_files_PA)

        print("Number of Training Samples:..", NumTrainSamples)

        pointsList = np.load(data_dir + '/S_train/points_list.npy')

        CornerPH = tf.placeholder(tf.float32, shape=(batch_size, 4,2))
        PatchPairsPH = tf.placeholder(tf.float32, shape=(batch_size, 128, 128 ,2))
        Patch2PH = tf.placeholder(tf.float32, shape=(batch_size, 128, 128, 1))
        Images1PH = tf.placeholder(tf.float32, shape=(batch_size, 240, 320, 1))
        patchIndicesPH = tf.placeholder(tf.int32, shape=(batch_size, 128, 128 ,2))

        LatestFile = None

        TrainOperation_unsup(PatchPairsPH, CornerPH, Patch2PH, Images1PH, patchIndicesPH, train_files_PA, train_files_PB, image_files, pointsList, NumTrainSamples, ImageSize,NumEpochs, batch_size, SaveCheckPoint, CheckPointPath, LatestFile,LogsPath)
    else:
        tf.reset_default_graph()
        print(" Supervised  Model Training using Tensorflow")
        CheckPointPath = data_dir + '/supervised_checkpoints/'
        LogsPath = data_dir + '/supervised_logs/'
        if(not (os.path.isdir(LogsPath))):
            print(LogsPath, "Creating this folder")
            os.makedirs(LogsPath)

        if(not (os.path.isdir(CheckPointPath))):
            print(CheckPointPath, "Creating this folder")
            os.makedirs(CheckPointPath)
        if LoadCheckPoint==1:
            LatestFile = FindLatestModel(CheckPointPath)
        else:
            LatestFile = None
        
        train_labels = pd.read_csv(data_dir + '/S_train/H_4pt.csv', index_col = False)
        train_labels = train_labels.to_numpy()
        print("Number of Training Samples:", NumTrainSamples)
        ImgPH = tf.placeholder(tf.float32, shape=(batch_size, patchSize, patchSize, 2))
        LabelPH = tf.placeholder(tf.float32, shape=(batch_size, 8))
        CornerPH = tf.placeholder(tf.float32, shape=(batch_size, 4,2))
        I2PH = tf.placeholder(tf.float32, shape=(batch_size, 128, 128,1))
        I1FullPH = tf.placeholder(tf.float32, shape=(batch_size, ImageSize[0], ImageSize[1],ImageSize[2]))

        TrainOperation_Sup(ImgPH, LabelPH,CornerPH, I2PH, I1FullPH, train_files_PA, train_files_PB, train_labels, NumTrainSamples, ImageSize,
        NumEpochs, batch_size, SaveCheckPoint, CheckPointPath,
        DivTrain, LatestFile, LogsPath)

if __name__ == '__main__':
    main()