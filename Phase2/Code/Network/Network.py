"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def CIFAR10Model(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """

    #############################
    # Fill your network here!
    #############################
    channel_1 = 32
    channel_2 = 16
    num_classes = 10
    initializer = tf.initializers.VarianceScaling(scale=2.0)
    conv1 = tf.keras.layers.Conv2D(filters = channel_1,kernel_size = (5,5), strides =(1,1), padding =
              'valid',activation = tf.nn.relu,kernel_initializer = initializer)(Img)

    # Convolutional layer with 3 x 3 kernels, with zero-padding of 1 + RELU
    conv2 = tf.keras.layers.Conv2D(filters = channel_2,kernel_size = (3,3),strides = (1,1), padding =
                                    'valid',activation
                                    = tf.nn.relu, kernel_initializer= initializer)(conv1)

    
    flatten = tf.keras.layers.Flatten()(conv2)
    # FCN and softmax activation
    fc = tf.keras.layers.Dense(num_classes,kernel_initializer=initializer)(flatten)


    # To flatten scores
    
    prSoftMax = tf.nn.softmax(logits = fc)
  
    prLogits = fc

    return prLogits, prSoftMax

def CIFAR10Resnet(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """

    #############################
    # Fill your network here!
    #############################

    num_classes = 10

    initializer = tf.initializers.VarianceScaling(scale=2.0)
    
    # Block 1
    conv1 = tf.keras.layers.Conv2D(filters = 64,kernel_size = (7,7), strides =(2,2), padding =
              ('same'))(Img)
    batch1 = tf.keras.layers.BatchNormalization(synchronized=True)(conv1)
    relu1 = tf.keras.layers.ReLU()(batch1)
    op1 = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding=('valid'))(relu1)

    # Block 2
      # Layer 1
    conv2_1 = tf.keras.layers.Conv2D(filters = 64,kernel_size = (3,3), strides =(1,1), padding =
              ('same'))(op1)
    batch2_1 = tf.keras.layers.BatchNormalization(synchronized=True)(conv2_1)
    relu2_1 = tf.keras.layers.ReLU()(batch2_1)
    conv2_2 = tf.keras.layers.Conv2D(filters = 64,kernel_size = (3,3), strides =(1,1), padding =
              ('same'))(relu2_1)
    batch2_2 = tf.keras.layers.BatchNormalization(synchronized=True)(conv2_2)
    var   = tf.keras.layers.Dropout(0.5)(batch2_2)
    op2_1 = tf.keras.layers.ReLU()(var+op1)
      # Layer 2
    conv2_3 = tf.keras.layers.Conv2D(filters = 64,kernel_size = (3,3), strides =(1,1), padding =
              ('same'))(op2_1)
    batch2_3 = tf.keras.layers.BatchNormalization(synchronized=True)(conv2_3)
    relu2_3 = tf.keras.layers.ReLU()(batch2_3)
    conv2_4 = tf.keras.layers.Conv2D(filters = 64,kernel_size = (3,3), strides =(1,1), padding =
              ('same'))(relu2_3)
    batch2_4 = tf.keras.layers.BatchNormalization(synchronized=True)(conv2_4)
    var   = tf.keras.layers.Dropout(0.5)(batch2_4)
    op2 = tf.keras.layers.ReLU()(var+op2_1)

    # Block 3
      # Layer 1
    conv3_1 = tf.keras.layers.Conv2D(filters = 128,kernel_size = (3,3), strides =(2,2), padding =
              ('same'))(op2)
    batch3_1 = tf.keras.layers.BatchNormalization(synchronized=True)(conv3_1)
    relu3_1 = tf.keras.layers.ReLU()(batch3_1)
    conv3_2 = tf.keras.layers.Conv2D(filters = 128,kernel_size = (3,3), strides =(1,1), padding =
              ('same'))(relu3_1)
    batch3_2 = tf.keras.layers.BatchNormalization(synchronized=True)(conv3_2)
    dropout3_2 = tf.keras.layers.Dropout(0.5)(batch3_2)
    op2 = tf.keras.layers.Conv2D(filters = 128,kernel_size = (1,1), strides =(2,2), padding =
              ('valid'))(op2)
    op3_1 = tf.keras.layers.ReLU()(dropout3_2+op2)
      # Layer 2
    conv3_3 = tf.keras.layers.Conv2D(filters = 128,kernel_size = (3,3), strides =(1,1), padding =
              ('same'))(op3_1)
    batch3_3 = tf.keras.layers.BatchNormalization(synchronized=True)(conv3_3)
    relu3_3 = tf.keras.layers.ReLU()(batch3_3)
    conv3_4 = tf.keras.layers.Conv2D(filters = 128,kernel_size = (3,3), strides =(1,1), padding =
              ('same'))(relu3_3)
    batch3_4 = tf.keras.layers.BatchNormalization(synchronized=True)(conv3_4)
    var   = tf.keras.layers.Dropout(0.5)(batch3_4)
    op3 = tf.keras.layers.ReLU()(var + op3_1)

    # Block 4
      #Layer 1
    conv4_1 = tf.keras.layers.Conv2D(filters = 256,kernel_size = (3,3), strides =(2,2), padding =
              ('same'))(op3)
    batch4_1 = tf.keras.layers.BatchNormalization(synchronized=True)(conv4_1)
    relu4_1 = tf.keras.layers.ReLU()(batch4_1)
    conv4_2 = tf.keras.layers.Conv2D(filters = 256,kernel_size = (3,3), strides =(1,1), padding =
              ('same'))(relu4_1)
    batch4_2 = tf.keras.layers.BatchNormalization(synchronized=True)(conv4_2)
    dropout4_2 = tf.keras.layers.Dropout(0.5)(batch4_2)
    op3 = tf.keras.layers.Conv2D(filters = 256,kernel_size = (1,1), strides =(2,2), padding =
              ('valid'))(op3)
    op4_1 = tf.keras.layers.ReLU()(dropout4_2+op3)
      # Layer 2
    conv4_3 = tf.keras.layers.Conv2D(filters = 256,kernel_size = (3,3), strides =(1,1), padding =
              ('same'))(op4_1)
    batch4_3 = tf.keras.layers.BatchNormalization(synchronized=True)(conv4_3)
    relu4_3 = tf.keras.layers.ReLU()(batch4_3)
    conv4_4 = tf.keras.layers.Conv2D(filters = 256,kernel_size = (3,3), strides =(1,1), padding =
              ('same'))(relu4_3)
    batch4_4 = tf.keras.layers.BatchNormalization(synchronized=True)(conv4_4)
    var   = tf.keras.layers.Dropout(0.5)(batch4_4)
    op4 = tf.keras.layers.ReLU()(var + op4_1)

    # Block 5
      # Layer 1
    conv5_1 = tf.keras.layers.Conv2D(filters = 512,kernel_size = (3,3), strides =(2,2), padding =
              ('same'))(op4)
    batch5_1 = tf.keras.layers.BatchNormalization(synchronized=True)(conv5_1)
    relu5_1 = tf.keras.layers.ReLU()(batch5_1)
    conv5_2 = tf.keras.layers.Conv2D(filters = 512,kernel_size = (3,3), strides =(1,1), padding =
              ('same'))(relu5_1)
    batch5_2 = tf.keras.layers.BatchNormalization(synchronized=True)(conv5_2)
    dropout5_2 = tf.keras.layers.Dropout(0.5)(batch5_2)
    op4 = tf.keras.layers.Conv2D(filters = 512,kernel_size = (1,1), strides =(2,2), padding =
              ('valid'))(op4)
    op5_1 = tf.keras.layers.ReLU()(dropout5_2+op4)
      # Layer 2
    conv5_3 = tf.keras.layers.Conv2D(filters = 512,kernel_size = (3,3), strides =(1,1), padding =
              ('same'))(op5_1)
    batch5_3 = tf.keras.layers.BatchNormalization(synchronized=True)(conv5_3)
    relu5_3 = tf.keras.layers.ReLU()(batch5_3)
    conv5_4 = tf.keras.layers.Conv2D(filters = 512,kernel_size = (3,3), strides =(1,1), padding =
              ('same'))(relu5_3)
    batch5_4 = tf.keras.layers.BatchNormalization(synchronized=True)(conv5_4)
    var   = tf.keras.layers.Dropout(0.5)(batch5_4)
    op5 = tf.keras.layers.ReLU()(var + op5_1)
    
    # Final Block
    avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(1,1), strides=(1,1))(op5)
    dense = tf.keras.layers.Dense(1000,kernel_initializer=initializer)(avg_pool)
    flatten = tf.keras.layers.Flatten()(dense)
    # FCN and softmax activation
    fc = tf.keras.layers.Dense(num_classes,kernel_initializer=initializer)(flatten)


    # To flatten scores
    
    prSoftMax = tf.nn.softmax(logits = fc)
  
    prLogits = fc

    return prLogits, prSoftMax
