#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 11:33:15 2017

@author: guido
"""
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import os
import glob # for bulk file import
import numpy as np
from tensorflow.python.framework import ops
import random as rn


#==============================================================================
# User configuration
#==============================================================================

w=400
h=400
dim=1
classes=2
percent_training = 0.25

##############################################################################

rn.seed(3)

def guidoBatch(images, y, minbatch_size):
    rango = range(0,images.shape[0])
    a = rn.sample(rango , minbatch_size)
    return (images[a,:], y[a])

###############################################################################
# Split into a training set and a test set using a stratified k fold
    
# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    images, y, test_size=percent_training , random_state=42)
    
# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
#mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# input X: 150x150 
#grayscale images, 
#the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, w, h, dim])

# correct answers will go here (this is the place of the labels, 2 is the number of categories)
Y_ = tf.placeholder(tf.float32, [None, classes])


# weights W[784, 10]   784=150*150
W = tf.Variable(tf.zeros([w*h*dim, classes]))

# biases b[2]
b = tf.Variable(tf.zeros([classes]))

# flatten the images into a single line of pixels
# -1 in the shape definition means "the only possible dimension that will preserve the number of elements"
XX = tf.reshape(X, [-1, w*h*dim])

# The model
Y = tf.nn.softmax(tf.matmul(XX, W) + b)

# loss function: cross-entropy = - sum( Y_i * log(Yi) )
#                           Y: the computed output vector
#                           Y_: the desired output vector

# cross-entropy
# log takes the log of each element, * multiplies the tensors element by element
# reduce_mean will add all the components in the tensor
# so here we end up with the total cross-entropy for all images in the batch
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0  # normalized for batches of 100 images,
                                                          # *10 because  "mean" included an unwanted division by 10

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training, learning rate = 0.005
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    # load batch of images and correct answers
#    batch_X, batch_Y = mnist.train.next_batch(100)
    batch_X, batch_Y = guidoBatch(X_train, y_train, 10)
    train_data={X: batch_X, Y_: batch_Y}

    # train
    sess.run(train_step, feed_dict=train_data)


batch_X, batch_Y = guidoBatch(images, y, 10)
train_data={X: batch_X, Y_: batch_Y}
a,c = sess.run([accuracy, cross_entropy], feed_dict=train_data)

#==============================================================================
# TESTING
#==============================================================================

test_data={X: X_test, Y_: y_test}
a,c = sess.run([accuracy, cross_entropy], feed_dict=test_data)