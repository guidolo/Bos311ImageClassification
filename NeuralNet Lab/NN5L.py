#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 11:33:15 2017

@author: guido
"""
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random as rn


#==============================================================================
# User configuration
#==============================================================================
def NN5L(images,y, percent_training, classes, w,h, dim, banchSize, iteration):
        
#    w=400
#    h=400
#    dim=1
#    classes=2
#    percent_training = 0.25                                                                                                                                 
#    banchSize = 10
    
    ##############################################################################
    
    rn.seed(3)
    
    def guidoBatch(images, y, minbatch_size):
        rango = range(0,images.shape[0])
        a = rn.sample(rango , minbatch_size)
        return (images[a,:], y[a])
    
    def guidoQuiye(images, y, i, minbatch_size):
        start = i*minbatch_size
        end = (i+1)*minbatch_size
        size = images.shape[0]
        epoch = 0
        if( end >= size):
            epoch = end/size
            i = (end - (size*epoch)) / minbatch_size
            start = i*minbatch_size
            end = (i+1)*minbatch_size
            if (end > size):
                end = size
        a = range(start, end)
        #print("Epoch: " + str( (epoch) ) + " iter: " + str(i) + " start: " + str(start) + " end: " + str(end))
        return (epoch, images[a,:], y[a])
    
    ###############################################################################
    # Split into a training set and a test set using a stratified k fold
        
    # split into a training and testing set
    
    X_train, X_test, y_train, y_test = train_test_split(
        images, y, test_size=percent_training , random_state=42)
    
    images =0 
    #==============================================================================
    # TENSOR FLOW
    #==============================================================================
    
    # input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
    X = tf.placeholder(tf.float32, [None, w, h, dim])
    # correct answers will go here
    Y_ = tf.placeholder(tf.float32, [None, classes])
    
    # five layers and their number of neurons (tha last layer has 10 softmax neurons)
    L = 100
    M = 50
    N = 10
    O = 5
    # Weights initialised with small random values between -0.2 and +0.2
    # When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10
    W1 = tf.Variable(tf.truncated_normal([w*h*dim, L], stddev=0.1))  # 784 = 28 * 28
    B1 = tf.Variable(tf.zeros([L]))
    W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
    B2 = tf.Variable(tf.zeros([M]))
    W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
    B3 = tf.Variable(tf.zeros([N]))
    W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
    B4 = tf.Variable(tf.zeros([O]))
    W5 = tf.Variable(tf.truncated_normal([O, classes], stddev=0.1))
    B5 = tf.Variable(tf.zeros([classes]))
    
    # The model
    XX = tf.reshape(X, [-1, w*h*dim])
    Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + B1)
    Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)
    Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3)
    Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4)
    Ylogits = tf.matmul(Y4, W5) + B5
    Y = tf.nn.softmax(Ylogits)
    
    # cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
    # TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
    # problems with log(0) which is NaN
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    cross_entropy = tf.reduce_mean(cross_entropy)*banchSize
    
    # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # matplotlib visualisation
    allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])], 0)
    allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1])], 0)
    #I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)
    #It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)
    #datavis = tensorflowvisu.MnistDataVis()
    
    # training step, learning rate = 0.003
    learning_rate = 0.003
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    
    # init
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    oldepoch=-1
    test_data={X: X_test, Y_: y_test}


    for i in range(iteration):
        # load batch of images and correct answers
    #    batch_X, batch_Y = mnist.train.next_batch(100)
        #batch_X, batch_Y = guidoBatch(X_train, y_train, banchSize)
        epoch, batch_X, batch_Y = guidoQuiye(X_train, y_train, i, banchSize )
        train_data={X: batch_X, Y_: batch_Y}
    
        # train
        sess.run(train_step, feed_dict=train_data)
        a, c,  w, b = sess.run([accuracy, cross_entropy,  allweights, allbiases], {X: batch_X, Y_: batch_Y})
        
        if (epoch != oldepoch):
            print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")
            a,c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
            print(str(i) + ": ********* epoch: " + str(oldepoch)  + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
            oldepoch = epoch
    
    
    #==============================================================================
    # TESTING
    #==============================================================================
    
    
    a, c = sess.run([accuracy, cross_entropy], test_data)
    print(str(i) + ": ********* epoch final"  + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
     
    sess=0
    
    
