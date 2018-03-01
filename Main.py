#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:19:13 2017

@author: guido
"""

import sys

sys.path.insert(0, '/home/guido/NEU/DA 5030 Introduction to Data MiningMachine Learning/FinalProject/Laboratory/ImageProcessing/')
import ImageProcessing as im

sys.path.insert(0, '/home/guido/NEU/DA 5030 Introduction to Data MiningMachine Learning/FinalProject/Laboratory/SVM Lab/')
import SVM as svmlab

sys.path.insert(0, '/home/guido/NEU/DA 5030 Introduction to Data MiningMachine Learning/FinalProject/Laboratory/NeuralNet Lab/')
import NN5L as nn
    
#imagesPerCat, downsampling, colour, verticalEdge, horizontalEdge,  canny, slic, sizeImage, dim):
    
#verticalEdge = black and wite
#verticalEdge = black and wite
#color = bw
#downsamplig = color
#canny = bw
#slic = color
    
#Dimmension = 1 or 3 // depend if the filter is bw or color


#X=0
#(X,Y) = im.ImageLoader(10, downsampling=True, colour='red', verticalEdge=True, horizontalEdge=False, canny=False, slic=False, sizeImage=400, dim=1, vectorx=True, vectory=True)
#svmlab.performSVM(X,Y,150, 0.25, 2, 400,400)

#images=0
(images,y) = im.ImageLoader(200, downsampling=True, colour='', verticalEdge=False, horizontalEdge=False, canny=False, slic=True, sizeImage=200, dim=3, vectorx=False, vectory=False)

nn.NN5L(images = images, y=y, percent_training=0.25, classes=2,w=200, h=200, dim=3, banchSize=10, iteration=100)
