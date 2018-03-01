#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:39:54 2017

@author: guido
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from skimage import color
#import skimage.filters as filters
#from skimage.transform import hough_circle
#from skimage.feature import peak_local_max 
#from skimage import morphology
#from skimage.draw import circle_perimeter
from skimage import img_as_float, img_as_ubyte
from skimage import segmentation as seg
#from skimage.morphology import watershed
#from scipy import ndimage as nd
#from scipy.ndimage import convolve

from skimage import feature
#from sklearn.model_selection import train_test_split
from skimage.transform import resize
#from time import time
#from sklearn.decomposition import PCA
#from sklearn.model_selection import GridSearchCV
#from sklearn.svm import SVC
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
import glob # for bulk file import
import os


#==============================================================================
# USER PARAMETERS
#==============================================================================
def ImageLoader(imagesPerCat, downsampling, colour, verticalEdge, horizontalEdge,  canny, slic, sizeImage, dim, vectorx, vectory):
    testMode = False
    
    resizeImage = True
    if (testMode):
        #how many images you want in each category
        imagesPerCat = 0 # 0 = all possible
        
        sizeImage = 400
        #w,h = (sizeImage,sizeImage)
        resizeImage = False
        verticalEdge = False
        downsampling = True
        canny = True
        slic=False
    
    
    # Set defaults
    if (testMode):
        #%matplotlib inline
        plt.rcParams['image.cmap'] = 'gray' # Display grayscale images in... grayscale.
        plt.rcParams['image.interpolation'] = 'none' # Use nearest-neighbour
        plt.rcParams['figure.figsize'] = 10, 10
    
    #==============================================================================
    # LOADING IMAGES TO MEMORY
    #==============================================================================
    
    #Set the working directory
    os.getcwd()
    os.chdir("/home/guido/NEU")
    
    ##loading trash images path
    imgpathsTrash = glob.glob("./GitHub/Image_Analysis_in_Python/images/trash/*.JPEG")
    imgpathsTrash = glob.glob("./DA 5030 Introduction to Data MiningMachine Learning/FinalProject/311pictures/Trash/*.jpg")
    
    
    ##loading grafiti images path
    imgpathsGrafiti = glob.glob("./DA 5030 Introduction to Data MiningMachine Learning/FinalProject/311pictures/Grafiti/*.jpg")
    
    
    ##loading other images path
    imgpathsOther = glob.glob("./DA 5030 Introduction to Data MiningMachine Learning/FinalProject/311pictures/Other/*.jpg")
    
    
    
    if imagesPerCat != 0 :
        imgpathsTrash = imgpathsTrash[0:imagesPerCat] 
        imgpathsGrafiti = imgpathsGrafiti[0:imagesPerCat]
        imgpathsOther = imgpathsOther[0:imagesPerCat]
    
    
    imgpaths = imgpathsTrash + imgpathsGrafiti + imgpathsOther
    imgpaths = imgpathsTrash + imgpathsGrafiti 
    
    
    #==============================================================================
    # Creating labels
    #==============================================================================
    
    if (vectory):
        #return y as a verctor (better for SVM)
        y1 = np.zeros(len(imgpathsTrash))
        y2 = np.zeros(len(imgpathsGrafiti)) + 1
        y3 = np.zeros(len(imgpathsOther)) + 2
        
        y= np.concatenate((y1,y2, y3), axis=0)
        y= np.concatenate((y1,y2), axis=0)
    
        print("Numer of images of category y1", y1.shape)
        print("Numer of images of category y2", y2.shape)
    else:
        #return y as a matrix (better for neural networks)
        y1 = [(0,1) for x in range(len(imgpathsTrash))]
        y2 = [(1,0) for x in range(len(imgpathsGrafiti))]
    
        y= np.concatenate((y1,y2), axis=0)
    
    
    #==============================================================================
    # PROCESSING IMAGES
    #==============================================================================
    
    
    ##extract the number of features
    #img = mpimg.imread(imgpaths[1])
    #img = downsample_image(img, 3)
    #img = find_vertical_edges(img)
    #img = resize(img, (sizeImage, sizeImage))
    #img_vec =  img.reshape((1, -1))
    #n_features = img_vec.shape[1]
    
    n_features = sizeImage*sizeImage*dim
    #create the master vector
    if (vectorx):
        X = np.zeros((len(imgpaths), n_features))
    else:
        X = np.zeros((len(imgpaths), sizeImage,sizeImage,dim))
        
    #Redifininf the opening of the images
    for i,x in enumerate(imgpaths):
            
        #read the image in a vector
        img = mpimg.imread(x)
                
        if(downsampling):
            #downsampling the image
            img = downsample_image(img, 3)
            
        if (colour == "red" or colour == "green" or colour == "blue"):
            img = plot_rgb_components(img, colour)
        
        
        if (verticalEdge):
            #finding edge
            img = find_vertical_edges(img)
            
        if (horizontalEdge):
            img = find_horizontal_edges(img)
    
        if (canny):
            imgbw = img_as_float(color.rgb2grey(img))
            img = canny_image(imgbw)
            
        if (slic):
            img = slic_image(img)

        if(resizeImage):
            #resize the image
            img = resize(img, (sizeImage, sizeImage)) 
    
        #flatend the image to a vector
        if (vectorx):
            img_vec =  img.reshape((1, -1)) 
            if (img_vec.shape[1] != n_features):
                print('image not procesed: %s' %x)
            else:
                #store the image 
                X[i] = img_vec
        else:
            img_matrix = img.reshape(sizeImage,sizeImage, dim)
            X[i] = img_matrix 
            
    return (X,y)

#==============================================================================
# EVALUATION 
#==============================================================================
    def testing():
        if (testMode): 
            imgset = [mpimg.imread(x) for x in imgpaths]
        
            #plot original
            plt.figure()
            for i,img in enumerate(imgset):
                plt.subplot(1, len(imgset), i+1)
                plt.imshow(img, cmap = 'gray')
            
            
            ##show Canny edge detector
            # Apply to image set
            sigma = 2.0
            for i,img in enumerate(imgset):
                img = resize(img, (400, 400))
                imgbw = img_as_float(color.rgb2grey(img))
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(imgbw)
                plt.subplot(1, 2, 2)
                plt.imshow(feature.canny(imgbw, sigma))
            
            for i,img in enumerate(imgset):
                plt.figure()
                plt.subplot(1, 3, 1)
                plt.title('Original')
                plt.imshow(img)
                plt.subplot(1, 3, 2)
                plt.title('Horizontal Edges')
                plt.imshow(find_horizontal_edges(img))
                plt.subplot(1, 3, 3)
                plt.title('Vertical Edges')
                plt.imshow(find_vertical_edges(img))
            
            plt.imshow(image)
            plt.imshow(img)
            plt.imshow(img_ds)


    #==============================================================================
    # IMAGES FILTERS
    #==============================================================================
    
##EDGE DETECTION
# Find horizontal edges using a simple shifting method
def find_horizontal_edges(img):
    imgbw = img_as_float(color.rgb2grey(img))
    return np.abs(imgbw[:, 1:] - imgbw[:, :-1])

# Find vertical edges using a simple shifting method
def find_vertical_edges(img):
    imgbw = img_as_float(color.rgb2grey(img))
    return np.abs(imgbw[1:, :] - imgbw[:-1, :])

##DOWNSAMPLE
# Downsample an image by skipping indicies
def downsample_image(img, skip):
     return img[::skip,::skip]

#Canny filter
def canny_image(img):
    sigma = 2.0
    return feature.canny(img, sigma)

#Slic
def slic_image(img):
    ns=12
    compact=70
    sigma=2.0
    
    # Calculate the mean color of slic regions, from the SciKit tutorial
    def mean_color(image, labels):
        out = np.zeros_like(image)
        for label in np.unique(labels):
            indices = np.nonzero(labels == label)
            out[indices] = np.mean(image[indices], axis=0)
        return out

    def plot_slic_segmentation(img, ns, c, s):
        labels = seg.slic(img, n_segments=ns, compactness=c, sigma=s, enforce_connectivity=True)
        return mean_color(img, labels)
    
    return plot_slic_segmentation(img, ns, compact, sigma)

# Show rgb as four plots
def plot_rgb_components(img, colour):
    if img.ndim == 2: # convert grayscale to rgb
        rgb = color.gray2rgb(img)
    elif img.ndim == 3:
        rgb = img
    else: # Not an image
        print("Must pass a valid RGB or grayscale image")
    
    if (colour=='red'):
        return rgb[:,:,0] # Red

    if (colour=='green'):
        return rgb[:,:,1] # green
        
    if (colour=='blue'):
        return rgb[:,:,2] # blue