#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:55:55 2017

@author: guido
"""

from sklearn.model_selection import train_test_split
from time import time
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#import glob # for bulk file import
#import os



#==============================================================================
# This algorith asumes the images already loaded in X
#==============================================================================

#==============================================================================
# USER PARAMETER
#==============================================================================
def performSVM(X,y,n_components, percent_training, n_classes, h, w):

#    percent_training = 0.25
#    #number of principal component to use in SVM
#    n_components = 300 #150
#    #number of clases
#    n_classes = 2
    
    ###############################################################################
    # Split into a training set and a test set using a stratified k fold
    
    # split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=percent_training , random_state=42)
    
    
    print("testing with:", y_test.shape, "images")
    
    #realease memory
    #X=0
    
    ###############################################################################
    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction
    
    
    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train.shape[0]))
    t0 = time()
    pca = PCA(n_components=n_components, svd_solver='randomized',
              whiten=True).fit(X_train)
    print("done in %0.3fs" % (time() - t0))
    
    #eigenfaces = pca.components_.reshape((n_components, h, w))
    
    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))
    
    ###############################################################################
    # Train a SVM classification model
    
    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    
    
    ###############################################################################
    # Quantitative evaluation of the model quality on the test set
    
    target_names = ['trash','grafiti', 'other']
    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    print("done in %0.3fs" % (time() - t0))
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))