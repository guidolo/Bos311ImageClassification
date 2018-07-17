#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 16:52:00 2017

@author: guido
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import os
import glob # for bulk file import
import numpy as np
from tensorflow.python.framework import ops

def read_labeled_image_list(image_list_file):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    for line in f:
        filename, label = line[:-1].split(' ')
        filenames.append(filename)
        labels.append(int(label))
    return filenames, labels

def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_png(file_contents, channels=3)
    return example, label

# Reads pfathes of images together with their labels
image_list, label_list = read_labeled_image_list(filename)


os.chdir("/home/guido/NEU")
image_list = glob.glob("./GitHub/Image_Analysis_in_Python/images/trash/*.JPEG")
label_list = np.zeros(len(image_list ))


images = ops.convert_to_tensor(image_list, dtype='string')
labels = ops.convert_to_tensor(label_list, dtype='int32')

# Makes an input queue
input_queue = tf.train.slice_input_producer([images, labels], num_epochs = 1000, shuffle=True)

image, label = read_images_from_disk(input_queue)

# Optional Preprocessing or Data Augmentation
# tf.image implements most of the standard image augmentation
image = preprocess_image(image)
label = preprocess_label(label)

# Optional Image and Label Batching
image_batch, label_batch = tf.train.batch([image, label], batch_size=100)


##############################################################################
# Typical setup to include TensorFlow.
import tensorflow as tf

# Make a queue of file names including all the JPEG images files in the relative
# image directory.
filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("./GitHub/Image_Analysis_in_Python/images/trash/*.JPEG"))




# Read an entire image file which is required since they're JPEGs, if the images
# are too large they could be split in advance to smaller files or use the Fixed
# reader to split up the file.
image_reader = tf.WholeFileReader()

# Read a whole file from the queue, the first returned value in the tuple is the
# filename which we are ignoring.
_, image_file = image_reader.read(filename_queue)

# Decode the image as a JPEG file, this will turn it into a Tensor which we can
# then use in training.
image = tf.image.decode_jpeg(image_file)

# Start a new session to show example output.
with tf.Session() as sess:
    # Required to get the filename matching to run.
    tf.initialize_all_variables().run()

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Get an image tensor and print its value.
    image_tensor = sess.run([image])
    print(image_tensor)

    # Finish off the filename queue coordinator.
    coord.request_stop()
coord.join(threads)


##############################################################################
# Typical setup to include TensorFlow.
import tensorflow as tf
import os

os.chdir("/home/guido/NEU")
# Make a queue of file names including all the JPEG images files in the relative
# image directory.
filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("./GitHub/Image_Analysis_in_Python/images/trash/*.JPEG"))


# Read an entire image file which is required since they're JPEGs, if the images
# are too large they could be split in advance to smaller files or use the Fixed
# reader to split up the file.
image_reader = tf.WholeFileReader()

# Read a whole file from the queue, the first returned value in the tuple is the
# filename which we are ignoring.
_, image_file = image_reader.read(filename_queue)

image_orig = tf.image.decode_jpeg(image_file)
image = tf.image.resize_images(image_orig, [224, 224])
image.set_shape((224, 224, 3))
batch_size = 50
num_preprocess_threads = 1
min_queue_examples = 256

images = tf.train.shuffle_batch([image],
                                        batch_size=batch_size,
                                        num_threads=num_preprocess_threads,
                                        capacity=min_queue_examples + 3 * batch_size,
                                        min_after_dequeue=min_queue_examples)

#init = tf.global_variables_initializer()
#sess = tf.Session()
#sess.run(init)

# Start a new session to show example output.
with tf.Session() as sess:
    # Required to get the filename matching to run.
    tf.initialize_all_variables().run()

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)


    # Get an image tensor and print its value.
    image_tensor = sess.run([image])
    print(image_tensor)

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
##############################################################################


#imgpaths = imgpathsTrash

img = mpimg.imread(imgpaths[1])
#img = downsample_image(img, 3)
img = find_vertical_edges(img)
img = resize(img, (150, 150))
#img_vec =  img.reshape((1, -1))
#n_features = img_vec.shape[1]

#create the master vector
images = np.zeros((len(imgpaths), 150, 150,1))

#Redifininf the opening of the images
for i,x in enumerate(imgpaths):
    #read the image in a vector
    img = mpimg.imread(x)
    #downsampling the image
#    img = downsample_image(img, 3)
    #finding edge
    img = find_vertical_edges(img)
    #resize the image
    img = resize(img, (150, 150))
    #flatend the image to a vector
#    img_vec =  img.reshape((1, -1)) 
#    if (img_vec.shape[1] != n_features):
#        print('image not procesed: %s' %x)
#    else:
        #store the image 
    images[i] = img.reshape(150,150,1)
    
y1 = [(0,0,1) for x in range(len(imgpathsTrash))]
y2 = [(0,1,0) for x in range(len(imgpathsGrafiti))]
y3 = [(1,0,0) for x in range(len(imgpathsOther))]

y= np.concatenate((y1,y2,y3), axis=0)

plt.imshow(images[1].reshape(150,150))

##############################################################################
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import random as rn

def guidoBatch(maximo, images, y):
    rango = range(0,maximo)
    a = rn.sample(rango , 100)
    return (images[a,:], y[a])

maximo = len(imgpaths)
    
# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 150, 150, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 3])
# weights W[784, 10]   784=28*28
W = tf.Variable(tf.zeros([22500, 3]))
# biases b[10]
b = tf.Variable(tf.zeros([3]))

# flatten the images into a single line of pixels
# -1 in the shape definition means "the only possible dimension that will preserve the number of elements"
XX = tf.reshape(X, [-1, 22500])

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
    batch_X, batch_Y = guidoBatch(maximo, images, y)
    train_data={X: batch_X, Y_: batch_Y}

    # train
    sess.run(train_step, feed_dict=train_data)

a,c = sess.run([accuracy, cross_entropy], feed_dict=train_data)

test_data={X: mnist.test.images, Y_: mnist.test.labels}
a,c = sess.run([accuracy, cross_entropy], feed_dict=test_data)