###############
#
# Created by Long Pham
#
# This python script allows the user to find images from Google, 
# parse the URL and then download the image from a simple query
# and save it to a drive. The user will be able to select the simple query
# like "Dog" in Google Search to produce the dataset. 
# I'm hoping that I will be able to select up to 100 or more images
#
###############


# Import Libraries







####################################################################
# 
# Using this python project "google image downloader", 
# we can scrape images from a simple google search query
#
# Ex.
#     python3 google-images-download.py --keywords "Polar bears, Beaches, ..." --limit 20
# https://pypi.org/project/google-images-download/1.0.1/
# git clone https://github.com/hardikvasa/google-images-download.git
#
####################################################################

################## IMPORTS / PACKAGES ##################
# Google Image Scraper
from google_images_download import google_images_download

# Keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

# OpenCV (Image Pre-processing)
import cv2


# Other Libraries
import os
import sys
import numpy as np
import matplotlib.pyplot as plt




# Globals (for now)
get_images = False
image_dir = './downloads'
categories = ['dog','cat']
IMG_SIZE = 128



################## CREATE DATABASE ##################
#TODO: ALLOW USER TO INPUT DATA OR NOT. IF SO, WHAT TYPE? FILL IN KEYWORDS

def get_images():
	if(get_images):
		response = google_images_download.googleimagesdownload()   #class instantiation

		# Change your query, query size and file types. 
		arguments = {'keywords':categories, # [item 1, item 2, item 3...]
						'limit':100, # over 100 requires selenium / chromedriver (extra)
						'format':'jpg',
						'safe_search':True,
						}   
		paths = response.download(arguments)   #passing the arguments to the function
		# print(paths)   #printing absolute paths of the downloaded images


################## AUGMENT IMAGES ##################
#TODO: ALLOW USER TO AUGMENT OR NOT. (RECOMMENDED IF <100 IMAGES)

training_data = []
# def normalize_data():
for category in categories:
	path = os.path.join(image_dir, category)
	class_num = categories.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat
	for img in os.listdir(path):
		img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
		new_img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
		training_data.append([new_img_array, class_num])
		print(len(training_data))

		#plt.imshow(new_img_array, cmap='gray')
		#plt.show()
inputd=input()
exit()

################## KERAS ENVIRONMENT ##################

# Load dataset
def load_data(num_classes):
    # load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # reshape dataset to feed it into model properly
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # normalize train/test data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0
    
    # convert class vectors to matrices as binary
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    print('Number of train samples in MNIST: ', x_train.shape[0])
    print('Number of test samples in MNIST: ', x_test.shape[0])
    
    return (x_train, y_train), (x_test, y_test)

# Build AlexNet-like model
def build_model(init_method, input_shape, first_layer_filter_size, dropout, num_classes):
    model = Sequential()
     
    # Convolution Layer 1
    model.add(Conv2D(first_layer_filter_size, kernel_size=(5,5), padding='same',
                     kernel_initializer=init_method, 
                     bias_initializer='zeros', 
                     input_shape=input_shape))
    model.add(Activation('relu'))
              
    # Convolution Layer 2
    model.add(Conv2D(96, kernel_size=(1,1), padding='same', 
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
              
    # Convolution Layer 3
    model.add(Conv2D(192, kernel_size=(5,5), padding='same', 
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
              
    # Convolution Layer 4
    model.add(Conv2D(192, kernel_size=(1,1), padding='same',
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    
    # Convolution Layer 5
    model.add(Conv2D(192, kernel_size=(3,3), padding='same',
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
    
    # Convolution Layer 6
    model.add(Conv2D(192, kernel_size=(1,1), padding='same',
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
    
    # Convolution Layer 7
    model.add(Conv2D(10, kernel_size=(1,1), padding='same',
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
    
    model.add(AveragePooling2D(pool_size=(6,6)))
    
    model.add(Flatten())

    model.add(Dropout(dropout))
    
    # Dense layer 3 (fc8)
    model.add(Dense(num_classes, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Activation('softmax'))

    return model



# def main():
# 	normalize_data()
