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
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import sgd
from keras.optimizers import Adam
from keras import backend as K
import keras

# OpenCV (Image Pre-processing)
import cv2


# Other Libraries
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from time import time

# TensorBoard
# from tensorflow.keras.callbacks import TensorBoard
# import tensorflow as tf

# Globals (for now)
get_images = False
image_dir = './downloads'
query = 'brown dog,cute cat'
IMG_SIZE = 128

init_method = 'glorot_uniform'
input_shape = (IMG_SIZE,IMG_SIZE,1)
first_layer_filter_size = 32
dropout = 0.5
num_classes = 2 
learning_rate = 0.05
momentum = 0.9
learning_rate_decay = 0.0005
batch_size = 16
validation_split = 0.2
loss_type = 'binary_crossentropy'
epochs = 50
num_tests = 2

################## CREATE DATABASE ##################
#TODO: ALLOW USER TO INPUT DATA OR NOT. IF SO, WHAT TYPE? FILL IN KEYWORDS

def get_images():
	response = google_images_download.googleimagesdownload()   #class instantiation

	# Change your query, query size and file types. 
	arguments = {'keywords':query, # [item 1, item 2, item 3...]
					'limit':100, # over 100 requires selenium / chromedriver (extra)
					'format':'jpg',
					'safe_search':True,
					}   
	paths = response.download(arguments)   #passing the arguments to the function
	# print(paths)   #printing absolute paths of the downloaded images

################## PRE-PROCESS IMAGES ##################
#TODO: ALLOW USER TO AUGMENT OR NOT. (RECOMMENDED IF <100 IMAGES)
def create_training_set():

	categories = ['dog', 'cat']
	# categories = query.split(',')
	training_data = []
	for category in categories:
		path = os.path.join(image_dir, category)
		class_num = categories.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat
		for img in os.listdir(path):
			try:
				img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
				new_img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
				training_data.append([new_img_array, class_num])
			
			except Exception as e:  # in the interest in keeping the output clean...
				pass
			#plt.imshow(new_img_array, cmap='gray')
			#plt.show()
	# print(training_data)
	# print(len(training_data))
	random.shuffle(training_data)

	X = [] # create image data
	y = [] # label

	for features, label in training_data:
		X.append(features)
		y.append(label)
	X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
	X = X/255.0

	y = keras.utils.to_categorical(y, num_classes)

	return(X,y)

################## KERAS ENVIRONMENT ##################

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
              
    # # Convolution Layer 4
    # model.add(Conv2D(192, kernel_size=(1,1), padding='same',
    #                  kernel_initializer='glorot_uniform', 
    #                  bias_initializer='zeros'))
    # model.add(Activation('relu'))
    
    # model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    
    # # Convolution Layer 5
    # model.add(Conv2D(192, kernel_size=(3,3), padding='same',
    #                  kernel_initializer='glorot_uniform', 
    #                  bias_initializer='zeros'))
    # model.add(Activation('relu'))
    
    # # Convolution Layer 6
    # model.add(Conv2D(192, kernel_size=(1,1), padding='same',
    #                  kernel_initializer='glorot_uniform', 
    #                  bias_initializer='zeros'))
    # model.add(Activation('relu'))
    
    # # Convolution Layer 7
    # model.add(Conv2D(10, kernel_size=(1,1), padding='same',
    #                  kernel_initializer='glorot_uniform', 
    #                  bias_initializer='zeros'))
    # model.add(Activation('relu'))
    
    model.add(AveragePooling2D(pool_size=(6,6)))
    
    model.add(Flatten())

    model.add(Dropout(dropout))
    
    # Dense layer 3 (fc8)
    model.add(Dense(num_classes, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Activation('softmax'))

    return model


def evaluate_model(history):
	# test data
	# save data?

	return

def plot_data(history, count):
	# save the data per epoch
	loss = history.history['loss']
	accuracy = history.history['acc']
	validation_loss = history.history['val_loss']
	validation_accuracy = history.history['val_acc']
	epochs = range(1,len(history.history['acc']) + 1)

	print('Printing Accuracy and Loss...')
	# plot the accuracy
	figure, ax = plt.subplots()
	ax.plot(epochs, accuracy, '-b', label='training accuracy')
	ax.plot(epochs, validation_accuracy, '-r', label='validation accuracy')
	ax.set_title('Training Accuracy and Validation Accuracy')
	ax.set_xlabel('Epochs (#)')
	ax.set_ylabel('Accuracy (%)')
	ax.legend()
	plt.ylim(bottom=0)
	plt.savefig(str(count) + '_accuracy' + '.png')
	plt.close()

    # plot the loss
	figure, ax = plt.subplots()
	ax.plot(epochs, loss, '-b', label='training loss')
	ax.plot(epochs, validation_loss, '-m', label='validation loss')
	ax.set_title('Training Loss and Validation Loss')
	ax.set_xlabel('Epochs (#)')
	ax.set_ylabel('Loss (%)')
	ax.legend()
	plt.ylim(bottom=0)
	plt.savefig(str(count) + '_loss' + '.png')
	plt.close()


def main():

	# if(get_images):
	# 	get_images()
	# 	exit()

	# Load Images
	print('Creating Dataset...')
	images, labels = create_training_set()

	# Create Keras Model
	print('Building model(s)...')
	model = build_model(init_method, input_shape, first_layer_filter_size, dropout, num_classes)
	
	# Compile uniform model
	print('Compiling model(s)...')
	
	#optimizer = sgd(learning_rate, momentum, learning_rate_decay)
	# optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	# tf.global_variables_initializer()

	model.compile(optimizer='adam', loss=loss_type, metrics=['accuracy'])



	for i in range(num_tests):
		try:
			print('Training model(s)...')
			history = model.fit(images,labels,
					batch_size=batch_size,
					epochs=epochs,
					validation_split=validation_split,
					shuffle=True)#,
					#callbacks=[tensorboard])

			# Evaluate Model
			print('Evaluating model(s)...')
			evaluate_model(history)

			# Plot Model
			plot_data(history, i)



		except KeyboardInterrupt:
			print('user input to end')
			K.clear_session()

if __name__ == "__main__":
	main()	