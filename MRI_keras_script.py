#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MRI_keras_script

Train a Conv3d Network in Keras to Classify Tumors Classes From MRI Alone

@author: ubuntr
"""

import talos
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv3D, BatchNormalization, Activation, MaxPooling3D, Flatten, Dropout
from keras.losses import CategoricalCrossentropy
from my_classes import DataGenerator

# Parameters
paramz = {'dim': (240, 240, 155),
          'batch_size': 4,
          'n_classes': 3,
          'n_channels': 4,
          'shuffle': True}

# Load in Dataset Info
# Read in classification labels
classLabels = pd.read_csv('/home/ubuntr/Desktop/Projects/MRI_WSI_MICCAI2020/CPM19/training_data_classification_labels.csv')
IDs = classLabels.CPM_RadPath_2019_ID
Y_class = classLabels['class']
age_in_days = classLabels.age_in_days

#Divide Data into Training and Validation Using Masks
percent_of_data_to_train = .8 #Size of training set (percent)
random_indicies = np.random.rand(len(Y_class))

data_mask = random_indicies < percent_of_data_to_train
val_mask = ~data_mask

ValIDs = IDs[val_mask]
TrainIDs = IDs[data_mask]

partition = {"train": list(TrainIDs) , "validation": list(ValIDs)}

#Generate Dictionary of Class Labels  
labels = {}
for index in range(0,len(IDs)):
    ID = IDs[index]
    Y = Y_class[index]
    label = np.nan
    if Y == 'A':
        label = 0
    elif Y == 'O':
        label = 1
    elif Y == 'G':
        label = 2
        
    labels[ID] = label


# Generators


# Design model
training_generator = DataGenerator(partition['train'], labels, **paramz)
validation_generator = DataGenerator(partition['validation'], labels, **paramz)
def MRIClassifier(x, y, valX, valY, params):
	training_generator = DataGenerator(partition['train'], labels, **paramz)
	validation_generator = DataGenerator(partition['validation'], labels, **paramz)
	model = Sequential([
	    Conv3D(params['first'], (3,3,3), strides=(1, 1, 1), padding='valid', input_shape = (240,240,155,4)),
	    BatchNormalization(),
	    Activation('relu'),
	    MaxPooling3D(pool_size=(2, 2, 2)),
	    Conv3D(params['second'], (3,3,3), strides=(1, 1, 1), padding='valid'),#, kernel_initializer='he_normal')),
	    BatchNormalization(),
	    Activation('relu'),
	    MaxPooling3D(pool_size=(2, 2, 2)),
	    Conv3D(params['third'], (3,3,3), strides=(1, 1, 1), padding='valid'),#, kernel_initializer='he_normal')),
	    BatchNormalization(),
	    Activation('relu'),
	    MaxPooling3D(pool_size=(3, 3, 3)),
	    Conv3D(params['fourth'], (3,3,3), strides=(1, 1, 1), padding='valid'),#, kernel_initializer='he_normal')),
	    BatchNormalization(),
	    Activation('relu'),
	    MaxPooling3D(pool_size=(3, 3, 3)),
	    Flatten(),
	    Dense(params['last'], activation='relu'),
	    #tf.keras.layers.Dropout(0.2),
	    Dense(3, activation = 'softmax') 
	    ])
	model.summary()
	loss_fn = CategoricalCrossentropy(from_logits=True)
	model.compile(optimizer='adam',
		      loss=loss_fn,
		      metrics=['accuracy'])


#model.compile()

	# Train model on dataset
	H = model.fit_generator(generator=training_generator,
		            validation_data=validation_generator,
		            #use_multiprocessing=True,
		            #workers=6,
		            #verbose=2,
		            epochs = 100)
	return H, model
p = {
	'first' : [5, 6, 7],
	'second' : [5, 6, 7],
	'third' : [5, 6, 7],
	'fourth' : [5, 6, 7],
	'last' : [4, 8, 12,16]
}

dummyX,dummyY=training_generator.__getitem__(0)
testX,testY=validation_generator.__getitem__(0)


t = talos.Scan(x = dummyX, y=dummyY, x_val=testX, y_val=testY, model = MRIClassifier, params = p, experiment_name = 'MRI3D')
hist_df = pd.DataFrame(t) 
hist_csv_file = 'talosHistory.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
