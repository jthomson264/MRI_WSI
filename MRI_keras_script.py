#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MRI_keras_script

Train a Conv3d Network in Keras to Classify Tumors Classes From MRI Alone

@author: ubuntr
"""
DataAugmentationFlag = True


from datetime import datetime
import talos
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Conv3D, BatchNormalization, Activation, MaxPooling3D, Flatten, Dropout, Add, Input
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

if DataAugmentationFlag == True:
    TrainIDs_withAugmentation = []
    for TrainID in list(TrainIDs):
    	TrainID_flipped = TrainID+"_flipped"
    	TrainIDs_withAugmentation.append(TrainID)
    	TrainIDs_withAugmentation.append(TrainID_flipped)
    	
    TrainIDs = TrainIDs_withAugmentation

partition = {"train": list(TrainIDs) , "validation": list(ValIDs)}

#Generate Dictionary of Class Labels  
labels = {}
for index in range(0,len(IDs)):
    ID = IDs[index]
    ID_flipped = ID+"_flipped"
    Y = Y_class[index]
    label = np.nan
    if Y == 'A':
        label = 0
    elif Y == 'O':
        label = 1
    elif Y == 'G':
        label = 2
    labels[ID] = label
    labels[ID_flipped] = label


# Generators


# Design model
training_generator = DataGenerator(partition['train'], labels, **paramz)
validation_generator = DataGenerator(partition['validation'], labels, **paramz)
def MRIClassifier(x, y, valX, valY, params):
	training_generator = DataGenerator(partition['train'], labels, **paramz)
	validation_generator = DataGenerator(partition['validation'], labels, **paramz)
    
	inputs = Input(shape=(240,240,155,4))
	output_1 = Conv3D(params['first'], (3,3,3), strides=(1, 1, 1), padding='valid')(inputs)
	output_2 = BatchNormalization()(output_1)
	output_3 = Activation('relu')(output_2)
	output_4 = MaxPooling3D(pool_size=(2, 2, 2))(output_3)
	output_5 = Conv3D(params['second'], (3,3,3), strides=(1, 1, 1), padding='same')(output_4)
	output_6 = BatchNormalization()(output_5)
	output_7 = Activation('relu')(output_6)
	output_8 = Add()([output_7, output_4])
	output_9 = Conv3D(params['third'], (3,3,3), strides=(1, 1, 1), padding='same')(output_8)
	output_10 = BatchNormalization()(output_9)
	output_11 = Activation('relu')(output_10)
	output_12 = Add()([output_11, output_8])
	output_13 = MaxPooling3D(pool_size=(2, 2, 2))(output_12)
	output_14 = Conv3D(params['fourth'], (3,3,3), strides=(1, 1, 1), padding='same')(output_13)
	output_15 = BatchNormalization()(output_14)
	output_16 = Activation('relu')(output_15)
	output_17 = MaxPooling3D(pool_size=(2, 2, 2))(output_16)
	output_18 = Flatten()(output_17)
	output_19 = Dense(params['last'], activation='relu')(output_18)
	predictions = Dense(3, activation='softmax')(output_19)
	loss_fn = CategoricalCrossentropy(from_logits=True)
	model = Model(inputs=inputs, outputs=predictions)
	model.summary()
	model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

#	model = Sequential([
#	    Conv3D(params['first'], (3,3,3), strides=(1, 1, 1), padding='valid', input_shape = (240,240,155,4)),
#	    BatchNormalization(),
#	    c,
#	    MaxPooling3D(pool_size=(2, 2, 2)),
#	    Conv3D(params['second'], (3,3,3), strides=(1, 1, 1), padding='valid'),#, kernel_initializer='he_normal')),
#	    BatchNormalization(),
#	    Activation('relu'),
#	    MaxPooling3D(pool_size=(2, 2, 2)),
#	    Conv3D(params['third'], (3,3,3), strides=(1, 1, 1), padding='valid'),#, kernel_initializer='he_normal')),
#	    BatchNormalization(),
#	    Activation('relu'),
#	    MaxPooling3D(pool_size=(3, 3, 3)),
#	    Conv3D(params['fourth'], (3,3,3), strides=(1, 1, 1), padding='valid'),#, kernel_initializer='he_normal')),
#	    BatchNormalization(),
#	    Activation('relu'),
#	    MaxPooling3D(pool_size=(3, 3, 3)),
#	    Flatten(),
#        #tf.keras.layers.Dropout(params['dropout']),
#	    Dense(params['last'], activation='relu'),
#        tf.keras.layers.Dropout(params['dropout']),
#	    Dense(3, activation = 'softmax') 
#	    ])
#	model.summary()
#	loss_fn = CategoricalCrossentropy(from_logits=True)
#	model.compile(optimizer='adam',
#		      loss=loss_fn,
#		      metrics=['accuracy'])


#model.compile()

	# Train model on dataset
	H = model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs = 1)
	
    # Save a summary of the model 
	nowString = now.strftime("%m_%d_%Y_%H_%M_%S")
	with open(nowString + '_ModelSummary.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
		model.summary(print_fn=lambda x: fh.write(x + '\n'))
    # Save the Training History  
	hist_df = pd.DataFrame(H.history) 
	with open(nowString + '_ModelTrainHistory.txt','w') as f:
		hist_df.to_csv(f)
        
	return H, model
p = {
	'first' : [6],
	'second' : [6],
	'third' : [6],
	'fourth' : [6],
    'last' : [6],
    'dropout': [.15, .0001],
    'learning_rate' : [.01, .001, .1]
        }

dummyX,dummyY=training_generator.__getitem__(0)
testX,testY=validation_generator.__getitem__(0)

now = datetime.now()
nowString = now.strftime("%m_%d_%Y_%H_%M_%S")
ScanObject = talos.Scan(x = dummyX, y=dummyY, x_val=testX, y_val=testY, model = MRIClassifier, params = p, experiment_name = 'MRI3D')

hist_df = pd.DataFrame(ScanObject.data) 
hist_csv_file = nowString + '_talosData' + '.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
