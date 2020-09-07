#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make npy files for data generator use


@author: ubuntr
"""

import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

print(tf.version)

print(tf.config.list_physical_devices('GPU'))

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

from tensorflow.keras import backend
print(backend.image_data_format())

# Read in classifcation labels
classLabels = pd.read_csv('/home/ubuntr/Desktop/Projects/MRI_WSI_MICCAI2020/CPM19/training_data_classification_labels.csv')
#val_classLabels = pd.read_csv('"H:\MRI_WSI_MICCAI2020\miccai2020-data.eastus.cloudapp.azure.com\CPM-RadPath_2020_Validation_Data\Radiology')
IDs = classLabels.CPM_RadPath_2019_ID
Y_class = classLabels['class']
age_in_days = classLabels.age_in_days
print(classLabels)

# Load the MRI data training set into X, and save each npy file
count_t1 = 0
count_t2 = 0
count_t1ce = 0
count_flair = 0



print('t1 count')

for row in IDs:#[data_mask]:
    
    fileRoot = "/home/ubuntr/Desktop/Projects/MRI_WSI_MICCAI2020/miccai2020-data.eastus.cloudapp.azure.com/CPM-RadPath_2020_Training_Data/Radiology/"+str(row)
   
    fileRoot = fileRoot + "/" + str(row)
    
    
    
    try:
        img_t1 = nib.load(fileRoot+"_t1.nii.gz")
        count_t1 += 1

        #X1 += [img_t1.get_fdata()]
    except:
        print("no t1")

    try:
        img_t2 = nib.load(fileRoot+"_t2.nii.gz")
        count_t2 += 1

        #X2 += [img_t2.get_fdata()]
    except:
        print("no t2 @ "+ fileRoot+"_t2.nii.gz")
        
    try:
        img_t1ce = nib.load(fileRoot+"_t1ce.nii.gz")
        count_t1ce += 1

        #Xc += [img_t1ce.get_fdata()]
    except:
        print("no t1ce")
        
    try:
        img_flair = nib.load(fileRoot+"_t1ce.nii.gz")
        count_flair += 1

        #Xf += [img_flair.get_fdata()]
    except:
        print("no t1ce")
    X = np.stack((img_t1.get_fdata(),img_t2.get_fdata(),img_t1ce.get_fdata(),img_flair.get_fdata()),axis=-1)
    np.save("/home/ubuntr/Desktop/Projects/MRI_WSI_MICCAI2020/data/" + str(row)+".npy", X)
    print(count_t1, end = ' ')