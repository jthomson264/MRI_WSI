#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make npy files for data generator use


@author: ubuntr
"""
## Choose Settings:
DataAugmentationFlag = True
TestingFlag = True

## Import Dependencies
from tqdm import tqdm
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

## Print Statements to Check Tensorflow GPU Connection
print(tf.version)
print(tf.config.list_physical_devices('GPU'))
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
from tensorflow.keras import backend
print(backend.image_data_format())

## Define Helper Functions

def show_slices(slices): # slice plotter
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")
      
def normalizeImage(array_to_be_normalized): # array normalizer
    return array_to_be_normalized/np.max(array_to_be_normalized)

## Read in Classifcation Labels
classLabels = pd.read_csv('/home/ubuntr/Desktop/Projects/MRI_WSI_MICCAI2020/CPM19/training_data_classification_labels.csv')
IDs = classLabels.CPM_RadPath_2019_ID
Y_class = classLabels['class']
#age_in_days = classLabels.age_in_days   # Not used yet
print(classLabels)

# Load the MRI data training set into X, and save each npy file

count_t1 = 0
count_t2 = 0
count_t1ce = 0
count_flair = 0
t1_max = 0
t2_max = 0
t1ce_max = 0
flair_max = 0
print('t1 count')

for row in tqdm(IDs): # iterate thru each patient
    print(count_t1, end = ' ')
    fileRoot = "/home/ubuntr/Desktop/Projects/MRI_WSI_MICCAI2020/miccai2020-data.eastus.cloudapp.azure.com/CPM-RadPath_2020_Training_Data/Radiology/"+str(row)
    fileRoot = fileRoot + "/" + str(row)
    
    try:
        img_t1 = nib.load(fileRoot+"_t1.nii.gz")
        count_t1 += 1
        #t1_max = np.max(t1_max,np.max(img_t1.get_fdata()))
        #X1 += [img_t1.get_fdata()]
    except:
        print("no t1 file found @ "+ fileRoot+"_t1.nii.gz")


    try:
        img_t2 = nib.load(fileRoot+"_t2.nii.gz")
        count_t2 += 1
        #t2_max = np.max(t2_max,np.max(img_t2.get_fdata()))
        #X2 += [img_t2.get_fdata()]
    except:
        print("no t2 file found @ "+ fileRoot+"_t2.nii.gz")

        
    try:
        img_t1ce = nib.load(fileRoot+"_t1ce.nii.gz")
        count_t1ce += 1
        #t1ce_max = np.max(t1ce_max,np.max(img_t1ce.get_fdata()))
        #Xc += [img_t1ce.get_fdata()]
    except:
        print("no t1ce file found @ "+ fileRoot+"_t1ce.nii.gz")
        
    try:
        img_flair = nib.load(fileRoot+"_flair.nii.gz")
        count_flair += 1
        #flair_max = np.max(flair_max,np.max(img_flair.get_fdata()))
        #Xf += [img_flair.get_fdata()]
    except:
        print("no flair file found @ "+ fileRoot+"_flair.nii.gz")
    
    X = np.stack((img_t1.get_fdata(),img_t2.get_fdata(),img_t1ce.get_fdata(),img_flair.get_fdata()),axis=-1)
    np.save("/home/ubuntr/Desktop/Projects/MRI_WSI_MICCAI2020/MRI_WSI/data/" + str(row)+".npy", X)
    
    if DataAugmentationFlag:
        X_flipped = np.stack((np.flip(img_t1.get_fdata(), axis=0),np.flip(img_t2.get_fdata(), axis=0),np.flip(img_t1ce.get_fdata(), axis=0),np.flip(img_flair.get_fdata(), axis=0)),axis=-1)
        np.save("/home/ubuntr/Desktop/Projects/MRI_WSI_MICCAI2020/MRI_WSI/data/" + str(row)+"_flipped.npy", X_flipped)
    
    if TestingFlag:
        anat_img_data = img_t1.get_fdata()
        flipped_img_data = np.flip(anat_img_data, axis=0)
        show_slices([anat_img_data[100, :, :], anat_img_data[:, 100, :], anat_img_data[:, :, 100]])
        plt.suptitle("Center slices for anatomical image")  
        show_slices([flipped_img_data[100, :, :], flipped_img_data[:, 100, :], flipped_img_data[:, :, 100]])
        plt.suptitle("Center slices for anatomical image")
        plt.show()
