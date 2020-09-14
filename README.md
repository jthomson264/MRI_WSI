# CPM-RadPath Challenge MICCAI 2020 
## This respository contains code entailing our attempt at the MICCAI 2020 Combined Radiology and Pathology Classification Challenge. More details about the challenge can be found here: https://miccai.westus2.cloudapp.azure.com/competitions/1

"The CPM-RadPath 2020 dataset consists of multi-institutional paired radiology scans and digitized histopathology
images of brain gliomas, obtained from the same patients, as well as their diagnostic classification label. Taking
into consideration the latest classification of CNS tumors, the classes used in the CPM-RadPath challenge are:
- A = Lower grade astrocytoma, IDH-mutant (Grade II or III)
- O = Oligodendroglioma, IDH-mutant, 1p/19q codeleted (Grade II or III)
- G = Glioblastoma and Diffuse astrocytic glioma with molecular features of glioblastoma, IDH-wildtype (Grade IV)."

## The code contained allows for the training and deployment of classifiers to predict classification label of CNS tumors based on MRI scans alone

The first script (SaveAndNormalizeMRIs.py) contained here allows for the loading and normalization of the radiology scans (T1, T2, T1-contrast enhanced, and flair images); this data is then saved in a new format (.NPY binary file) to allow for rapid loading. The second script (trainKerasModel.py) defines, trains, and deploys a 3D Convolutional Neural Network with a custom data generator.

# Installation/Setup:
First install Python3, clone master, & ensure you have pip package manager. Next, ensure that you install all the following dependencies: tensorflow 2.0 (pip install tensorflow), pandas (pip install pandas), nibabel (pip install nibabel), matplotlib (pip install matplotlib), tqdm (pip install tqdm)

Next, run the following bash commands to download the Radiology dataset from the MICCAI Challenge (~2.0GB):

wget -r --no-parent --reject "index.html*" http://miccai2020-data.eastus.cloudapp.azure.com/CPM-RadPath_2020_Training_Data/Radiology/

wget http://miccai2020-data.eastus.cloudapp.azure.com/CPM-RadPath_2020_Training_Data/training_data_classification_labels.csv

# Next, ensure that the filepath defined in "Save_MRIs_into_npy_files.py" points to the folder that contains the newly downloaded CPM-RadPath dataset and run the script with the following command; also, choose the appropriate augmentation and testing settings at the top of the file.  You may need to create a folder named "data" in the working directory (Warning: data folder will be 126GB with augmentation enabled).

python Save_MRIs_into_npy_files.py

# Finally, you can run the following command to instantiate and train the Keras classifier

python MRI_keras_script.py
