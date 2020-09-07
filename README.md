# Set-up:
Using a Bash Command Line: 
First install Python3 & ensure you have pip package manager. Next, ensure that you install all the following dependencies:

 tensorflow 2.0
pip install tensorflow
 pandas
pip install pandas
 nibabel
pip install nibabel
 matplotlib
pip install matplotlib

# Next, run the following command in a new folder to download the Combined Radiology and Pathology dataset from the MICCAI Challenge:

wget -r --no-parent --reject "index.html*" http://miccai2020-data.eastus.cloudapp.azure.com/CPM-RadPath_2020_Training_Data/

# Next, ensure that the filepath defined in "Save_MRIs_into_npy_files.py" points to the folder that contains the newly downloaded CPM-RadPath dataset and run the script witht he following command.  You may need to create a folder named "data" in the working directory

python Save_MRIs_into_npy_files.py

# Finally, you can run the following command to instantiate and train the Keras classifier

python MRI_keras_script.py
