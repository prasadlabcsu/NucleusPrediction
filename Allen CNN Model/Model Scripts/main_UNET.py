# This is the main script for the entire VAE-GAN model creation and training.
# This is the script which should be called when training or defining VAE-GAN models.
# The other scripts are imported as modules which contain the bulk of the functions and classes which build
# the model, generate the batches of training and validation set data, and train the model parameters.

# Hyperparameters

Actin_Flag = 0     # Flag to use the images including actin as input
Membrane_Flag = 1  # Flag to use the images including the membrane as input. Both flags will use the combined actin and membrane images
Bounded_Flag = 0   # Flag to use the images which include actin bounded by the cell membrane, otherwise the entire actin image is used
seednum = 14       # Seed used in randomly shuffled datasets
n_epochs = 20      # Maximum number of training epochs
batch_size = 10    # Size of batches per iteration each epoch
k = 10             # Number of cross-folds for separating data sets and determining how many models to train
model_nums = [9]   # The model numbers to train in this session (should be between 0 and k-1) WARNING: Only train one model at a time or re-initialize the model and optimizer before starting the next.

# Importing packages and modules

    # Necessary packages for the environment: 
        # numpy
        # tensorflow
            # cuda (only if running on GPUs)
        # imageio
        # scipy
        # elasticdeform

import numpy as np
import glob
import random
import copy
import math
import os

from utils_UNET import * # Imports everything from the utils module script, including imported packages
from augmentation_UNET import *
from losses_UNET import *
from models_UNET import *
from train_UNET import *

# Images lists
    # Each lists all the file paths for the respective image types and sorted by the image number by the function num_sort

Actin_list = sorted(glob.glob('/nfs/home/*slawto/*Actin_Images_TIFF/*Actin_Standard*.tiff'),key=num_sort)
Nucleus_list = sorted(glob.glob('/nfs/home/*slawto/*Actin_Images_TIFF/*Nucleus_Standard*.tiff'),key=num_sort)
Membrane_list = sorted(glob.glob('/nfs/home/*slawto/*Actin_Images_TIFF/*Membrane_Mask_Standard*.tiff'),key=num_sort)
A_M_list = sorted(glob.glob('/nfs/home/*slawto/*Actin_Images_TIFF/*A_M_Standard*.tiff'),key=num_sort)
Bounded_list = sorted(glob.glob('/nfs/home/*slawto/*Actin_Images_TIFF/*A_M_Bounded_Standard*.tiff'),key=num_sort)

# Create the k cross-folds training and validation sets

N = len(Actin_list)       # Total sample size (3824)
ImNums = np.arange(N) + 1 # List of image numbers from 1 to 3824

ImNums_Shuffled = copy.deepcopy(ImNums) # This block randomly shuffles the image numbers according the seed given
random.seed(seednum)
random.shuffle(ImNums_Shuffled)

V = math.floor(N/k) # Validation set size (382 with 10-fold cross validation)
Vf = V + N - V*k    # Final validation set size from rounding

Valid_list = [[0] * N] * 3 # Table with the sample number, image number, and validation set number for each of the cross-fold models
Valid_sets = np.array(Valid_list)
for n in ImNums:
    Valid_sets[0,n-1] = n
    Valid_sets[1,n-1] = ImNums_Shuffled[n-1]
    Valid_sets[2,n-1] = math.floor((n-1)/V) + 1
    if math.floor(n/V) + 1 > k:
        Valid_sets[2,n-1] = k

# Choose appropriate input image data

if Bounded_Flag == 1:
    Input_list = Bounded_list # Input images are the bounded actin images if the Bounded_Flag is 1
else:
    if Actin_Flag == 1:
        if Membrane_Flag == 1:
            Input_list = A_M_list # Input images are the actin and membrane images if the Actin_Flag and Membrane_Flag are both 1
        else:
            Input_list = Actin_list # Input images are the actin images if the Actin_Flag is the only flag set to 1
    else:
        Input_list = Membrane_list # Input images are the membrane images if all other flags are not 1
         
for i in model_nums: # For each of the cross-fold models: ...

    # Create training and validation image dataset per fold

    path = "/nfs/home/slawto/RESULTS_UNET" # Creates the RESULTS folder if none exists
    if os.path.exists(path)==False:
        os.mkdir(path)

    sets = {'train': [], 'valid': []} # Training and Validation sets are created from the list of input and nuclear image file paths

    for n in ImNums:
        if Valid_sets[2,n-1] == i+1:
            sets['valid'].append([Input_list[Valid_sets[1,n-1]-1], Nucleus_list[Valid_sets[1,n-1]-1]])
        else:
            sets['train'].append([Input_list[Valid_sets[1,n-1]-1], Nucleus_list[Valid_sets[1,n-1]-1]])
    
    sets_path = path + "/Sets_Model_" + str(i+1) + ".txt" # Path in the RESULTS folder where the model sets will be saved
    valid_str = str(sets['valid'])
    train_str = str(sets['train'])
        
    with open(sets_path, 'w') as f: # Creates the Sets_Model_# file
        pass  
    
    with open(sets_path, 'a') as f: # Saves the training and validation sets of image file paths for reference
        f.write('Validation Set:' + valid_str + '\n')
        f.write('=============================================================== \n')
        f.write('Training Set:' + train_str + '\n')

    train_gen = DataGenerator(sets['train'], batch_size=batch_size, augmentation=True) # Creates the training and validation data generator classes for the models to call when training
    valid_gen = DataGenerator(sets['valid'], batch_size=batch_size, augmentation=True)
    
    # Train the vox2vox model
    h = fit(train_gen, valid_gen, n_epochs, i) # Calls the training function for the entire model
