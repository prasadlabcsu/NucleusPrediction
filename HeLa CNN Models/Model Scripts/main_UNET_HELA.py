# This is the main script for the entire VAE-GAN model creation and training.
# This is the script which should be called when training or defining VAE-GAN models.
# The other scripts are imported as modules which contain the bulk of the functions and classes which build
# the model, generate the batches of training and validation set data, and train the model parameters.

# Hyperparameters

seednum = 14       # Seed used in randomly shuffled datasets
n_epochs = 10      # Maximum number of training epochs
batch_size = 10    # Size of batches per iteration each epoch
k = 10             # Number of cross-folds for separating data sets and determining how many models to train
model_nums = [10]   # The model numbers to train in this session (should be between 1 and k) WARNING: Only train one model at a time or re-initialize the model and optimizer before starting the next.
group = "H11" # "H11" "Ble" "Y27" "CTL"

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

from utils_UNET_HELA import * # Imports everything from the utils module script, including imported packages
from augmentation_UNET_HELA import *
from losses_UNET_HELA import *
from models_UNET_HELA import *
from train_UNET_HELA import *

# Images lists
    # Each lists all the file paths for the respective image types and sorted by the image number by the function num_sort

Input_list = sorted(glob.glob('/nfs/home/*slawto/*NEW_HeLa_Images_TIFF/*' + group + '_Membrane_Mask*.tiff'),key=num_sort)
Nucleus_list = sorted(glob.glob('/nfs/home/*slawto/*NEW_HeLa_Images_TIFF/*' + group + '_Nucleus_Mask*.tiff'),key=num_sort)

# Create the k cross-folds training and validation sets

N = len(Nucleus_list)       # Total sample size
ImNums = np.arange(N) + 1 # List of image numbers

ImNums_Shuffled = copy.deepcopy(ImNums) # This block randomly shuffles the image numbers according the seed given
random.seed(seednum)
random.shuffle(ImNums_Shuffled)

V = math.floor(N/k) # Validation set size
Vf = V + N - V*k    # Final validation set size from rounding

Valid_list = [[0] * N] * 3 # Table with the sample number, image number, and validation set number for each of the cross-fold models
Valid_sets = np.array(Valid_list)
for n in ImNums:
    Valid_sets[0,n-1] = n
    Valid_sets[1,n-1] = ImNums_Shuffled[n-1]
    Valid_sets[2,n-1] = math.floor((n-1)/V) + 1
    if math.floor(n/V) + 1 > k:
        Valid_sets[2,n-1] = k

model_nums[0] -= 1   # Corrects to indexing by 0 instead of 1
for i in model_nums: # For each of the cross-fold models: ...

    # Create training and validation image dataset per fold

    path = "/nfs/home/slawto/RESULTS_UNET_NEW_HeLa" # Creates the RESULTS folder if none exists
    if os.path.exists(path)==False:
        os.mkdir(path)

    sets = {'train': [], 'valid': []} # Training and Validation sets are created from the list of input and nuclear image file paths

    for n in ImNums:
        if Valid_sets[2,n-1] == i+1:
            sets['valid'].append([Input_list[Valid_sets[1,n-1]-1], Nucleus_list[Valid_sets[1,n-1]-1]])
        else:
            sets['train'].append([Input_list[Valid_sets[1,n-1]-1], Nucleus_list[Valid_sets[1,n-1]-1]])
    
    sets_path = path + "/Sets_Group_" + group + "_Model_" + str(i+1) + ".txt" # Path in the RESULTS folder where the model sets will be saved
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
    h = fit(train_gen, valid_gen, n_epochs, i, group) # Calls the training function for the entire model
